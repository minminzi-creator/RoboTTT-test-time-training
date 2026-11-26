# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Bounded eight-rank acceptance; opt-in observers leave training outputs intact."""

import argparse
import functools
import hashlib
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY = Path("/mnt/public/chenzm/robottt/h800_sft/code/Isaac-GR00T")
PYTHON = "/mnt/public/chenzm/robottt/train_gr00t_baseline_20261016/gr00t_baseline_origin_2e/resume_runtime/layout/env/gr00t-baseline-env-20261012/bin/python"
SCRIPT = "examples/sft/run_robottt3_acceptance.py"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(values):
    """Hash local shards, including names, shapes and dtype; consume no RNG."""
    import torch

    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            flattened = tensor.reshape(-1)
            if flattened.stride(0) != 1:
                flattened = flattened.clone(memory_format=torch.contiguous_format)
            digest.update(flattened.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def tree_digest(value):
    """Hash actual nested inputs without changing tensors or consuming RNG."""
    from collections.abc import Mapping

    def leaves(item, name="root"):
        if isinstance(item, Mapping):
            for key in sorted(item):
                yield from leaves(item[key], f"{name}/{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                yield from leaves(child, f"{name}/{index}")
        else:
            yield name, item

    return tensor_digest(leaves(value))


def rng_snapshot():
    import numpy as np
    import torch

    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": tensor_digest([("rng", torch.get_rng_state())]),
        "torch_cuda": tensor_digest([("rng", torch.cuda.get_rng_state())])
        if torch.cuda.is_initialized()
        else None,
    }


def parameter_layout(worker):
    """Read orig-parameter FQNs and local ownership; never unshard parameters."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp._common_utils import _get_param_to_fqns

    names = _get_param_to_fqns(worker.model, dedup_shared_params=False)
    shards = {}
    for module in worker.model.modules():
        if not isinstance(module, FSDP) or module._handle is None:
            continue
        flat = module._handle.flat_param
        fields = (flat._params, flat._shard_param_infos, flat._shapes, flat._numels)
        if len({len(field) for field in fields}) != 1:
            raise RuntimeError("FSDP orig-parameter layout fields differ in length")
        for parameter, info, shape, numel in zip(*fields):
            owned = info.numel_in_shard if info.in_shard else 0
            if parameter.numel() != owned:
                raise RuntimeError(
                    "Expected sharded orig parameters at observer boundary"
                )
            shards[id(parameter)] = {
                "shape": list(shape),
                "numel": numel,
                "local_numel": owned,
                "range": [info.intra_param_start_idx, info.intra_param_end_idx]
                if info.in_shard
                else None,
                "offset_in_shard": info.offset_in_shard if info.in_shard else None,
            }
    result = {}
    for group in worker.optimizer.param_groups:
        for parameter in group["params"]:
            aliases = sorted(names[parameter])
            if shards and id(parameter) not in shards:
                raise RuntimeError(f"Missing FSDP ownership metadata: {aliases}")
            layout = shards.get(
                id(parameter),
                {
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                    "local_numel": parameter.numel(),
                    "range": [0, parameter.numel() - 1] if parameter.numel() else None,
                    "offset_in_shard": 0 if parameter.numel() else None,
                },
            )
            result[parameter] = (aliases[0], {**layout, "aliases": aliases})
    return result


def state_description(state):
    import torch

    result = {}
    for key, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            item = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_digest([(key, value)]),
            }
            if value.numel() == 1:
                item["value"] = value.item()
            result[key] = item
        else:
            result[key] = value
    return result


def optimizer_snapshot(worker):
    layout = parameter_layout(worker)
    values, steps, groups = [], [], []
    nonempty, empty = {}, {}
    for index, group in enumerate(worker.optimizer.param_groups):
        groups.append(
            {
                **{key: value for key, value in group.items() if key != "params"},
                "params": [layout[p][0] for p in group["params"]],
            }
        )
        for offset, parameter in enumerate(group["params"]):
            name, shard = layout[parameter]
            state = worker.optimizer.state.get(parameter, {})
            destination = nonempty if parameter.numel() else empty
            destination[name] = {
                "layout": shard,
                "state_present": parameter in worker.optimizer.state,
                "state": state_description(state),
            }
            for key, value in sorted(state.items()):
                values.append((f"{index}/{offset}/{key}", value))
                if key == "step":
                    steps.append(float(value))
    return {
        "model_shard_sha256": tensor_digest(worker.model.named_parameters()),
        "optimizer_shard_sha256": tensor_digest(values),
        "optimizer_nonempty": nonempty,
        "optimizer_empty": empty,
        "param_groups": groups,
        "buffers_sha256": tensor_digest(worker.model.named_buffers()),
        "rng": rng_snapshot(),
        "adam_steps": sorted(set(steps)),
        "optimizer_steps": worker.optimizer_steps,
        "data_offset": worker._data_iter_offset,
        "scheduler": worker.lr_scheduler.state_dict(),
        "lr": [group["lr"] for group in worker.optimizer.param_groups],
    }


def gradient_snapshot(worker):
    return {
        name: None
        if parameter.grad is None
        else state_description({"grad": parameter.grad})
        for parameter, (name, _) in parameter_layout(worker).items()
        if parameter.numel()
    }


def make_worker_class():
    """Build the observed class before Ray exports its remote methods."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        FSDPGr00tSequenceSftWorker as Worker,
    )

    original_init = Worker.init_worker
    original_run = Worker.run_training
    original_batch = Worker.forward_backward_batch
    original_save = Worker.save_checkpoint
    original_load = Worker.load_checkpoint

    def emit(worker, event, **values):
        destination = Path(worker.cfg.runner.logger.log_path)
        record = {"event": event, "rank": worker._rank, "time": time.time(), **values}
        with (destination / f"acceptance_rank{worker._rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    @functools.wraps(original_init)
    def initialize(worker):
        original_init(worker)
        if worker._world_size != 8:
            raise ValueError("Acceptance requires eight real ranks")
        worker._acceptance_segments = []
        worker._acceptance_collectives = 0
        worker._acceptance_collective_elements = 0
        worker._acceptance_optimizer_calls = 0
        worker._acceptance_batch_saved = False
        worker._acceptance_forward_active = False

        def observe_forward(module, args, output):
            worker._acceptance_forward_active = False
            memory = output.get("memory") if isinstance(output, dict) else None
            if memory is None:
                raise RuntimeError("Expected candidate FastState at the model boundary")
            with torch.no_grad():
                norms = torch.stack(
                    [
                        torch.stack(
                            [part.detach().float().square().sum() for part in layer]
                        )
                        .sum()
                        .sqrt()
                        for layer in memory.layers
                    ]
                )
                maxima = torch.stack(
                    [
                        torch.stack([part.detach().abs().max() for part in layer]).max()
                        for layer in memory.layers
                    ]
                )
                worker._acceptance_segments.append(
                    {
                        "layer_l2": norms.cpu().tolist(),
                        "layer_max_abs": maxima.cpu().tolist(),
                        "position": memory.position.cpu().tolist(),
                        "ttt_update_rms": float(output["ttt_update_rms"].detach()),
                    }
                )

        worker.model.register_forward_hook(observe_forward)

        def observe_inputs(module, args, kwargs):
            worker._acceptance_forward_active = True
            emit(
                worker,
                "forward_input",
                step=worker.global_step + 1,
                segment=len(worker._acceptance_segments),
                data_sha256=tree_digest(kwargs["data"]),
                memory_sha256=tree_digest(kwargs.get("memory")),
                buffers_sha256=tensor_digest(worker.model.named_buffers()),
                rng=rng_snapshot(),
            )

        worker.model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
        encoders = [
            module
            for name, module in worker.model.named_modules()
            if name.endswith(".action_encoder")
        ]
        if len(encoders) != 1:
            raise RuntimeError(
                "Expected one native action encoder for input observation"
            )

        def observe_encoder(module, args):
            emit(
                worker,
                "action_input",
                step=worker.global_step + 1,
                input_sha256=tree_digest(args),
            )

        encoders[0].register_forward_pre_hook(observe_encoder)
        if os.environ.get("ROBOTTT_TRACE_FORWARD") == "1":
            classes = {
                "Qwen3Backbone",
                "Qwen3VLVisionPatchEmbed",
                "Qwen3VLVisionBlock",
                "Qwen3VLTextDecoderLayer",
                "Qwen3VLTextRotaryEmbedding",
                "NativeTTTBlock",
                "FastWeightTTT",
            }
            suffixes = (
                ".state_encoder",
                ".action_encoder",
                ".action_decoder",
                ".vlln",
                ".vl_self_attention",
            )
            for name, module in worker.model.named_modules():
                if type(module).__name__ not in classes and not name.endswith(suffixes):
                    continue

                def observe_boundary(module, args, kwargs, output, *, name=name):
                    if (
                        worker.global_step != 1
                        or worker._acceptance_segments
                        or not worker._acceptance_forward_active
                    ):
                        return
                    emit(
                        worker,
                        "module_boundary",
                        step=2,
                        name=name,
                        module_type=type(module).__name__,
                        training=module.training,
                        input_sha256=tree_digest((args, kwargs)),
                        output_sha256=tree_digest(output),
                    )

                module.register_forward_hook(observe_boundary, with_kwargs=True)
        original_reduce = dist.reduce_scatter_tensor

        @functools.wraps(original_reduce)
        def observe_reduce(output, input, *args, **kwargs):
            result = original_reduce(output, input, *args, **kwargs)
            worker._acceptance_collectives += 1
            worker._acceptance_collective_elements += input.numel()
            return result

        dist.reduce_scatter_tensor = observe_reduce
        original_clip = worker._strategy.clip_grad_norm_

        @functools.wraps(original_clip)
        def observe_clip(*args, **kwargs):
            emit(
                worker,
                "gradient_before_clip",
                step=worker.global_step + 1,
                parameters=gradient_snapshot(worker),
            )
            value = original_clip(*args, **kwargs)
            worker._acceptance_grad_norm = float(value)
            return value

        worker._strategy.clip_grad_norm_ = observe_clip

        def before_optimizer(optimizer, args, kwargs):
            emit(
                worker,
                "optimizer_before",
                step=worker.global_step + 1,
                snapshot=optimizer_snapshot(worker),
                gradients=gradient_snapshot(worker),
            )

        def after_optimizer(optimizer, args, kwargs):
            worker._acceptance_optimizer_calls += 1
            emit(
                worker,
                "optimizer_after",
                step=worker.global_step + 1,
                snapshot=optimizer_snapshot(worker),
            )

        worker.optimizer.register_step_pre_hook(before_optimizer)
        worker.optimizer.register_step_post_hook(after_optimizer)
        wrappers = [
            module for module in worker.model.modules() if isinstance(module, FSDP)
        ]
        strategies = sorted({str(module.sharding_strategy) for module in wrappers})
        if not wrappers or strategies != ["ShardingStrategy.FULL_SHARD"]:
            raise ValueError(f"Expected real FULL_SHARD FSDP, got {strategies}")
        emit(
            worker,
            "initialized",
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            flash_attention_deterministic=os.environ.get(
                "FLASH_ATTENTION_DETERMINISTIC"
            ),
            world_size=worker._world_size,
            optimizer=type(worker.optimizer).__name__,
            fsdp_wrappers=len(wrappers),
            sharding=strategies,
            parameters=worker._parameters,
        )

    @functools.wraps(original_batch)
    def batch(worker, data, *args, **kwargs):
        if worker._rank == 0 and not worker._acceptance_batch_saved:
            from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import segment_batch

            path = Path(worker.cfg.runner.logger.log_path) / "first_anchor.pt"
            with path.open("xb") as stream:
                torch.save(segment_batch(data, 0, 1, torch.device("cpu")), stream)
            worker._acceptance_batch_saved = True
        return original_batch(worker, data, *args, **kwargs)

    @functools.wraps(original_run)
    def run(worker):
        worker._acceptance_segments = []
        worker._acceptance_collectives = 0
        worker._acceptance_collective_elements = 0
        worker._acceptance_optimizer_calls = 0
        worker._acceptance_grad_norm = None
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = original_run(worker)
        torch.cuda.synchronize()
        emit(
            worker,
            "step",
            step=worker.global_step + 1,
            seconds=time.monotonic() - started,
            raw_grad_norm=worker._acceptance_grad_norm,
            optimizer_calls=worker._acceptance_optimizer_calls,
            optimizer_steps=worker.optimizer_steps,
            reduce_scatter_calls=worker._acceptance_collectives,
            reduce_scatter_input_elements=worker._acceptance_collective_elements,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            segments=worker._acceptance_segments,
            metrics=result,
        )
        if (
            worker._acceptance_optimizer_calls != 1
            or worker._acceptance_collectives == 0
        ):
            raise RuntimeError("Missing AdamW update or FSDP reduce-scatter")
        return result

    @functools.wraps(original_save)
    def save(worker, path, step=0):
        emit(
            worker, "checkpoint_saving", step=step, snapshot=optimizer_snapshot(worker)
        )
        original_save(worker, path, step)
        emit(worker, "checkpoint_saved", step=step, snapshot=optimizer_snapshot(worker))

    @functools.wraps(original_load)
    def load(worker, path):
        original_load(worker, path)
        emit(
            worker,
            "checkpoint_loaded",
            step=worker.global_step,
            snapshot=optimizer_snapshot(worker),
        )

    class AcceptanceWorker(Worker):
        init_worker = initialize
        forward_backward_batch = batch
        run_training = run
        save_checkpoint = save
        load_checkpoint = load

    return AcceptanceWorker


def train():
    """Use the production runner with an explicitly selected observed worker."""
    import hydra
    import torch.multiprocessing as mp

    from examples.sft.train_vla_sft import require_explicit_log_path
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    mp.set_start_method("spawn", force=True)

    @hydra.main(
        version_base="1.1",
        config_path=str(ROOT / "examples/sft/config"),
        config_name="robocasa365BowlandCup_ttt_sequence",
    )
    def entry(cfg):
        require_explicit_log_path(cfg)
        cfg = validate_cfg(cfg)
        cluster = Cluster(cluster_cfg=cfg.cluster)
        placement = HybridComponentPlacement(cfg, cluster)
        actor = (
            make_worker_class()
            .create_group(cfg)
            .launch(
                cluster,
                name=cfg.actor.group_name,
                placement_strategy=placement.get_strategy("actor"),
            )
        )
        runner = SFTRunner(cfg=cfg, actor=actor)
        runner.init_workers()
        for rank in range(8):
            path = Path(cfg.runner.logger.log_path) / f"acceptance_rank{rank}.jsonl"
            record = json.loads(path.read_text().splitlines()[0])
            if record["event"] != "initialized" or record["rank"] != rank:
                raise RuntimeError(f"Missing observer initialization for rank {rank}")
        LOG.info("All eight observer initializations verified; starting training")
        runner.run()

    entry()


def checkpoint(output, run, step):
    return output / run / "robottt3/checkpoints" / f"global_step_{step}"


def commands(output):
    result = {}
    for run in ("continuous", "resumed"):
        resume = (
            "null" if run == "continuous" else str(checkpoint(output, "continuous", 1))
        )
        result[run] = [
            PYTHON,
            "-u",
            SCRIPT,
            "--train",
            "--config-name",
            "robocasa365BowlandCup_ttt_sequence",
            "runner.max_steps=2",
            "runner.save_interval=-1",
            "runner.save_steps=[1,2]",
            "runner.val_check_interval=-1",
            f"runner.resume_dir={resume}",
            f"runner.logger.log_path={output / run}",
        ]
    result["assessment"] = [PYTHON, "-u", SCRIPT, "--output", str(output), "--assess"]
    return result


def source_hashes():
    paths = set()
    for directory in ("rlinf", "examples/sft", "evaluations"):
        for suffix in ("*.py", "*.yaml"):
            paths.update(
                p
                for p in (ROOT / directory).rglob(suffix)
                if not p.name.startswith("._")
            )
    return {str(path.relative_to(ROOT)): sha_file(path) for path in sorted(paths)}


def launch(output):
    """Run the approved two launches; never retry failures or overwrite outputs."""
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(f"Require fresh output with existing parent: {output}")
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [
        [int(item.strip()) for item in line.split(",")]
        for line in inventory.splitlines()
    ]
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if len(rows) != 8 or any(mem > 64 or util for _, mem, util in rows) or processes:
        raise RuntimeError(
            f"Eight idle GPUs required: {inventory}; processes={processes}"
        )
    output.mkdir()
    hashes = source_hashes()
    write_json(output / "source_before.json", hashes)
    (output / "git_status_before.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    )
    with (output / "source_changes.patch").open("wb") as stream:
        subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, stdout=stream, check=True)
    plan = {"started": time.time(), "commands": commands(output), "status": "running"}
    write_json(output / "launch.json", plan)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        RAY_ADDRESS="local",
        RAY_USAGE_STATS_ENABLED="0",
        RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES="4294967296",
        RLINF_CODE_WORKING_DIR="0",
        PYTHONPATH=str(ROOT) + os.pathsep + str(DEPENDENCY),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        USE_TF="0",
        NO_ALBUMENTATIONS_UPDATE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HYDRA_FULL_ERROR="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TORCHINDUCTOR_CACHE_DIR=str(output / "torchinductor_cache"),
        TRITON_CACHE_DIR=str(output / "triton_cache"),
    )
    try:
        for run, command in plan["commands"].items():
            if run != "assessment":
                (output / run).mkdir()
                env["RAY_TMPDIR"] = tempfile.mkdtemp(prefix="r3a-", dir="/tmp")
            else:
                env.pop("RLINF_EXT_MODULE", None)
                env["CUDA_VISIBLE_DEVICES"] = "0"
            plan["phase"] = run
            plan["ray_tmpdir"] = env.get("RAY_TMPDIR")
            with (output / f"{run}.log").open("x") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                plan["child_pid"] = process.pid
                write_json(output / "launch.json", plan)
                while process.poll() is None:
                    time.sleep(10)
                    with (output / "resources.jsonl").open("a") as monitor:
                        gpu = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=index,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        )
                        monitor.write(
                            json.dumps({"time": time.time(), "phase": run, "gpus": gpu})
                            + "\n"
                        )
                if process.returncode:
                    raise RuntimeError(f"{run} exited {process.returncode}; no retry")
        plan["status"] = "completed"
    except BaseException as error:
        plan["status"] = "failed"
        plan["error"] = repr(error)
        raise
    finally:
        plan["ended"] = time.time()
        write_json(output / "launch.json", plan)
        current = source_hashes()
        write_json(output / "source_after.json", current)
        write_json(output / "source_freeze.json", {"unchanged": hashes == current})


def difference(pairs):
    import torch

    maximum, squared, reference, count = 0.0, 0.0, 0.0, 0
    exact = True
    for left, right in pairs:
        a, b = left.detach().float(), right.detach().float()
        if a.shape != b.shape:
            raise ValueError("Compared tensors have different shapes")
        if not a.numel():
            continue
        delta = a - b
        maximum = max(maximum, float(delta.abs().max()))
        squared += float(delta.double().square().sum())
        reference += float(a.double().square().sum())
        count += a.numel()
        exact = exact and torch.equal(left, right)
    return {
        "exact_equal": exact,
        "max_abs": maximum,
        "rms_diff": (squared / max(count, 1)) ** 0.5,
        "relative_l2": (squared / max(reference, 1e-30)) ** 0.5,
        "reference_l2": reference**0.5,
        "difference_l2": squared**0.5,
    }


def compare_snapshots(saved, loaded):
    equal = {key: saved[key] == loaded[key] for key in saved}
    required = (
        "model_shard_sha256",
        "optimizer_nonempty",
        "param_groups",
        "buffers_sha256",
        "rng",
        "optimizer_steps",
        "data_offset",
        "scheduler",
        "lr",
    )
    equal["effective_state_equal"] = all(equal[key] for key in required)
    equal["nonempty_mismatches"] = [
        name
        for name in sorted(
            saved["optimizer_nonempty"].keys() | loaded["optimizer_nonempty"].keys()
        )
        if saved["optimizer_nonempty"].get(name)
        != loaded["optimizer_nonempty"].get(name)
    ]
    equal["empty_state_differences"] = [
        name
        for name in sorted(
            saved["optimizer_empty"].keys() | loaded["optimizer_empty"].keys()
        )
        if saved["optimizer_empty"].get(name) != loaded["optimizer_empty"].get(name)
    ]
    return equal


def strict_resume_pass(report):
    expected = {
        "forward_input": 8,
        "action_input": 32,
        "gradient_before_clip": 1,
        "optimizer_before": 1,
        "optimizer_after": 1,
    }
    if (
        not report["effective_restore_pass"]
        or not report["step2_weights"]["exact_equal"]
    ):
        return False
    if any(
        len(report[key]) != 8
        for key in (
            "step2_equality_per_rank",
            "step2_sample_identity_equal",
            "step2_boundary_comparisons",
        )
    ):
        return False
    if not all(x["effective_state_equal"] for x in report["step2_equality_per_rank"]):
        return False
    if not all(report["step2_sample_identity_equal"]):
        return False
    return all(
        boundary["continuous_delta_reference_valid"]
        and boundary["resumed_delta_reference_valid"]
        and all(
            len(boundary[event]) == count
            and all(all(row.values()) for row in boundary[event])
            for event, count in expected.items()
        )
        for boundary in report["step2_boundary_comparisons"]
    )


def enforce_acceptance(report, output):
    evaluation = report["evaluation"]
    report["acceptance_passed"] = bool(
        report["strict_resume_pass"]
        and report["dimension_padding"]["passed"]
        and evaluation["finite"]
        and evaluation["actions_shape"] == [1, 40, 132]
        and evaluation["memory_position"] == [1]
        and evaluation["dimension_prediction"]["exact_equal"]
        and evaluation["dimension_memory"]["exact_equal"]
    )
    write_json(output / "report.json", report)
    if not report["acceptance_passed"]:
        raise RuntimeError(f"Acceptance failed; inspect {output / 'report.json'}")


def assess(output):
    """Exercise actual checkpoint loader/head and compare continuous/resumed state."""
    import torch
    from omegaconf import OmegaConf

    from evaluations.gr00t_native_policy import NativeEvalPolicy
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import native_ttt as api

    torch.set_num_threads(4)
    report = {
        "protocol": {
            "training": {
                "memory_source": "supervised_noisy",
                "write_tokens": 33,
                "token_shape": 57,
                "action_horizon": 16,
                "anchor_stride": 16,
                "memory_reset": "sample window",
                "tbptt": 4,
                "mask_padded_action_dims": True,
            },
            "inference": {
                "memory_source": "predicted_final",
                "write_tokens": 57,
                "action_horizon": 40,
                "executed_steps_between_updates": 8,
                "memory_reset": "episode/env reset",
                "mask_padded_action_dims": True,
            },
            "aligned": False,
        }
    }
    records = {}
    for run in ("continuous", "resumed"):
        records[run] = [
            [
                json.loads(line)
                for line in (output / run / f"acceptance_rank{rank}.jsonl")
                .read_text()
                .splitlines()
            ]
            for rank in range(8)
        ]
    restoration, rerun, samples, boundaries = [], [], [], []
    for rank in range(8):
        saved = next(
            x["snapshot"]
            for x in records["continuous"][rank]
            if x["event"] == "checkpoint_saved" and x["step"] == 1
        )
        loaded = next(
            x["snapshot"]
            for x in records["resumed"][rank]
            if x["event"] == "checkpoint_loaded"
        )
        restoration.append(compare_snapshots(saved, loaded))
        a = next(
            x["snapshot"]
            for x in records["continuous"][rank]
            if x["event"] == "checkpoint_saved" and x["step"] == 2
        )
        b = next(
            x["snapshot"]
            for x in records["resumed"][rank]
            if x["event"] == "checkpoint_saved" and x["step"] == 2
        )
        rerun.append(compare_snapshots(a, b))
        boundary = {}
        for event in (
            "forward_input",
            "action_input",
            "gradient_before_clip",
            "optimizer_before",
            "optimizer_after",
        ):
            events = [
                [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in ("time", "rank", "event")
                    }
                    for item in records[run][rank]
                    if item["event"] == event and item["step"] == 2
                ]
                for run in ("continuous", "resumed")
            ]
            if not events[0] or len(events[0]) != len(events[1]):
                raise RuntimeError(
                    f"Missing paired {event} observations for rank {rank}"
                )
            boundary[event] = [
                {
                    key: compare_snapshots(left[key], right[key])[
                        "effective_state_equal"
                    ]
                    if key == "snapshot"
                    else left[key] == right[key]
                    for key in left
                }
                for left, right in zip(*events)
            ]
        for run in ("continuous", "resumed"):

            def observation(event):
                return next(
                    item["snapshot"]
                    for item in records[run][rank]
                    if item["event"] == event and item["step"] == 2
                )

            before = observation("optimizer_before")
            after = observation("optimizer_after")
            stored = observation("checkpoint_saved")
            boundary[f"{run}_delta_reference_valid"] = (
                before["model_shard_sha256"] == saved["model_shard_sha256"]
                and after["model_shard_sha256"] == stored["model_shard_sha256"]
            )
        boundaries.append(boundary)
        descriptions = []
        for run in ("continuous", "resumed"):
            descriptions.append(
                next(
                    json.loads(line)["samples"]
                    for line in (output / run / f"sequence_samples_rank{rank}.jsonl")
                    .read_text()
                    .splitlines()
                    if json.loads(line)["step"] == 2
                )
            )
        samples.append(descriptions[0] == descriptions[1])
    report["restoration_per_rank"] = restoration
    report["step2_equality_per_rank"] = rerun
    report["step2_sample_identity_equal"] = samples
    report["step2_boundary_comparisons"] = boundaries
    report["effective_restore_pass"] = all(
        x["effective_state_equal"] for x in restoration
    )
    report["training_records"] = records
    weights = [
        checkpoint(output, run, 2) / "actor/model_state_dict/full_weights.pt"
        for run in ("continuous", "resumed")
    ]
    a, b = [
        torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for path in weights
    ]
    if a.keys() != b.keys():
        raise ValueError("Continuous/resumed model keys differ")
    report["step2_weights"] = difference((a[key], b[key]) for key in a)
    initial = torch.load(
        checkpoint(output, "continuous", 1) / "actor/model_state_dict/full_weights.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if a.keys() != initial.keys():
        raise ValueError("Step1/step2 model keys differ")
    if not all(
        x[f"{run}_delta_reference_valid"]
        for x in boundaries
        for run in ("continuous", "resumed")
    ):
        raise RuntimeError("Saved weights cannot reconstruct the observed AdamW update")
    parameter_names = set()
    for record in records["continuous"]:
        snapshot = next(
            item["snapshot"]
            for item in record
            if item["event"] == "checkpoint_saved" and item["step"] == 1
        )
        for entries in (snapshot["optimizer_nonempty"], snapshot["optimizer_empty"]):
            for entry in entries.values():
                parameter_names.update(entry["layout"]["aliases"])
    if not parameter_names or not parameter_names <= a.keys():
        raise RuntimeError(
            "Optimizer FQNs do not match the saved model parameter names"
        )
    report["step2_parameter_updates"] = difference(
        (a[key].float() - initial[key].float(), b[key].float() - initial[key].float())
        for key in sorted(parameter_names)
    )
    report["strict_resume_pass"] = strict_resume_pass(report)
    del a, b, initial
    write_json(output / "report.json", report)
    config = OmegaConf.load(
        ROOT / "evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml"
    ).rollout.model
    policy = NativeEvalPolicy(config, str(weights[0]), torch.device("cuda:0"))
    data = torch.load(output / "continuous/first_anchor.pt", weights_only=True)
    torch.manual_seed(20261104)
    with torch.no_grad():
        features, state, inputs = api._encode(policy.model, data, training=False)
        actions = inputs.action
        if actions.shape != (1, 40, 132):
            raise ValueError(f"Unexpected real action shape: {actions.shape}")
        valid = torch.ones(1, 1, dtype=torch.bool, device=actions.device)
        noise = torch.randn_like(actions)
        altered = noise.clone()
        altered[..., 12:] = torch.randn_like(altered[..., 12:])
        tau = 0.5
        buckets = torch.full(
            (1,),
            int(tau * policy.model.action_head.num_timestep_buckets),
            device=actions.device,
            dtype=torch.long,
        )

        def velocity(epsilon, memory):
            return api._velocity(
                policy.model.action_head,
                tau * actions + (1 - tau) * epsilon,
                buckets,
                inputs.embodiment_id,
                features,
                state,
                memory,
                valid,
                active_tokens=33,
            )

        _, incoming, _ = velocity(noise, None)
        prediction, memory, _ = velocity(noise, incoming)
        repeat_prediction, repeat_memory, _ = velocity(noise, incoming)
        changed_prediction, changed_memory, _ = velocity(altered, incoming)
        report["dimension_padding"] = {
            "tau": tau,
            "changed_dimensions": [12, 132],
            "input_dtype": str(actions.dtype),
            "real_checkpoint": str(weights[0]),
            "real_saved_training_anchor": True,
            "fixed_incoming_memory": True,
            "prediction": difference(
                [(prediction[:, :16, :12], changed_prediction[:, :16, :12])]
            ),
            "fast_state": difference(
                (x, y)
                for layer, changed in zip(memory.layers, changed_memory.layers)
                for x, y in zip(layer, changed)
            ),
            "repeat_prediction": difference([(prediction, repeat_prediction)]),
            "repeat_memory": difference(
                (x, y)
                for layer, repeated in zip(memory.layers, repeat_memory.layers)
                for x, y in zip(layer, repeated)
            ),
        }
        report["dimension_padding"]["passed"] = all(
            report["dimension_padding"][key]["exact_equal"]
            for key in (
                "prediction",
                "fast_state",
                "repeat_prediction",
                "repeat_memory",
            )
        )
        noise = torch.randn(1, 40, 132, device=actions.device, dtype=actions.dtype)
        result = policy.rollout.predict(data, [0], policy_noise=noise)
        changed_noise = noise.clone()
        changed_noise[..., 12:] = torch.randn_like(changed_noise[..., 12:])
        changed_result = api.native_ttt_predict(
            policy.model, data, policy_noise=changed_noise
        )
        report["evaluation"] = {
            "identity": policy.identity,
            "actions_shape": list(result["actions"].shape),
            "finite": bool(torch.isfinite(result["actions"]).all()),
            "memory_position": result["memory"].position.cpu().tolist(),
            "scope": "real loader and native rollout on saved processed training observation; no simulator success evaluation",
            "dimension_prediction": difference(
                [(result["actions"][..., :12], changed_result["actions"][..., :12])]
            ),
            "dimension_memory": difference(
                (x, y)
                for left, right in zip(
                    result["memory"].layers, changed_result["memory"].layers
                )
                for x, y in zip(left, right)
            ),
        }
    enforce_acceptance(report, output)
    LOG.info("Acceptance report saved to %s", output / "report.json")


def main():
    logging.basicConfig(level=logging.INFO)
    if "--train" in sys.argv:
        sys.argv.remove("--train")
        train()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assess", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    if args.assess:
        assess(args.output)
    elif args.execute:
        launch(args.output)
    else:
        LOG.info("Preview only: %s", json.dumps(commands(args.output), indent=2))


if __name__ == "__main__":
    main()
