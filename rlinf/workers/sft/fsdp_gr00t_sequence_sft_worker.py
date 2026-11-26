# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native GR00T baseline SFT with deterministic packed episode windows."""

import hashlib
import json
import logging
import math
import os
import subprocess
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.utils import seed_everything
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def configure_sequence_determinism(enabled: bool) -> None:
    """Fix CUDA and compiler algorithm selection before model initialization."""
    if type(enabled) is not bool:
        raise TypeError("deterministic_training must be a boolean")
    if not enabled:
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError(
            "Deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    # Transformers' external FlashAttention backend reads its own per-call flag.
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    # PyTorch 2.11 also fixes Inductor's numerically significant autotuning choices.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class TrainingProfiler:
    """Capture one zero-based, half-open optimizer-step window per worker."""

    def __init__(
        self,
        config,
        rank,
        world_size,
        max_steps,
        *,
        synchronize,
        start_nsight,
        stop_nsight,
        nsight_range,
        nsight_enabled=False,
    ):
        self.mode = config.get("mode", "off")
        self.active = False
        self.finished = False
        self.capture = None
        self.rank = rank
        self.synchronize = synchronize
        self.start_nsight = start_nsight
        self.stop_nsight = stop_nsight
        self.nsight_range = nsight_range
        if self.mode not in {"off", "torch", "nsys"}:
            raise ValueError("Profiling mode must be off, torch or nsys")
        if (self.mode == "nsys") != nsight_enabled:
            raise ValueError(
                "Nsight worker wrapping must be enabled only for nsys mode"
            )
        self.selected = False
        if self.mode == "off":
            return
        self.start = config.get("start_step", 18)
        self.end = config.get("end_step", 20)
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end <= max_steps
        ):
            raise ValueError(
                "Profiling requires 0 <= start_step < end_step <= max_steps"
            )
        ranks = config.get("ranks")
        if ranks is not None and (
            not ranks
            or any(type(r) is not int or not 0 <= r < world_size for r in ranks)
            or len(set(ranks)) != len(ranks)
        ):
            raise ValueError(
                "Profiling ranks must be unique valid ranks, or null for all"
            )
        self.selected = ranks is None or rank in ranks
        directory = config.get("output_dir")
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError(
                "Profiling output_dir must be an explicit absolute directory"
            )
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError(
                "Create the approved profiling output directory before launch"
            )
        self.options = {}
        for key in ("record_shapes", "with_stack"):
            value = config.get(key, False)
            if type(value) is not bool:
                raise ValueError(f"Profiling {key} must be a boolean")
            self.options[key] = value
        suffix = f"rank{rank}_steps{self.start}_{self.end}"
        self.trace = self.directory / f"torch_profile_{suffix}.json"
        self.marker = self.directory / f"capture_{suffix}.json"

    def range(self, name):
        """Emit stage labels only while this rank is being captured."""
        if not self.active:
            return nullcontext()
        if self.mode == "torch":
            return torch.profiler.record_function(f"robottt/{name}")
        return self.nsight_range(f"robottt/{name}")

    def _start(self):
        if self.trace.exists():
            raise FileExistsError(self.trace)
        self.metadata = {
            "mode": self.mode,
            "rank": self.rank,
            "start_step": self.start,
            "end_step_exclusive": self.end,
            "status": "recording",
            **self.options,
        }
        with self.marker.open("x") as stream:
            json.dump(self.metadata, stream)
        try:
            self.synchronize()
            if self.mode == "torch":
                self.capture = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    **self.options,
                )
                self.capture.start()
            else:
                self.start_nsight(self.start)
        except BaseException as error:
            self.finished = True
            try:
                if self.capture is not None:
                    self.capture.stop()
                elif self.mode == "nsys":
                    self.stop_nsight()
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"Profiler startup cleanup failed: {cleanup_error!r}"
                    )
            self.metadata["status"] = "failed"
            self.marker.write_text(json.dumps(self.metadata, indent=2) + "\n")
            raise
        self.active = True

    def _finish(self, status):
        try:
            self.synchronize()
        finally:
            try:
                if self.mode == "torch":
                    self.capture.stop()
                    self.capture.export_chrome_trace(str(self.trace))
                else:
                    self.stop_nsight()
                self.metadata["status"] = status
                self.marker.write_text(json.dumps(self.metadata, indent=2) + "\n")
            finally:
                self.active = False
                self.finished = True

    @contextmanager
    def step(self, step):
        """Include data loading through optimizer update, exporting on exit."""
        if not self.selected or self.finished or not self.start <= step < self.end:
            yield
            return
        if not self.active:
            if step != self.start:
                raise ValueError("Profiling must begin at start_step, not mid-window")
            self._start()
        try:
            with self.range(f"optimizer_step/{step}"):
                yield
            if self.mode == "torch":
                self.capture.step()
        except BaseException as error:
            try:
                self._finish("interrupted")
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Profiler cleanup also failed: {cleanup_error!r}")
            raise
        else:
            if step + 1 == self.end:
                self._finish("complete")


def sequence_policy_seeds(batch, start, end, seed, device):
    """Derive independent per-anchor policy noise without advancing native RNG."""
    import numpy as np

    b, length = batch["anchor_valid"].shape
    descriptions = batch["descriptions"]
    if len(descriptions) != b or not 0 <= start < end <= length:
        raise ValueError("Policy seeds require valid sequence descriptions and segment")
    result = torch.zeros((b, end - start), dtype=torch.long)
    for row, description in enumerate(descriptions):
        anchors = description["anchors"]
        for index in range(start, end):
            if not batch["anchor_valid"][row, index]:
                continue
            if index >= len(anchors):
                raise ValueError("Valid anchor is missing its sample identity")
            result[row, index - start] = int(
                np.random.SeedSequence(
                    [
                        seed,
                        description["epoch"],
                        description["episode"],
                        anchors[index],
                        90211,
                    ]
                ).generate_state(1)[0]
            )
    return result.to(device=device)


def validate_resume_lr_plan(plan: dict) -> None:
    """Validate an explicit compression of the remaining reference schedule."""
    if (
        set(plan) != {"mode", "anchor_step", "original_total_steps", "total_steps"}
        or plan["mode"] != "compress_remaining_cosine"
    ):
        raise ValueError("Unsupported resume LR schedule")
    if (
        any(
            type(plan[key]) is not int
            for key in ("anchor_step", "original_total_steps", "total_steps")
        )
        or not 0
        < plan["anchor_step"]
        < plan["total_steps"]
        < plan["original_total_steps"]
    ):
        raise ValueError("Invalid resume LR schedule boundaries")


def compressed_resume_lambda(
    step: int,
    *,
    original,
    anchor_step: int,
    original_total_steps: int,
    total_steps: int,
) -> float:
    """Preserve the anchor LR and advance only the remaining cosine phase."""
    mapped = step
    if step > anchor_step:
        mapped = min(
            original_total_steps,
            anchor_step
            + (step - anchor_step)
            * (original_total_steps - anchor_step)
            / (total_steps - anchor_step),
        )
    return original(mapped)


def apply_resume_lr_schedule(scheduler, plan: dict) -> None:
    """Wrap restored lambdas without stepping or changing optimizer state."""
    validate_resume_lr_plan(plan)
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError("Resume LR compression requires LambdaLR")
    if not plan["anchor_step"] <= scheduler.last_epoch <= plan["total_steps"]:
        raise ValueError("Restored scheduler is outside the resume LR interval")
    wrapped = []
    for original, base_lr, saved_lr, group in zip(
        scheduler.lr_lambdas,
        scheduler.base_lrs,
        scheduler.get_last_lr(),
        scheduler.optimizer.param_groups,
        strict=True,
    ):
        if isinstance(original, partial) and original.func is compressed_resume_lambda:
            raise ValueError("Resume LR schedule is already applied")
        # partial has an empty __dict__, preserving LambdaLR's DCP state schema.
        function = partial(
            compressed_resume_lambda,
            original=original,
            anchor_step=plan["anchor_step"],
            original_total_steps=plan["original_total_steps"],
            total_steps=plan["total_steps"],
        )
        expected = base_lr * function(scheduler.last_epoch)
        if not all(
            math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15)
            for value in (saved_lr, group["lr"])
        ):
            raise ValueError("Restored LR does not match the declared resume schedule")
        wrapped.append(function)
    scheduler.lr_lambdas = wrapped


def validate_resume_metadata(
    previous: dict,
    current: dict,
    *,
    migration: dict | None = None,
    checkpoint_identity: dict | None = None,
) -> None:
    """Reject resume across data, code, precision or optimization protocols."""
    # Match the checkpoint JSON representation, including integer mapping keys.
    previous = json.loads(json.dumps(previous))
    current = json.loads(json.dumps(current))
    old_plan = previous.get("resume_lr_schedule")
    new_plan = current.get("resume_lr_schedule")
    if old_plan != new_plan:
        if old_plan is not None or new_plan is None or migration is None:
            raise ValueError("Strict sequence resume mismatch: resume_lr_schedule")
        validate_resume_lr_plan(new_plan)
        if (
            migration.get("format") != "sequence_resume_lr_migration_v1"
            or checkpoint_identity is None
            or migration.get("checkpoint") != checkpoint_identity
            or checkpoint_identity.get("step") != new_plan["anchor_step"]
            or migration.get("resume_lr_schedule") != new_plan
            or migration.get("old_source_sha256") != previous["source"]["source_sha256"]
            or migration.get("new_source_sha256") != current["source"]["source_sha256"]
        ):
            raise ValueError("Checkpoint is not covered by the explicit LR migration")
        # Only the reviewed source hashes may migrate; all other identity stays strict.
        previous["source"]["source_sha256"] = current["source"]["source_sha256"]
    for key in ("world_size", "protocol", "data", "processor", "source", "parameters"):
        if previous[key] != current[key]:
            raise ValueError(f"Strict sequence resume mismatch: {key}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_manifest(manifest: dict, actual_sizes: dict) -> None:
    """Require both inference weights and the complete recorded resume state."""
    required = {
        "model_state_dict/full_weights.pt",
        "data.pt",
        "rng.pt",
        "sequence_metadata.json",
        "processor_bundle.json",
        "sequence_progress.pt",
        "dcp_checkpoint/.metadata",
    }
    files = manifest["files"]
    shards = {
        name
        for name in files
        if name.startswith("dcp_checkpoint/") and name.endswith(".distcp")
    }
    if type(manifest["step"]) is not int or manifest["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if not required.issubset(files) or not shards:
        raise ValueError(
            "Checkpoint must include weights, DCP shards and all resume sidecars"
        )
    for name, size in files.items():
        if type(size) is not int or size <= 0 or actual_sizes.get(name) != size:
            raise ValueError(f"Incomplete checkpoint member: {name}")


class FSDPGr00tSequenceSftWorker(FSDPVlaSftWorker):
    """Own segment backward, full-batch iteration and strict resumable state."""

    def model_provider_func(self):
        seed_everything(self.cfg.actor.seed)
        model = super().model_provider_func()
        profile = getattr(self, "_training_profile", None)
        if profile is not None and profile.selected:
            model.action_head._training_profile_range = profile.range
        self._parameters = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        self._effective_model = {
            "native_config": model.config.to_dict(),
            "state_dropout_prob": model.action_head.state_dropout_prob,
            "module_dropout_values": sorted(
                {
                    module.p
                    for module in model.modules()
                    if isinstance(module, torch.nn.Dropout)
                }
            ),
        }
        return model

    def build_lr_scheduler(self, optimizer, optim_config, last_epoch=-1):
        """Retain the sequence protocol's nonzero first warmup update."""
        if optim_config.get("lr_scheduler") != "linear_warmup_cosine":
            return super().build_lr_scheduler(optimizer, optim_config, last_epoch)
        import math

        warmup = int(optim_config.lr_warmup_steps)
        total = int(optim_config.total_training_steps)
        if warmup < 0 or total <= warmup:
            raise ValueError("Expected total_training_steps > lr_warmup_steps >= 0")

        def multiplier(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return (1 + math.cos(math.pi * min(progress, 1))) / 2

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, multiplier, last_epoch=last_epoch
        )

    def build_dataloader(self, data_paths, eval_dataset=False):
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            build_gr00t_sequence_dataloader,
        )

        if eval_dataset or data_paths != self.cfg.data.train_data_paths:
            raise ValueError("Sequence SFT requires the configured training dataset")
        return build_gr00t_sequence_dataloader(self.cfg, self._world_size, self._rank)

    def get_max_steps_per_epoch(self):
        # Epochs are tracked by sample identity, not truncated optimizer updates.
        if self.cfg.runner.max_epochs > 0 or self.cfg.runner.max_steps <= 0:
            raise ValueError(
                "Packed sequence SFT requires max_steps > 0 and max_epochs <= 0"
            )
        samples = self.cfg.data.expected_episodes
        if self.cfg.data.robottt.get("sampling_mode") == "independent_h40":
            samples = self.data_config["data"]["sample_count"]
        return max(1, samples // self.global_batch_size)

    def init_worker(self):
        from rlinf.scheduler.hardware import AcceleratorUtil

        plan = self.cfg.runner.get("resume_lr_schedule")
        if plan is not None:
            validate_resume_lr_plan(plan)
            if (
                not self.cfg.runner.get("resume_dir")
                or self.cfg.actor.optim.lr_scheduler != "cosine"
                or self.cfg.actor.optim.total_training_steps
                != plan["original_total_steps"]
                or not plan["anchor_step"]
                < self.cfg.runner.max_steps
                <= plan["total_steps"]
            ):
                raise ValueError(
                    "Resume LR plan requires the original cosine and a resume checkpoint"
                )
        configure_sequence_determinism(
            self.cfg.actor.get("deterministic_training", False)
        )
        wrapper = self.cfg.cluster.get("profiling") or {}
        wrapped = wrapper.get("enabled", False) and self.cfg.actor.group_name in (
            wrapper.get("worker_groups") or []
        )
        self._training_profile = TrainingProfiler(
            self.cfg.actor.get("profiling", {}),
            self._rank,
            self._world_size,
            self.cfg.runner.max_steps,
            synchronize=self.torch_platform.synchronize,
            start_nsight=self.start_profile,
            stop_nsight=self.stop_profile,
            nsight_range=lambda name: AcceleratorUtil.profiling_range(
                self._accelerator_type, name
            ),
            nsight_enabled=bool(wrapped and wrapper.get("backend") == "nsight"),
        )
        torch.set_num_threads(self.cfg.actor.cpu_threads)
        torch.set_float32_matmul_precision("highest")
        super().init_worker()
        seed_everything(self.cfg.actor.seed + self._rank)
        source = [self.source_metadata() if self._rank == 0 else None]
        torch.distributed.broadcast_object_list(source, src=0)
        self._sequence_metadata = {
            "format": "rlinf_native_gr00t_sequence_v1",
            "world_size": self._world_size,
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
            "protocol": {
                "actor": OmegaConf.to_container(self.cfg.actor, resolve=True),
                "data": OmegaConf.to_container(self.cfg.data, resolve=True),
                "effective_model": self._effective_model,
                "batching": "cross_epoch_full_microbatch",
                "loss": (
                    "masked_action_mean_then_window_mean"
                    if self.cfg.data.robottt.get("sampling_mode") == "independent_h40"
                    else "masked_action_mean_then_full_sequence_mean"
                ),
            },
            "data": self.data_config["data"],
            "processor": self.data_config["processor"],
            "parameters": self._parameters,
            "source": source[0],
        }
        if plan is not None:
            self._sequence_metadata["resume_lr_schedule"] = OmegaConf.to_container(
                plan, resolve=True
            )

    def source_metadata(self) -> dict:
        import importlib.metadata

        import gr00t

        root = Path(__file__).resolve().parents[3]
        dependency = Path(gr00t.__file__).resolve().parent.parent
        names = [
            "examples/sft/train_vla_sft.py",
            "rlinf/runners/sft_runner.py",
            "rlinf/utils/runner_utils.py",
            "rlinf/data/datasets/gr00t_n1d7/sequence_dataset.py",
            "rlinf/data/datasets/gr00t_n1d7/window_dataset.py",
            "rlinf/data/datasets/gr00t_n1d7/window_sampling.py",
            "rlinf/models/embodiment/gr00t/gr00t_n1d7/sft.py",
            "rlinf/models/embodiment/gr00t/gr00t_n1d7/__init__.py",
            "rlinf/workers/sft/fsdp_gr00t_sequence_sft_worker.py",
            "rlinf/workers/sft/fsdp_sft_worker.py",
            "rlinf/workers/sft/fsdp_vla_sft_worker.py",
            "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
            "rlinf/hybrid_engines/fsdp/strategy/base.py",
            "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
        ]
        if self.cfg.actor.model.rl_head_config.use_ttt or (
            self.cfg.actor.model.rl_head_config.ttt.num_register_tokens
        ):
            names += [
                "rlinf/models/embodiment/gr00t/gr00t_n1d7/" + filename
                for filename in ("native_ttt.py", "ttt.py", "rollout.py")
            ]
            names += [
                "rlinf/hybrid_engines/fsdp/mixed_precision.py",
                "rlinf/utils/fsdp_config.py",
            ]
        dependency_names = [
            "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
            "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
            "gr00t/model/modules/dit.py",
            "gr00t/model/modules/qwen3_backbone.py",
            "gr00t/data/dataset/lerobot_episode_loader.py",
        ]
        assets = {}
        for label, folder in (
            ("base", self.cfg.actor.model.model_path),
            ("backbone", self.cfg.actor.model.backbone_model_path),
        ):
            directory = Path(folder)
            assets[label] = {
                p.name: sha256_file(p)
                for p in sorted(directory.iterdir())
                if p.is_file()
                and (
                    p.suffix == ".safetensors"
                    or p.name
                    in {
                        "config.json",
                        "model.safetensors.index.json",
                        "processor_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "preprocessor_config.json",
                    }
                )
            }
        return {
            "rlinf_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "gr00t_commit": subprocess.check_output(
                ["git", "-C", str(dependency), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {name: sha256_file(root / name) for name in names},
            "gr00t_sha256": {
                name: sha256_file(dependency / name) for name in dependency_names
            },
            "assets": assets,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "diffusers",
                    "torchdata",
                    "ray",
                    "albumentations",
                )
            },
        }

    def step_batching(self) -> tuple[int, float, int | None]:
        """Return synchronized microbatch count and backward scale for this step."""
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import window_backward_scale

        plan = self.data_config["data"].get("matched_sequence_budget")
        if plan is None:
            return self.gradient_accumulation, 1 / self.gradient_accumulation, None
        windows = plan["windows"][self.global_step]
        capacity = self._world_size * self.cfg.actor.micro_batch_size
        return (
            (windows + capacity - 1) // capacity,
            window_backward_scale(
                windows, self._world_size, self.cfg.actor.micro_batch_size
            ),
            windows,
        )

    def forward_backward_batch(
        self, batch, is_last_micro_batch: bool, backward_scale: float | None = None
    ) -> dict:
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import segment_batch

        profile = getattr(self, "_training_profile", None)
        phase = profile.range if profile is not None else nullcontext
        if backward_scale is None:
            backward_scale = 1 / self.gradient_accumulation
        b, length = batch["anchor_valid"].shape
        segment_length = self.cfg.actor.model.tbptt_segment_length
        if b != self.cfg.actor.micro_batch_size:
            raise ValueError(
                f"Expected full microbatch {self.cfg.actor.micro_batch_size}, got {b}"
            )
        if segment_length <= 0 or length % segment_length:
            raise ValueError("Sequence length must divide into complete segments")
        total_loss = 0.0
        use_ttt = getattr(
            getattr(self.cfg.actor.model, "rl_head_config", None), "use_ttt", False
        )
        memory = None
        updates = []
        for start in range(0, length, segment_length):
            end = start + segment_length
            with phase("data_to_device"):
                segment = segment_batch(batch, start, end, self.device)
            if use_ttt:
                segment["policy_noise_seed"] = sequence_policy_seeds(
                    batch, start, end, self.cfg.actor.seed, self.device
                )
            sync = is_last_micro_batch and end == length
            with self.before_micro_batch(self.model, sync):
                with self.amp_context:
                    kwargs = {"memory": memory} if use_ttt else {}
                    output = self.model(
                        forward_type=ForwardType.SFT, data=segment, **kwargs
                    )
                loss = output["loss"]
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        finite, op=torch.distributed.ReduceOp.MIN
                    )
                if not finite.item():
                    raise FloatingPointError(
                        "Non-finite sequence loss; optimizer not stepped"
                    )
                with phase("backward"):
                    self.grad_scaler.scale(loss * backward_scale).backward()
            if use_ttt:
                if output.get("memory") is None:
                    raise RuntimeError("TTT forward must return its candidate memory")
                memory = output["memory"].detach()
                updates.append(float(output["ttt_update_rms"].detach()))
            total_loss += loss.detach().item()
        return {
            "loss": total_loss,
            "valid_anchors": batch["anchor_valid"].sum().item(),
            "sequence_samples": b,
            "ttt_update_rms": sum(updates) / len(updates) if updates else 0.0,
        }

    def run_training(self) -> dict:
        collector = self._training_profile
        with self._inline_profile_guard() as capture:
            if capture and self.global_step == collector.start:
                logging.info("profiler start")
                torch.cuda.cudart().cudaProfilerStart()
                self._cuda_profile_started = True
                from torch.profiler import ProfilerActivity, profile

                self.profile_context = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=collector.options["record_shapes"],
                    with_stack=collector.options["with_stack"],
                )
                collector.capture = self.profile_context
                self.profile_context.start()
                collector.active = True

            metrics = self._run_training_step()

            if capture and self.global_step + 1 == collector.end:
                logging.info("profiler stop")
                collector.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                self._cuda_profile_started = False
                self.profile_context.stop()
                self._profile_stopped = True
                trace_path = str(collector.trace)
                self.profile_context.export_chrome_trace(trace_path)

        return metrics

    @contextmanager
    def _inline_profile_guard(self):
        """Validate capture ownership and clean up an interrupted inline capture."""
        collector = self._training_profile
        capture = (
            collector.mode == "torch"
            and collector.selected
            and not collector.finished
            and collector.start <= self.global_step < collector.end
        )
        if not capture:
            yield False
            return
        if not collector.active:
            if self.global_step != collector.start:
                raise ValueError("Profiling must begin at start_step, not mid-window")
            if collector.trace.exists():
                raise FileExistsError(collector.trace)
            collector.metadata = {
                "mode": collector.mode,
                "rank": self._rank,
                "start_step": collector.start,
                "end_step_exclusive": collector.end,
                "status": "recording",
                **collector.options,
            }
            with collector.marker.open("x") as stream:
                json.dump(collector.metadata, stream)
            self.profile_context = None
            self._profile_stopped = False
            self._cuda_profile_started = False
        try:
            if self.global_step == collector.start:
                collector.synchronize()
            yield True
        except BaseException as error:
            partial = collector.active and not self._profile_stopped
            cleanup = []
            if self._cuda_profile_started:
                cleanup.append(lambda: torch.cuda.cudart().cudaProfilerStop())
            if self.profile_context is not None and not self._profile_stopped:
                cleanup.append(self.profile_context.stop)
            if partial:
                cleanup.append(
                    lambda: self.profile_context.export_chrome_trace(
                        str(collector.trace)
                    )
                )
            collector.metadata["status"] = "interrupted" if partial else "failed"
            cleanup.append(
                lambda: collector.marker.write_text(
                    json.dumps(collector.metadata, indent=2) + "\n"
                )
            )
            for operation in cleanup:
                try:
                    operation()
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"Profiler cleanup also failed: {cleanup_error!r}"
                        )
            self._cuda_profile_started = False
            self._profile_stopped = True
            collector.active = False
            collector.finished = True
            raise
        else:
            if self.global_step + 1 == collector.end:
                collector.active = False
                collector.finished = True
                collector.metadata["status"] = "complete"
                collector.marker.write_text(
                    json.dumps(collector.metadata, indent=2) + "\n"
                )

    def _run_training_step(self) -> dict:
        """Perform one optimizer update without owning the torch capture."""
        collector = self._training_profile
        inline_torch = collector.mode == "torch"
        lifecycle = nullcontext() if inline_torch else collector.step(self.global_step)
        phase = (
            collector.range(f"optimizer_step/{self.global_step}")
            if inline_torch
            else nullcontext()
        )
        with lifecycle, self.worker_timer("run_training"), phase:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            values = []
            samples = []
            microbatches, backward_scale, budget = self.step_batching()
            for index in range(microbatches):
                with self._training_profile.range("data_loading"):
                    batch = next(self.data_iter)
                if budget is not None and (
                    batch["budget_step"] != self.global_step + 1
                    or batch["budget_microbatch"] != index
                    or batch["budget_windows"] != budget
                ):
                    raise ValueError("H40 data cursor and optimizer budget disagree")
                values.append(
                    self.forward_backward_batch(
                        batch, index + 1 == microbatches, backward_scale
                    )
                )
                samples.extend(d for d in batch["descriptions"] if d["valid"] > 0)
                self._data_iter_offset += 1
            if budget is not None:
                actual = torch.tensor(
                    sum(x["valid_anchors"] for x in values),
                    device=self.device,
                    dtype=torch.int64,
                )
                torch.distributed.all_reduce(actual)
                if actual.item() != budget:
                    raise ValueError(
                        f"Expected {budget} global windows, got {actual.item()}"
                    )
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = self._strategy.clip_grad_norm_(model=self.model)
            finite = torch.isfinite(torch.as_tensor(grad_norm, device=self.device)).to(
                torch.int32
            )
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(
                    "Non-finite gradient norm; optimizer not stepped"
                )
            lr_used = self.optimizer.param_groups[0]["lr"]
            with self._training_profile.range("optimizer"):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            self.optimizer_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            # log info-metrics
            metrics = {
                "loss": sum(x["loss"] for x in values) * backward_scale,
                "valid_anchors": sum(x["valid_anchors"] for x in values),
                "global_batch_size": len(samples) * self._world_size,
                "learning_rate": lr_used,
                "next_learning_rate": self.optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "ttt_update_rms": sum(x["ttt_update_rms"] for x in values)
                / microbatches,
            }
            if budget is not None:
                metrics["budget_windows"] = budget
                metrics["microbatches"] = microbatches
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            directory = Path(self.cfg.runner.logger.log_path)
            with (directory / f"sequence_samples_rank{self._rank}.jsonl").open(
                "a"
            ) as stream:
                stream.write(
                    json.dumps({"step": self.global_step + 1, "samples": samples})
                    + "\n"
                )
            if self._rank == 0:
                with (directory / "sequence_metrics.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({"step": self.global_step + 1, **metrics}) + "\n"
                    )
            return metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        directory = Path(save_path)
        marker = directory.parent / "_SUCCESS"
        if marker.exists():
            raise FileExistsError(
                f"Refusing to overwrite complete checkpoint: {directory.parent}"
            )
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            (directory / "sequence_metadata.json").write_text(
                json.dumps(self._sequence_metadata, indent=2)
            )
            (directory / "processor_bundle.json").write_text(
                json.dumps(self.data_config["processor"], indent=2)
            )
            torch.save(
                {
                    "step": step,
                    "optimizer_steps": self.optimizer_steps,
                    "microbatches_consumed": self._data_iter_offset,
                },
                directory / "sequence_progress.pt",
            )
        torch.distributed.barrier()
        if self._rank == 0:
            sizes = {
                str(path.relative_to(directory)): path.stat().st_size
                for path in directory.rglob("*")
                if path.is_file()
            }
            manifest = {"step": step, "files": sizes}
            validate_checkpoint_manifest(manifest, sizes)
            marker.write_text(json.dumps(manifest, indent=2))
        torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        directory = Path(load_path)
        manifest = json.loads((directory.parent / "_SUCCESS").read_text())
        sizes = {
            str(path.relative_to(directory)): path.stat().st_size
            for path in directory.rglob("*")
            if path.is_file()
        }
        validate_checkpoint_manifest(manifest, sizes)
        previous = json.loads((directory / "sequence_metadata.json").read_text())
        plan = self._sequence_metadata.get("resume_lr_schedule")
        migration = None
        if previous.get("resume_lr_schedule") is None and plan is not None:
            migration_path = self.cfg.runner.get("resume_lr_migration")
            if not migration_path:
                raise ValueError("The first LR migration requires a reviewed manifest")
            migration = json.loads(Path(migration_path).read_text())
        validate_resume_metadata(
            previous,
            self._sequence_metadata,
            migration=migration,
            checkpoint_identity={
                "path": str(directory.resolve()),
                "step": manifest["step"],
                "metadata_sha256": sha256_file(directory / "sequence_metadata.json"),
            },
        )
        progress = torch.load(directory / "sequence_progress.pt", weights_only=True)
        if progress["step"] != manifest["step"]:
            raise ValueError("Checkpoint completion step and progress disagree")
        super().load_checkpoint(load_path)
        if plan is not None:
            if self.lr_scheduler.last_epoch != progress["step"]:
                raise ValueError("Restored scheduler and checkpoint step disagree")
            apply_resume_lr_schedule(self.lr_scheduler, plan)
            logging.info(
                "Restored step %s with continuous LR %s; resume LR plan %s",
                progress["step"],
                self.lr_scheduler.get_last_lr(),
                plan,
            )
        self.optimizer_steps = progress["optimizer_steps"]
        self.global_step = progress["step"]
        self._data_iter_offset = progress["microbatches_consumed"]
