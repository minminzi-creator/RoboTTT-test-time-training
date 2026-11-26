# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Strict checkpoint-bound, no-grad native TTT carry/causality/reset acceptance."""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

EVAL_SOURCES = (
    "evaluations/check_gr00t_native_memory.py",
    "evaluations/gr00t_native_policy.py",
    "evaluations/gr00t_native_worker.py",
    "evaluations/run_gr00t_native_trials.py",
    "evaluations/eval_embodied_agent.py",
    "evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml",
)
REQUIRED_CHECKS = (
    "strict_policy",
    "forward_protocol",
    "carry",
    "causal_effect",
    "reset",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes(root, names):
    root = Path(root).resolve()
    result = {}
    for name in sorted(set(names)):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaping identity source: {name}")
        result[name] = _sha(path)
    return result


def report_identity(checkpoint, model_cfg_dict, repo) -> dict:
    """Stdlib-only identity; never import torch or load checkpoint tensors."""
    weights, repo = Path(checkpoint).resolve(), Path(repo).resolve()
    actor = weights.parent.parent
    metadata = json.loads((actor / "sequence_metadata.json").read_text())
    dependency = Path(model_cfg_dict["dependency_path"]).resolve()
    dependency_names = [
        str(p.relative_to(dependency)) for p in (dependency / "gr00t").rglob("*.py")
    ]
    if not dependency_names:
        raise ValueError("Missing pinned GR00T dependency source")
    dependency_names += list(metadata["source"].get("gr00t_sha256", {}))
    stat = weights.stat()
    return {
        "checkpoint": str(weights),
        "repo": str(repo),
        "weight_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "checkpoint_sha256": {
            str(p.relative_to(actor.parent)): _sha(p)
            for p in (
                actor.parent / "_SUCCESS",
                actor / "sequence_metadata.json",
                actor / "processor_bundle.json",
            )
        },
        "model_config": json.loads(json.dumps(model_cfg_dict, allow_nan=False)),
        "source_sha256": _source_hashes(
            repo, [*metadata["source"]["source_sha256"], *EVAL_SOURCES]
        ),
        "dependency_sha256": _source_hashes(dependency, dependency_names),
    }


def validate_report(report_path, checkpoint, model_cfg_dict, repo) -> dict:
    """Fail closed on absent, failed, incomplete or stale memory acceptance."""
    try:
        report = json.loads(Path(report_path).read_text())
        if report.get("version") != 1 or report.get("passed") is not True:
            raise ValueError("Memory report is not a passing version-1 report")
        if any(
            report.get("checks", {}).get(key, {}).get("passed") is not True
            for key in REQUIRED_CHECKS
        ):
            raise ValueError("Memory report lacks required passing checks")
        if report.get("identity") != report_identity(checkpoint, model_cfg_dict, repo):
            raise ValueError(
                "Memory report identity differs from checkpoint/config/source"
            )
        return report
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid memory acceptance report: {error}") from error


def _difference(a, b):
    import torch

    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Comparison requires matching, finite tensors")
    delta = (a.detach().float() - b.detach().float()).abs()
    return {
        "max_abs": float(delta.max()),
        "rms": float(delta.square().mean().sqrt()),
        "changed": int(torch.count_nonzero(delta)),
        "scale": float(torch.maximum(a.abs().max(), b.abs().max())),
    }


def _check_memory(memory, position):
    import torch

    if (
        memory.position.dtype != torch.long
        or memory.position.tolist() != [position]
        or not memory.layers
    ):
        raise ValueError("Memory position must advance exactly once per chunk")
    for layer in memory.layers:
        if len(layer) != 4:
            raise ValueError("Expected four fast-state tensors per layer")
        for value in layer:
            if (
                value.dtype != torch.float32
                or value.ndim != 4
                or value.shape[0] != 1
                or not value.numel()
            ):
                raise TypeError("Expected nonempty FP32 [1,H,D,D] fast memory")
            if (
                value.requires_grad
                or value.grad_fn is not None
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Fast memory must be detached and finite")


def _memory_difference(a, b):
    if len(a.layers) != len(b.layers):
        raise ValueError("Memory layer counts differ")
    rows = [_difference(x, y) for l, r in zip(a.layers, b.layers) for x, y in zip(l, r)]
    return {
        "max_abs": max(r["max_abs"] for r in rows),
        "scale": max(r["scale"] for r in rows),
        "changed": sum(r["changed"] for r in rows),
    }


def _tolerance(repeat):
    # Four times observed repeatability plus one FP32 relative rounding unit.
    return 4 * repeat["max_abs"] + (2**-23) * repeat["scale"]


def _require_same_memory(actual, expected):
    if (
        actual is None
        or actual.position.tolist() != expected.position.tolist()
        or _memory_difference(actual, expected)["max_abs"] != 0
    ):
        raise ValueError("Rollout must carry and commit the exact fast-state values")


def run_checks(policy, env_obs, scene, device, checks):
    """Exercise the actual policy and rollout; hooks observe, never replace kernels."""
    import torch

    from evaluations.gr00t_native_policy import convert_action, convert_observation
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import Memory

    model, head, rollout = policy.model, policy.model.action_head, policy.rollout
    identity = policy.identity
    checks["strict_policy"] = {
        "passed": bool(
            identity["strict_load"]
            and identity["finite_weights"]
            and identity["use_ttt"]
            and identity["parameter_dtypes"] == ["torch.bfloat16", "torch.float32"]
            and identity["buffer_dtypes"] == ["torch.float32"]
            and identity["denoising_steps"] == 4
            and identity["inference_active_tokens"] == 33
            and identity["inference_memory_action_horizon"] == 16
            and policy.chunks == 8
        ),
        "policy": identity,
    }
    if not checks["strict_policy"]["passed"] or rollout is None:
        raise ValueError(
            "Expected strict native BF16/TTT FP32 H16, active33, four-step carry policy"
        )
    observations = convert_observation(env_obs)
    inputs = policy.processor.process_observation(observations, policy.embodiment)
    state = {k[6:]: v for k, v in observations.items() if k.startswith("state.")}
    seeds = [scene + 10_000 * i for i in range(3)]
    noises = [
        torch.randn(
            (1, 40, 132),
            device=device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        for seed in seeds
    ]
    protocol = []

    def predict(call, expected_memory=None):
        rows, velocities, buckets, final = [], [], [], {}

        def before(module, args, kwargs):
            memory = kwargs["memory"]
            if not rows and expected_memory is not None:
                _require_same_memory(memory, expected_memory)
            versions = (
                []
                if memory is None
                else [x._version for layer in memory.layers for x in layer]
            )
            rows.append(
                {
                    "memory_id": id(memory),
                    "position": [0] if memory is None else memory.position.tolist(),
                    "active_tokens": kwargs.get("active_tokens"),
                    "versions": versions,
                }
            )
            buckets.append(kwargs["timestep"].tolist())
            if torch.is_grad_enabled():
                raise ValueError("Memory check requires pure no_grad")

        def after(module, args, kwargs, output):
            incoming = kwargs["memory"]
            if incoming is not None and (
                incoming.position.tolist() != rows[-1]["position"]
                or [x._version for layer in incoming.layers for x in layer]
                != rows[-1]["versions"]
            ):
                raise ValueError("A DiT forward mutated its incoming memory")
            candidate = output[1]
            if candidate.position.tolist() != [rows[-1]["position"][0] + 1]:
                raise ValueError("DiT candidate must advance one anchor")
            final["pointers"] = [
                x.data_ptr() for layer in candidate.layers for x in layer
            ]

        def decoded(module, args, output):
            if not velocities:
                velocities.append(output[:, -40:, :12][:, :16].detach().float().clone())

        handles = [
            head.model.register_forward_pre_hook(before, with_kwargs=True),
            head.model.register_forward_hook(after, with_kwargs=True),
            head.action_decoder.register_forward_hook(decoded),
        ]
        try:
            result = call()
        finally:
            for handle in handles:
                handle.remove()
        valid = len(rows) == 5 and all(
            row == rows[0] and row["active_tokens"] == 33 for row in rows
        )
        valid &= buckets == [
            [int(i / 4 * head.num_timestep_buckets)] for i in range(4)
        ] + [[head.num_timestep_buckets - 1]]
        valid &= [
            x.data_ptr() for layer in result["memory"].layers for x in layer
        ] == final["pointers"]
        if (
            not valid
            or result["actions"].shape != (1, 40, 132)
            or velocities[0].shape != (1, 16, 12)
        ):
            raise ValueError(
                "Expected four reads of unchanged memory and one final committed writer"
            )
        _check_memory(result["memory"], rows[0]["position"][0] + 1)
        if (
            not torch.isfinite(result["actions"]).all()
            or result["actions"].requires_grad
        ):
            raise ValueError("Predicted actions must be detached and finite")
        protocol.append(
            {
                "calls": len(rows),
                "incoming_position": rows[0]["position"],
                "same_incoming_memory": True,
                "active_tokens": 33,
                "buckets": buckets,
                "final_candidate_returned": True,
            }
        )
        return result, velocities[0]

    def direct(memory, noise):
        return predict(
            lambda: model.predict_with_memory(inputs, memory=memory, policy_noise=noise)
        )

    def carry(noise, expected_memory=None):
        result, velocity = predict(
            lambda: rollout.predict(inputs, [scene], policy_noise=noise),
            expected_memory,
        )
        _require_same_memory(rollout._memory, result["memory"])
        return result, velocity

    def decode(result):
        decoded = policy.processor.decode_action(
            result["actions"].float().cpu().numpy(), policy.embodiment, state=state
        )
        value = torch.from_numpy(convert_action(decoded, 8, policy.threshold))
        if value.shape != (1, 8, 12):
            raise ValueError("Expected decoded [1,8,12] actions")
        return value

    with torch.no_grad():
        rollout.reset()
        first, _ = carry(noises[0])
        initial = Memory(
            tuple(
                block.ttt.initial_state(1) for block in head.model.transformer_blocks
            ),
            torch.zeros(1, dtype=torch.long, device=device),
        )
        changes = [_memory_difference(first["memory"], initial)]
        previous = first["memory"]
        for index in (1, 2):
            result, _ = carry(noises[index], previous)
            _check_memory(result["memory"], index + 1)
            changes.append(_memory_difference(result["memory"], previous))
            previous = result["memory"]
        carried = previous
        checks["carry"] = {
            "passed": all(row["max_abs"] > 0 for row in changes),
            "positions": [1, 2, 3],
            "memory_changes": changes,
            "policy_noise_seeds": seeds,
        }
        # Identical position removes the RoPE-position confound from the ablation.
        neutral = Memory(initial.layers, carried.position.detach().clone())
        a, va = direct(carried, noises[2])
        b, vb = direct(carried, noises[2])
        c, vc = direct(neutral, noises[2])
        comparisons = {}
        for name, x, repeat, ablated in (
            ("velocity_h16x12", va, vb, vc),
            (
                "model_actions_h8x12",
                a["actions"][:, :8, :12],
                b["actions"][:, :8, :12],
                c["actions"][:, :8, :12],
            ),
            ("decoded_actions_h8x12", decode(a), decode(b), decode(c)),
        ):
            floor, effect = _difference(x, repeat), _difference(x, ablated)
            comparisons[name] = {
                "repeatability": floor,
                "effect": effect,
                "tolerance": _tolerance(floor),
                "beyond_repeatability": effect["max_abs"] > _tolerance(floor),
            }
        checks["causal_effect"] = {
            "passed": comparisons["model_actions_h8x12"]["beyond_repeatability"]
            and comparisons["decoded_actions_h8x12"]["beyond_repeatability"],
            "position_equal": carried.position.tolist() == neutral.position.tolist(),
            "position": carried.position.tolist(),
            "comparisons": comparisons,
        }
        # Measure fresh-call repeatability before assessing reset reproduction.
        fresh, _ = direct(None, noises[0])
        action_floor = _difference(first["actions"], fresh["actions"])
        memory_floor = _memory_difference(first["memory"], fresh["memory"])
        rollout.reset()
        cleared = (
            rollout._memory is None
            and rollout._env_ids is None
            and not rollout._pending_resets
        )
        reset, _ = carry(noises[0])
        action_delta = _difference(first["actions"], reset["actions"])
        memory_delta = _memory_difference(first["memory"], reset["memory"])
        checks["reset"] = {
            "passed": cleared
            and reset["memory"].position.tolist() == [1]
            and action_delta["max_abs"] <= _tolerance(action_floor)
            and memory_delta["max_abs"] <= _tolerance(memory_floor),
            "cleared": cleared,
            "position": reset["memory"].position.tolist(),
            "actions": action_delta,
            "memory": memory_delta,
            "action_repeatability": action_floor,
            "memory_repeatability": memory_floor,
            "action_tolerance": _tolerance(action_floor),
            "memory_tolerance": _tolerance(memory_floor),
        }
    checks["forward_protocol"] = {"passed": True, "predictions": protocol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "reference", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene-seed", type=int, default=7, choices=range(7, 15))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    report = {"version": 1, "passed": False, "identity": None, "checks": {}}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config).rollout.model
        model_cfg = OmegaConf.to_container(cfg, resolve=True)
        report["identity"] = report_identity(args.checkpoint, model_cfg, repo)
        sys.path[:0] = [str(repo), str(Path(cfg.dependency_path).resolve())]
        import numpy as np
        import torch

        from evaluations.gr00t_native_policy import (
            NativeEvalPolicy,
            observation_hash,
            reference_for_scene,
        )

        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("Specify an explicit CUDA device, for example cuda:0")
        torch.cuda.set_device(device)
        reference, rank = reference_for_scene(
            Path(args.reference) / "preflight/trial_0", args.scene_seed, 4
        )
        reset_path, audit_path = (
            reference / f"reset_rank{rank}.npz",
            reference / f"trial_audit_rank{rank}.json",
        )
        audit = json.loads(audit_path.read_text())
        with np.load(reset_path, allow_pickle=False) as saved:
            env_obs = {key: saved[key].copy() for key in saved.files}
        digest = observation_hash(env_obs)
        if (
            audit["scene_seed"] != args.scene_seed
            or audit["initial_observation_sha256"] != digest
        ):
            raise ValueError("Reference observation disagrees with baseline audit")
        report["audit"] = {
            "reference": str(reference.resolve()),
            "rank": rank,
            "scene_seed": args.scene_seed,
            "reset_sha256": _sha(reset_path),
            "audit_sha256": _sha(audit_path),
            "observation_sha256": digest,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        policy = NativeEvalPolicy(
            cfg, str(Path(args.checkpoint).resolve()), device, memory_mode="carry"
        )
        run_checks(policy, env_obs, args.scene_seed, device, report["checks"])
        if report["identity"] != report_identity(args.checkpoint, model_cfg, repo):
            raise ValueError("Checkpoint/config/source changed during memory check")
        report["passed"] = all(
            report["checks"].get(key, {}).get("passed") is True
            for key in REQUIRED_CHECKS
        )
        if not report["passed"]:
            raise ValueError("One or more required memory checks failed")
    except Exception as error:  # noqa: BLE001 - every failed acceptance must emit a failed report.
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "output": str(output),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
