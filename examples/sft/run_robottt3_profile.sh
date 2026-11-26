#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
# Preview by default. Only --execute starts a bounded, fresh training probe.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python3 - "$@" <<'PY'
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

parser = argparse.ArgumentParser(description="Preview or run an isolated RoboTTT3 profiling probe.")
parser.add_argument("--mode", choices=("nsys", "torch"), required=True)
parser.add_argument("--gpus", required=True, help="Physical GPU indices, e.g. 4,5,6,7")
parser.add_argument("--output", required=True, help="New absolute output directory; agree its name before execution")
parser.add_argument("--python", required=True, help="Existing training environment Python")
parser.add_argument("--dependency", required=True, help="Existing pinned Isaac-GR00T checkout")
parser.add_argument("--nsys", default="nsys", help="Nsight Systems executable (nsys mode only)")
parser.add_argument("--start-step", type=int, default=18, help="Zero-based inclusive optimizer index")
parser.add_argument("--end-step", type=int, default=20, help="Exclusive optimizer index; also ends the probe")
parser.add_argument("--ranks", default="all", help="Capture ranks, e.g. 0,3; default all")
parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=None, help="Record tensor shapes (default: enabled for torch)")
parser.add_argument("--with-stack", action=argparse.BooleanOptionalAction, default=None, help="Record call stacks (default: enabled for torch)")
parser.add_argument("--execute", action="store_true", help="Without this flag, only validate arguments and print the plan")
args = parser.parse_args()
if args.record_shapes is None:
    args.record_shapes = args.mode == "torch"
if args.with_stack is None:
    args.with_stack = args.mode == "torch"

def fail(message):
    parser.error(message)

try:
    gpus = [int(value) for value in args.gpus.split(",")]
except ValueError:
    fail("--gpus must contain integer device indices")
if len(gpus) not in (4, 8) or len(set(gpus)) != len(gpus) or min(gpus) < 0:
    fail("select exactly four or eight distinct GPUs")
if not 0 <= args.start_step < args.end_step <= 10000:
    fail("require 0 <= start-step < end-step <= 10000")
ranks = None
if args.ranks != "all":
    try:
        ranks = [int(value) for value in args.ranks.split(",")]
    except ValueError:
        fail("--ranks must be all or comma-separated integers")
    if not ranks or len(set(ranks)) != len(ranks) or any(r < 0 or r >= len(gpus) for r in ranks):
        fail("--ranks must be distinct ranks within the selected world size")
if args.mode == "nsys" and (args.record_shapes or args.with_stack):
    fail("--record-shapes and --with-stack apply only to torch mode")
output = Path(args.output)
if not output.is_absolute() or not output.parent.is_dir():
    fail("--output must be absolute and its parent must already exist")
if output.exists() or output.is_symlink():
    fail("refusing to reuse or overwrite an existing --output path")
# Hydra interprets these characters as overrides, not literal path characters.
for value in (str(output), args.dependency):
    if any(char in value for char in "\n\r'\"[]{}=,$\\"):
        fail("paths must not contain Hydra override syntax characters")
root = Path.cwd()
rank_value = "null" if ranks is None else "[" + ",".join(map(str, ranks)) + "]"
command = [args.python, "-u", "examples/sft/train_vla_sft.py",
           "--config-name", "robocasa365BowlandCup_ttt_sequence",
           f"cluster.component_placement.actor='{','.join(map(str, gpus))}'",
           "actor.micro_batch_size=1", "actor.global_batch_size=8",
           f"actor.model.dependency_path='{args.dependency}'",
           f"runner.max_steps={args.end_step}", "runner.resume_dir=null",
           "runner.save_interval=-1", "runner.save_steps=[]", "runner.val_check_interval=-1",
           f"runner.logger.log_path='{output}'",
           f"actor.profiling.mode={args.mode}",
           f"actor.profiling.start_step={args.start_step}",
           f"actor.profiling.end_step={args.end_step}",
           f"actor.profiling.ranks={rank_value}",
           f"actor.profiling.output_dir='{output}'",
           f"actor.profiling.record_shapes={str(args.record_shapes).lower()}",
           f"actor.profiling.with_stack={str(args.with_stack).lower()}",
           f"cluster.profiling.enabled={str(args.mode == 'nsys').lower()}",
           f"cluster.profiling.output_dir='{output}'"]
plan = dict(command=command, physical_gpus=gpus, world_size=len(gpus),
            global_batch_size=8, micro_batch_size=1, gradient_accumulation=8 // len(gpus),
            window_zero_based=[args.start_step, args.end_step], mode=args.mode,
            checkpoint_saving=False, execute=args.execute)
print(json.dumps(plan, indent=2), flush=True)
if not args.execute:
    sys.exit(0)
if not Path(args.python).is_file() or not os.access(args.python, os.X_OK):
    fail("--python must be an existing executable")
if not (Path(args.dependency) / "gr00t").is_dir():
    fail("--dependency must contain the native gr00t package")
env = os.environ.copy()
if args.mode == "nsys":
    nsys = shutil.which(args.nsys)
    if nsys is None:
        fail("nsys not found; provide --nsys with its installed path")
    subprocess.run([nsys, "--version"], check=True)
    env["PATH"] = str(Path(nsys).resolve().parent) + os.pathsep + env.get("PATH", "")

def query(fields, kind):
    result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

# Fail closed on missing/unknown GPU state. This is a preflight, not a GPU reservation.
inventory = {int(row[0]): row for row in query("index,uuid,memory.used,utilization.gpu", "gpu")}
processes = query("gpu_uuid,pid", "compute-apps")
for index in gpus:
    if index not in inventory:
        fail(f"GPU {index} does not exist")
    row = inventory[index]
    if int(row[2]) > 64 or int(row[3]) != 0 or any(p[0] == row[1] for p in processes):
        fail(f"GPU {index} is not idle; no training started")

output.mkdir()  # Exclusive creation also prevents concurrent reuse of this run name.
runtime = tempfile.mkdtemp(prefix="r3p-", dir="/tmp")
env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), CUDA_DEVICE_ORDER="PCI_BUS_ID",
           RAY_ADDRESS="local", RAY_TMPDIR=runtime, RAY_USAGE_STATS_ENABLED="0",
           RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES="4294967296", RLINF_CODE_WORKING_DIR="0",
           PYTHONPATH=str(root) + os.pathsep + str(Path(args.dependency).resolve()),
           PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", USE_TF="0",
           NO_ALBUMENTATIONS_UPDATE="1", TOKENIZERS_PARALLELISM="false",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HYDRA_FULL_ERROR="1",
           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
           TORCHINDUCTOR_CACHE_DIR=str(output / "torchinductor_cache"),
           TRITON_CACHE_DIR=str(output / "triton_cache"))
plan["ray_tmpdir"] = runtime
plan["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
(output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
with (output / "source_changes.patch").open("wb") as patch:
    subprocess.run(["git", "diff", "HEAD"], stdout=patch, check=True)
training = None
stop_signal = None
stop_deadline = None
stop_stage = 0

def signal_owned(signum):
    if training is not None and training.poll() is None:
        try:
            os.killpg(training.pid, signum)
        except ProcessLookupError:
            pass

def request_stop(signum, frame):
    global stop_signal, stop_deadline
    if stop_signal is None:
        stop_signal = signum
        stop_deadline = time.monotonic() + 60
        # SIGINT lets Hydra/Python unwind and Ray tear down its owned workers.
        signal_owned(signal.SIGINT)

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)
with (output / "train.log").open("x") as log:
    try:
        training = subprocess.Popen(command, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        if stop_signal is not None:
            signal_owned(signal.SIGINT)
        while True:
            try:
                exit_code = training.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if stop_deadline is not None and time.monotonic() >= stop_deadline:
                    signal_owned(signal.SIGTERM if stop_stage == 0 else signal.SIGKILL)
                    stop_stage += 1
                    stop_deadline = time.monotonic() + 10
    finally:
        if training is not None and training.poll() is None:
            signal_owned(signal.SIGINT)
            try:
                training.wait(timeout=60)
            except subprocess.TimeoutExpired:
                signal_owned(signal.SIGKILL)
                training.wait()
plan["exit_code"] = exit_code
plan["stop_signal"] = stop_signal
(output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
sys.exit(128 + stop_signal if stop_signal is not None else exit_code)
PY
