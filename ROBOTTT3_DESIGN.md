# RoboTTT 3.0: Native Sequence32 Alignment

## Current training configuration entry points — 2026-11-26

Select either complete YAML with `examples/sft/train_vla_sft.py --config-name NAME`.
Both live in `examples/sft/config` and compose independently.

| NAME | Protocol | Run settings |
| --- | --- | --- |
| `robocasa365BowlandCup_sft_gr00tn1d7_sequence` | Sequence32, H16, stride16, TBPTT16, TTT disabled | Fresh initialization, 50K optimizer schedule |
| `robocasa365BowlandCup_ttt_sequence` | Sequence32, H16, stride16, TBPTT4, supervised-noisy memory, padding-aware TTT | Existing 10K checkpoint resume, remaining cosine phase compressed to 30K |

The TTT configuration retains H40 policy tensors, H16 memory range, action-dimension
masking, and FP32 fast-state policy. Both use eight ranks, microbatch1/global8,
and require an explicit output path. Each retains its own pre-consolidation
resolved values. Profiling and acceptance select the TTT name above. Evaluation
continues to use `evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml`.

### Configuration migration record

The old five YAMLs and the affected scripts/documents are preserved in
`.robocasa365BowlandCup_configs-bak_20261126T080306Z.tar.gz` at the repository root. The archive also contains the effective
configuration snapshots used for equivalence checks. Historical configuration
names and algorithm descriptions below describe earlier revisions; use the
two entry points above for the current checkout.

## Historical alignment (2026-11-04)

The user approved aligning the existing `RLinf-robottt3.0` checkout with
`/mnt/public/chenzm/robottt/train_gr00t_baseline_20261102/config.yaml` and its
`sequence32_h16_code` snapshot. This section supersedes the sequence128 protocol
below; historical design and launch evidence are retained, not reinterpreted as
verification of the new run.

- Train with `robocasa_bowlandcup_robottt3`, inheriting the new
  `robocasa_bowlandcup_gr00t_sequence32_aligned` control. Keep old sequence128
  configurations unchanged for historical reproduction.
- Same104 episodes and native assets; supervised H16/stride16/max32,
  terminal probability0, seed42. Eight ranks, microbatch1/global8. The user
  approved RoboTTT TBPTT4; the sequence32 baseline remains TBPTT16.
- Each RoboTTT sequence has eight4-anchor backward segments. Fast-state values
  carry across segments; only their gradient history is detached. The full
  sequence loss denominator and one optimizer update per global batch remain
  unchanged. Data, optimizer and training budget are aligned, but TBPTT and
  segment-wise supervision RNG consumption differ from the baseline. Reduced
  activation memory is expected, not guaranteed speedup or numerical stability.
  Strict resume rejects a TBPTT16 checkpoint under the new TBPTT4 protocol.
- H16 is the supervision horizon, not the model horizon. Processor tensors
  remain40x132; the action mask selects exactly the first16x12 values (192).
  Predictions and the TTT writer retain full H40x132, with16 registers +1 state
  +40 actions =57 tokens per anchor. Cropping the writer is an algorithm change
  outside this alignment.
- AdamW LR2e-5, betas .9/.95, eps1e-8, weight decay1e-5, clip1. Native `cosine`
  scheduler, warmup200, total50000, minimum5e-6 (ratio .25), zero first-update
  LR. The legacy `linear_warmup_cosine` is not this baseline's scheduler.
- Save steps10000,20000,30000,35000,40000,45000,50000 via `runner.save_steps`.
  Profiling defaults off. Output path remains null and must be agreed before a
  fresh run; this code alignment does not authorize a training launch.
- Evaluation still predicts40 frames, decodes16 using the saved processor and
  executes8. All decoded components must agree on batch size and horizon.
  Existing paired scene/trial and success protocols are unchanged.
- Preserve predicted-history writes, FP32 fast states, segment detach and strict
  source/protocol checks. Old sequence128 checkpoints cannot be resumed or
  evaluated with changed source under strict identity checks; retain their code
  snapshots. The metadata now records the actual supervised modality horizon.
- Current development connection is `ssh rlinf-40507`; historical port43545
  GPU acceptance below does not establish acceptance of this changed protocol.

## Historical sequence128 design and acceptance

Use this contract to keep RoboTTT training and evaluation aligned with the
native sequence128 baseline. The user approved this design and named
the independent local/server code directory `RLinf-robottt3.0` on 2026-10-19.
Base snapshot: `94db2a4a99ad3c771ef67c81ca447c4c9e0c5ca5` (native baseline
experiment snapshot over RLinf `7e90641c`). No baseline checkout is modified.

## Scientific contract

- Native Isaac-GR00T commit `23ace64f17aa5015259b8609d371eb61a357c776`;
  GR00T-N1.7-3B and Cosmos-Reason2-2B assets, not the legacy realworld-pni path.
- Same 104-episode data, processor, packed sequence sampler, augmentation,
  H40/stride8/max128, terminal probability 0, seed42, sequence-equal masked loss.
- Eight workers, microbatch1/global8/accumulation1; TBPTT16. AdamW LR2.5e-5,
  betas .9/.95, eps1e-8, weight decay1e-5, clip1; warmup200/cosine10000.
- Native backbone/head use the baseline FSDP BF16 compute/FP32 master and
  reduction policy. TTT learned initial fast weights and inner updates need
  explicit FP32 execution; initialization/reset must not silently quantize W0.
- Native noise-before-time sampling and state dropout .2 are retained; nn.Dropout
  remains disabled. New policy noise must not advance supervision RNG streams.
- TTT off follows the original native forward exactly, with no registers.
  TTT on retains the existing fast-weight mathematics and 16-register design;
  it is not claimed to be a no-op initialization. Include a registers-only
  structural control and carry/reset evaluation modes to isolate memory effects.

## State and gradient contract

Each anchor reads incoming memory. Compute native supervised flow loss and
discard its candidate. Generate four-step actions without gradients using the
same incoming memory. Stop-gradient only the predicted actions, then make one
differentiable final write at bucket999 and commit its candidate once. Future
anchor loss can differentiate through earlier writes within the segment;
backward precedes detach at segment boundaries. GT affects supervision, never
the predicted-history writer. Use full model-space H40x132 predictions.

Memory is explicit input/output, not mutable module state. It resets per
episode/environment; padding neither updates memory nor advances position.
Activation recomputation cannot commit. Rollout owns state, validates finite
values, and handles environment identity/reset explicitly. Four denoising calls
and the final writer all read the same incoming state.

## Boundaries

Preserve native dependency checkout. A narrow adapter at the native attention
residual/FFN boundary is allowed, with native-module parity tests; do not import
the legacy ExplicitDiT/flow training stack. Baseline checkpoint, data, weights,
environment and evaluation outputs are never overwritten. New formal runs must
start from original pretrained weights, not resume a baseline or legacy TTT
checkpoint under a new protocol. Formal training was subsequently approved with
checkpoints at3000,5000,6000,8000,9000,10000 only. Formal evaluation requires
separate approval; output directory names require user agreement.

## Acceptance

Native TTT-off output/loss/gradient parity; register-only parity with the native
augmented-token path; genuine fast-weight update and causal/padding/reset tests;
predicted-write GT independence and future-loss gradient; train/eval writer
equivalence with fixed RNG and dropout disabled; sequence weighting and cursor
resume unchanged; checkpoint metadata identifies algorithm and source; finite
state checks; native-module CPU tests plus explicit GPU/FSDP acceptance status.
Full-size GPU capacity, throughput, durable save/resume and 24-trial simulation
are separate execution evidence, never inferred from CPU tests.

## Historical code and configuration entry points

Server code root: `/mnt/public/chenzm/robottt/workspace/RLinf-robottt3.0`.
Training and inference host: `root@117.74.66.77`, SSH port43545, eight A800 80GB
devices0–7, as explicitly selected by the user. This replaces the initially
examined four-GPU port41766 host; both see the same mounted code directory.
Training entry: `examples/sft/train_vla_sft.py`, using one of these config names:

| Variant | Config under `examples/sft/config` | Memory |
| --- | --- | --- |
| Native sequence128 | `robocasa_bowlandcup_gr00t_sequence128_aligned` | None, no registers |
| Structural control | `robocasa_bowlandcup_gr00t_sequence128_registers` | None, 16 registers |
| RoboTTT3 | `robocasa_bowlandcup_robottt3` | Predicted-final history, 16 registers |

All three share data, optimizer, sampling and horizon. Their output path defaults
to null and must be chosen explicitly before running. The existing `h800_sft`
asset/dependency aliases resolve to the preserved baseline runtime and assets;
these are references, not copied model weights or data.

The historical baseline used four ranks and accumulation2. The user-selected
eight-rank setup preserves global batch8 but changes rank-local RNG consumption
and reduction order. It is not a bitwise continuation of the old run. Use the
three eight-rank controls together when making a strict architecture comparison;
strict checkpoint resume rejects a changed world size.

Training calls the sequence worker, then native `sft.build_model` and
`native_sequence_forward`. Only the opt-in adapter uses `native_ttt.py` and
`ttt.py`. The worker owns memory across TBPTT segments. Its source manifest
records the added modules and mixed-precision implementation, while strict
resume checks compare the actor/data protocol and source hashes.

Inference uses `predict_with_memory` with native processor output. Its returned
actions are the complete normalized model-space chunk, not RoboCasa controls.
`NativeTTTRollout` owns detached memory by stable environment ID, supports
carry/reset modes, and validates a candidate before committing.

The simulator call chain is `evaluations/run_gr00t_native_trials.py` →
`evaluations/eval_embodied_agent.py` → `NativeEvalRolloutWorker` →
`NativeEvalPolicy` → `NativeActionAdapter` → `NativeTTTRollout` →
`native_ttt_predict`. The policy performs native observation processing and
decodes the full H40x132 prediction before selecting the first eight controls.
The rollout worker resets memory for each episode, checks termination after
each action, and does not advance memory after an environment has finished.

Evaluation uses `evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml`:
eight environment/rollout pairs on devices0–7, one scene per GPU, seeds7–14 and
trials0–2. The queue plans three eight-step preflight batches followed by three
2400-step formal batches (24 formal episodes). Policy RNG uses
`scene_seed + 1000000 * trial_id`. Retain OSMesa rendering, physical-state
pairing and the baseline image-tolerance checks. Pass the preserved baseline
evaluation root via `--reference` to make a cross-model paired comparison;
without it, checks establish repeatability within the new run only.
The reference layout defaults to eight workers. Add `--reference-width 4`
when reading the preserved Shanghai four-worker baseline results.

Use `--dry-run` to inspect the plan without creating files, loading a model or
starting Ray. Execution requires `--execute`, a complete checkpoint and a fresh,
agreed output path. The loader checks the checkpoint manifest, source hashes,
saved model protocol and processor before strict weight loading. Its conversion
to BF16 excludes the FP32 fast-weight branch. Full-size model loading, eight-GPU
Ray placement, step1 save/resume through step2, and one eight-action RoboCasa
chunk on each of eight environments passed the bounded preflight on port43545.
See `ROBOTTT3_PLAN.md` for logs and resource measurements. This does not establish
uninterrupted/resumed bitwise equivalence, real-environment memory carry across
chunks, long-run stability or task success. The user-approved formal training
uses `results/robottt3.0/train`; the paired24-trial evaluation remains a separate
execution gate. See Task5 of the plan for the formal launch record.
