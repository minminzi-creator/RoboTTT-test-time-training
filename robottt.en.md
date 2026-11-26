# RoboTTT 3.0 — Implementation and Code Guide

Use this English guide to trace data sampling, Flow Matching supervision, TTT fast-weight updates, and simulation evaluation to their implementations. The [Chinese version](robottt.zh-CN.md) is available separately; both editions use the same equations, configuration names, and source references.

**Verified scope:** this document describes **Sequence32 / H16 / TBPTT4** in the server checkout `/mnt/public/chenzm/robottt/workspace/RLinf-robottt3.0`, verified on 2026-11-26. The H8 development variant in another checkout is outside this scope. Repository source links are relative; external dependency links point to the pinned upstream revision.

## 1. What This Code Implements

RoboTTT3.0 adds adaptive fast-weight memory to the action DiT of pretrained GR00T N1.7. RLinf supplies distributed training and evaluation, GR00T supplies vision-language encoding and action generation, and TTT compresses historical features into runtime `FastState` for later observations. The configured task is RoboCasa365 `BowlAndCup`.

“Implicit memory” means that history is represented by fast-weight values rather than a complete list of past images or actions. Implemented memory updates are not proof of learned task memory. Matched evaluation and ablations are needed to distinguish task success from a causal contribution of memory.

## 2. Where to Start Reading

Start with the experiment configuration, then follow the data, training orchestration, and model computation. These six files form the primary reading path.

| Order | File | Question Answered |
| --- | --- | --- |
| 1 | [TTT Training YAML][ttt-config] | Which data, sequence length, memory mode, and precision are used? |
| 2 | [sequence_dataset.py][dataset] | How are anchors selected, actions processed, and batches assembled? |
| 3 | [fsdp_gr00t_sequence_sft_worker.py][worker] | How are TBPTT, backward passes, and optimizer updates organized? |
| 4 | [sft.py][sft] | How are model loading, TTT forward routing, and loss normalization implemented? |
| 5 | [native_ttt.py][native] | How does TTT enter the DiT and carry memory? |
| 6 | [ttt.py][ttt] | How are fast-weight updates and readouts computed? |

[train_vla_sft.py][entry] creates the Ray cluster and sequence worker; [sft_runner.py][runner] drives the loop. The worker builds the DataLoader and model, executes forward/backward computation, and saves checkpoints when requested by the runner.

## 3. Two Independent Configurations

The two YAMLs contain their own settings and do not inherit from each other. Shared data settings do not imply identical training variables: the baseline uses TBPTT16, while TTT uses TBPTT4.

| Setting | GR00T Sequence Baseline | RoboTTT3.0 |
| --- | --- | --- |
| YAML | [robocasa365BowlandCup_sft_gr00tn1d7_sequence.yaml][baseline-config] | [robocasa365BowlandCup_ttt_sequence.yaml][ttt-config] |
| Data | BowlAndCup, 104 episodes | BowlAndCup, 104 episodes |
| Sequence | Up to 32 anchors, stride 16 | Up to 32 anchors, stride 16 |
| Supervised targets | H16 × 12 | H16 × 12 |
| Native action tensor | H40 × 132 | H40 × 132 |
| Per-GPU / global batch | 1 / 8, 8 GPUs | 1 / 8, 8 GPUs |
| TBPTT segment | 16 | 4 |
| TTT / registers | Off, 0 registers | On, 16 registers |
| Training memory | None | `supervised_noisy` |
| Current run settings | No resume directory; 50K schedule | Resume at 10K; compress the remaining cosine decay to finish at 30K |

**Resume configuration:** the TTT YAML retains the historical `resume_dir`, `resume_lr_migration`, and `resume_lr_schedule`. The combination of `runner.max_steps=30000` and `actor.optim.total_training_steps=50000` is intentional for this migration. `compressed_resume_lambda()` / `apply_resume_lr_schedule()` preserve the learning-rate connection at the resume point. Changing just one field does not create a fresh 30K schedule. Both YAMLs require an explicit `runner.logger.log_path`.

## 4. From Episodes to Training Sequences

An anchor is an observation starting point. `modality_dict()` selects three camera images, state, and language at that point, plus the H16 action target beginning there. Future images are not inputs to that anchor.

`PackedEpisodeSampler` distributes a deterministic episode stream. Within each episode, `window_indices()` chooses a random legal start and takes regularly spaced anchors at stride16. For episode length `L`, the valid count and start range are:

```python
n = min(32, (L - 16) // 16 + 1)
last_start = L - 16 - (n - 1) * 16
s = uniform_integer(0, last_start)  # inclusive; illustrative notation
anchors = [s + 16 * i for i in range(n)]
```

With `terminal_window_probability=0.0`, no extra terminal-alignment branch is applied, although a random start can still be the last legal start. For a short episode, the dataset duplicates the final processed payload to reach 32 slots and marks padded slots false in `anchor_valid` and `supervision`. These slots do not contribute loss, memory updates, or position advancement. Each new training window starts with `memory=None`; earlier episode history is not automatically warmed up.

The Processor normalizes H16×12 valid actions and pads them to H40×132. `action_mask` selects only the first 16 steps and 12 dimensions. `SequenceCollator` organizes the batch and time dimensions into model inputs and retains each sequence's valid-anchor count as its loss denominator; `segment_batch()` slices these inputs for TBPTT.

## 5. Training Forward and Supervised Loss

The model first obtains vision-language features and state representations through `_encode()`, then samples noise and time separately for the action target of each anchor. Here `A_t` denotes Processor-transformed model-space actions, not raw physical commands.

$$
\widetilde A_t=\tau_t A_t+(1-\tau_t)\epsilon_t,\qquad v_t^*=A_t-\epsilon_t.
$$

`_velocity()` returns the predicted velocity `prediction`, candidate memory `candidate`, and an update metric `update`. Training uses masked velocity MSE, normalized by the valid action elements per anchor and then by the supervised-anchor count of the full sequence. All TBPTT segments retain that full-sequence denominator, so splitting does not reweight sequences.

In the current `supervised_noisy` mode, `memory = candidate` carries the supervised forward's memory to the next anchor. The retained `predicted_final` branch instead samples predicted actions and then updates memory through `_write()`. This sketch explains the current path; it is not a drop-in replacement for the implementation.

```python
memory = None
for segment in split_into_tbptt_segments(batch, length=4):
    loss = 0
    for anchor in segment:
        prediction, candidate, update = velocity(anchor, memory)
        loss += masked_flow_loss(prediction, anchor, full_sequence_denominator)
        memory = candidate  # keep the graph within the segment
    backward(loss * gradient_accumulation_scale)
    memory = memory.detach()  # retain values; truncate the graph
# After all segments and microbatches: synchronize as configured, clip, AdamW step.
```

“One supervised DiT forward per anchor” describes logical forward calls. Activation checkpointing may recompute parts of the forward during backward.

## 6. Fast and Slow Weight Updates

Distinguish runtime memory from persistent model parameters. Each TTT layer has a `FastState` containing `w1/b1/w2/b2`; `Memory` groups these states with position counters. Runtime states travel through explicit inputs and outputs. Learned initial fast weights, projection layers, learning-rate parameters, and gates belong to model parameters.

| Update | Object | Signal | When |
| --- | --- | --- | --- |
| TTT inner update | Runtime FastState | Feature reconstruction target | During each valid-anchor forward |
| Outer AdamW | Trainable slow weights, including initial fast-weight parameters | Gradients of Flow Matching loss through the computation graph | After all segments and accumulated microbatches |

TTT projects hidden features into Q/K/V and uses `target = v - k`. Reconstruction errors of the fast-weight MLP on K produce analytical gradients; `dual_update()` and `scan_update()` compute updates and readouts. This update uses neither environment rewards nor a per-anchor AdamW call. See `reconstruction_grad()` and `dual_update()` for the exact normalization and learning-rate matrix operations.

The inner objective has the following form, with RoPE applied to K but not V. It drives fast-weight updates and is not added as a separate scalar term to the outer Flow Matching loss.

$$
f_W(k)=\operatorname{GELU}(kW_1+b_1)W_2+b_2,\qquad
\ell_{\mathrm{inner}}=\tfrac12\left\|\operatorname{LN}_{\gamma,\beta}(f_W(k))-(v-k)\right\|^2.
$$

Within an anchor, inner gradients for all tokens are computed at the same incoming weights. The triangular update gives query i an inclusive-prefix readout, while the final update row forms the next anchor's candidate. Queries do not all read the final candidate, and gradients are not recomputed at newly updated weights after every token. `NativeTTTBlock.forward()` adds the TTT residual after the attention residual and before the FFN.

The inner update remains in the differentiable graph. Within a TBPTT4 segment, a later anchor's Flow Matching loss can propagate through fast-state updates into earlier writes and related slow weights. Segment-boundary `detach()` retains values but cuts gradients across that boundary. TBPTT4 neither clears memory every four anchors nor invokes AdamW every four anchors.

A full window contains 32 anchors split into eight TBPTT4 segments. With one window per GPU and eight GPUs, global batch8 contributes up to 256 valid anchors to one AdamW update; padded sequence slots are excluded. FSDP synchronizes at the configured backward boundary, followed by gradient clipping and the optimizer step.

## 7. How Both Padding Axes Are Isolated

Time padding and action-dimension padding are separate issues. A loss mask alone does not isolate their effects on network features.

| Axis | Shape and Valid Range | Current Handling |
| --- | --- | --- |
| Action time | First H16 of H40 | Mask the last 24 action keys/values in every self-attention layer; restrict TTT to the active prefix |
| Action dimensions | First 12 of 132 | `_velocity()` zeros the last 120 dimensions before the action encoder |

The DiT tensor still has `16 Register + 1 State + 40 Action = 57` tokens. TTT operates on the active prefix of `16 + 1 + 16 = 33` and pads the remaining residual positions with zeros. RoPE uses the original 57-position coordinates before selecting the prefix, retaining the spacing between anchors. Computing `dual_update()` on 33 tokens makes its final update row correspond to the active prefix.

These action-time and action-dimension restrictions apply to the current TTT supervision and inference paths. Entire invalid-anchor masks are a third constraint operating on complete anchors, distinct from the two action-padding axes above.

Dimension isolation acts on the action encoder input; it does not force the decoder's final 120 output dimensions to zero. Generated noise and model-space outputs retain the H40×132 shape.

## 8. How Inference Carries Memory

[run_gr00t_native_trials.py][trials] organizes evaluation trials, and [eval_embodied_agent.py][eval-entry] launches environment and rollout workers through the RLinf runner. [gr00t_native_policy.py][policy] validates checkpoints/Processors and converts observations/actions. [rollout.py][rollout] owns memory by environment identity, while `native_ttt_predict()` generates actions.

One prediction encodes the current observation and performs four denoising steps using the same incoming memory to generate H40 actions. Intermediate denoising candidates are not successively committed. A final `_write()` uses the predicted actions to produce memory for the next round. Processor decoding and environment adaptation produce controls; the environment executes the first H8 before requesting another prediction.

Inference runs without outer AdamW or a training graph, while analytical inner fast-weight updates still run. `memory_mode=carry` preserves memory within an episode; `reset` is an ablation that resets it on every prediction. Starting a new episode resets its memory so different seeds/episodes do not share history.

Reset restores learned initial fast weights and resets the relevant position counters; it does not zero all fast weights. The `reset` ablation still retains token-prefix adaptation within the current chunk, so it is not equivalent to disabling TTT. One normal prediction uses four denoising forwards and one final write forward, but advances memory position only once. The rollout owner commits the candidate only after validation.

| Protocol | Training | Inference |
| --- | --- | --- |
| Write source | Supervised features from noisy action targets | Features of final predicted actions |
| Active TTT range | 33 tokens | 33 tokens |
| Valid action dimensions | 12 | 12 |
| Update interval | Anchor stride16 | Predict again after H8 execution |
| Memory boundary | Reset per window; truncate gradients at TBPTT4 | Reset per episode; the training-window boundary does not apply |

The active ranges are aligned, but write sources, experienced history lengths, and update intervals still differ. These are properties of the current implementation, not a claim of complete train/inference equivalence.

## 9. Source Index by Responsibility

Beyond the primary reading path, the following files handle model assembly, distributed execution, and evaluation. Inactive data branches are labeled separately.

| Area | File | Responsibility |
| --- | --- | --- |
| Assembly | [Model `__init__.py`][model-init] | Route sequence model construction through `sequence_training` |
| Assembly | [gr00t_action_model.py][action-model] | Generic GR00T RL adapter; sequence construction reuses its local-backbone redirect |
| Alternative data | [window_dataset.py][window-data], [window_sampling.py][window-sampling] | `independent_h40`; inactive under the current Sequence32 configuration |
| Worker bases | [fsdp_vla_sft_worker.py][vla-worker], [fsdp_sft_worker.py][sft-worker] | Shared VLA/SFT worker infrastructure |
| Distributed execution | [fsdp_model_manager.py][manager], [strategy/fsdp.py][strategy] | Model, optimizer, sharding, checkpoint saving and loading |
| Precision | [mixed_precision.py][precision], [fsdp_config.py][fsdp-config] | FSDP wrapping and per-module precision |
| Evaluation config | [bowlandcup_gr00t_native_ttt.yaml][eval-config] | Eight GPUs, seeds 7–14, H8 execution, memory mode |
| Evaluation worker | [gr00t_native_worker.py][eval-worker] | Scene and policy seeds, predictions, terminal records |
| Evaluation loop | [embodied_eval_runner.py][eval-runner] | Shared RLinf evaluation loop |
| Environment | [robocasa365_env.py][env], [robocasa365.yaml][env-config] | Simulator reset, step, and observation interfaces; environment defaults |
| Diagnostics | [check_gr00t_native_memory.py][memory-check] | Check memory updates, carry, and reset |
| Profiling | [run_robottt3_profile.sh][profile] | Isolated profiler launcher; preview by default |
| Acceptance | [run_robottt3_acceptance.py][acceptance] | Short training and checkpoint-resume checks |
| Historical records | [ROBOTTT3_DESIGN.md][design], [ROBOTTT3_PLAN.md][plan] | Current entry points and dated historical design/acceptance records |

The actual GR00T Processor, DiT, and vision-language backbone live in the external dependency. Read them to understand action normalization, padding, and decoding. The configuration pins dependency commit `23ace64f17aa5015259b8609d371eb61a357c776`.

| External File | Responsibility |
| --- | --- |
| [gr00t_n1d7.py][upstream-model] | Native model, action head, `sample_time()`, action encoding/decoding |
| [processing_gr00t_n1d7.py][processor] | Image and state processing, action normalization, padding, masks |
| [dit.py][dit] | Self-attention, cross-attention, FFN |
| [qwen3_backbone.py][backbone] | Vision-language features |
| [lerobot_episode_loader.py][loader] | Episode, Parquet, and video loading |

## 10. What to Check Before Running

Training requires the deployment's Python environment, importable RLinf and the pinned Isaac-GR00T, with this checkout first on `PYTHONPATH`. Model and dataset paths come from the training YAML. Decide fresh training versus resume, output directory, and the LR schedule before selecting one of the two `--config-name` values above, without the `.yaml` suffix.

The current FSDP setup uses FP32 master parameters, generally BF16 model compute parameters, FP32 gradient reduction, and an explicit FP32 policy for the `FastWeightTTT` subtree. Full model-weight saving is enabled. Strict resume also depends on the optimizer, scheduler, DataLoader cursor, RNG, and protocol identity; loading model weights alone is not strict resume. Consult the worker's `save_checkpoint()` / `load_checkpoint()`.

The evaluation launcher requires explicit checkpoint, output, Isaac-GR00T, RoboCasa, and robosuite paths. Use `--dry-run` for a plan or `--execute` for execution. It schedules eight seeds with three repetitions each, totaling 24 rollouts with at most 2400 environment actions each. SFT reads demonstrations, while simulation evaluation additionally requires a RoboCasa/robosuite environment that can construct PandaOmron.

Inspect `sequence_samples_rank*.jsonl` for actual sampled anchors, and `sequence_metrics.jsonl` plus TensorBoard for loss, learning rate, gradient norm, and `ttt_update_rms`. The latter monitors fast-weight changes; it is not memory accuracy. Evidence of useful implicit memory requires matched conditions such as carry/reset and task-decision/success measurements. Falling loss or nonzero FastState alone is insufficient.

This is a source guide, not a new runtime acceptance report. Preparing this document does not change the training/inference implementation or launch a run.

[ttt-config]: examples/sft/config/robocasa365BowlandCup_ttt_sequence.yaml
[baseline-config]: examples/sft/config/robocasa365BowlandCup_sft_gr00tn1d7_sequence.yaml
[entry]: examples/sft/train_vla_sft.py
[runner]: rlinf/runners/sft_runner.py
[dataset]: rlinf/data/datasets/gr00t_n1d7/sequence_dataset.py
[worker]: rlinf/workers/sft/fsdp_gr00t_sequence_sft_worker.py
[sft]: rlinf/models/embodiment/gr00t/gr00t_n1d7/sft.py
[native]: rlinf/models/embodiment/gr00t/gr00t_n1d7/native_ttt.py
[ttt]: rlinf/models/embodiment/gr00t/gr00t_n1d7/ttt.py
[rollout]: rlinf/models/embodiment/gr00t/gr00t_n1d7/rollout.py
[model-init]: rlinf/models/embodiment/gr00t/gr00t_n1d7/__init__.py
[action-model]: rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py
[window-data]: rlinf/data/datasets/gr00t_n1d7/window_dataset.py
[window-sampling]: rlinf/data/datasets/gr00t_n1d7/window_sampling.py
[vla-worker]: rlinf/workers/sft/fsdp_vla_sft_worker.py
[sft-worker]: rlinf/workers/sft/fsdp_sft_worker.py
[manager]: rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
[strategy]: rlinf/hybrid_engines/fsdp/strategy/fsdp.py
[precision]: rlinf/hybrid_engines/fsdp/mixed_precision.py
[fsdp-config]: rlinf/utils/fsdp_config.py
[eval-config]: evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml
[trials]: evaluations/run_gr00t_native_trials.py
[eval-entry]: evaluations/eval_embodied_agent.py
[policy]: evaluations/gr00t_native_policy.py
[eval-worker]: evaluations/gr00t_native_worker.py
[eval-runner]: rlinf/runners/embodied_eval_runner.py
[env]: rlinf/envs/sim/robocasa365/robocasa365_env.py
[env-config]: examples/embodiment/config/env/robocasa365.yaml
[memory-check]: evaluations/check_gr00t_native_memory.py
[profile]: examples/sft/run_robottt3_profile.sh
[acceptance]: examples/sft/run_robottt3_acceptance.py
[design]: ROBOTTT3_DESIGN.md
[plan]: ROBOTTT3_PLAN.md
[upstream-model]: https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/gr00t_n1d7.py
[processor]: https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
[dit]: https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/dit.py
[backbone]: https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/qwen3_backbone.py
[loader]: https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/data/dataset/lerobot_episode_loader.py
