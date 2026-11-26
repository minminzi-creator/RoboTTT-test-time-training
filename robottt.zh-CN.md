# RoboTTT — 实现与代码导读

[English version](robottt.en.md)

通过这份说明，你可以从数据采样一路追踪到 Flow Matching 监督、TTT 快权重更新和仿真评测，并找到每一步对应的代码。本文件为中文版，包含实现讲解、关键公式和源码索引。仓库内源码使用相对链接；外部依赖链接指向固定版本的上游源码。

## 1. 这份代码实现什么

RoboTTT3.0 在预训练 GR00T N1.7 的动作 DiT 中加入可更新的快权重记忆。RLinf 提供分布式训练与评测框架；GR00T 提供视觉语言编码和动作生成；TTT 模块把历史特征压缩进运行时 `FastState`，供后续观测使用。当前任务为 RoboCasa365 `BowlAndCup`。

这里的“隐式记忆”指历史存储在快权重数值中，完成记忆读写。实现了记忆读写，不等于已经证明机器人学会了任务所需的记忆；成功率和记忆贡献需要通过同协议评测与消融区分。

## 2. 从哪里开始读

先看配置明确实验，再按数据、训练调度和模型计算的顺序阅读。以下六处组成训练主线。

| 顺序  | 文件                                                                                                                                             | 要回答的问题                     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| 1   | [TTT 训练配置](examples/sft/config/robocasa365BowlandCup_ttt_sequence.yaml)                  | 用什么数据、序列长度、记忆模式和精度？        |
| 2   | [sequence_dataset.py](rlinf/data/datasets/gr00t_n1d7/sequence_dataset.py)                | 如何选择 Anchor、处理动作并组成 batch？ |
| 3   | [fsdp_gr00t_sequence_sft_worker.py](rlinf/workers/sft/fsdp_gr00t_sequence_sft_worker.py) | 如何切分 TBPTT、反向传播并更新模型？      |
| 4   | [sft.py](rlinf/models/embodiment/gr00t/gr00t_n1d7/sft.py)                                | 如何加载模型、选择 TTT 前向并归一化 loss？ |
| 5   | [native_ttt.py](rlinf/models/embodiment/gr00t/gr00t_n1d7/native_ttt.py)                  | TTT 怎样进入 DiT，Memory 怎样递推？  |
| 6   | [ttt.py](rlinf/models/embodiment/gr00t/gr00t_n1d7/ttt.py)                                | 快权重内部更新和读取怎么算？             |

训练入口为 [train_vla_sft.py](examples/sft/train_vla_sft.py)，它创建 Ray Cluster 和 sequence worker，由 [sft_runner.py](rlinf/runners/sft_runner.py) 驱动循环。worker 构建 DataLoader 与模型，每步调用模型前向和反向，runner 按计划要求保存 checkpoint。

## 3. 两份独立配置

这两个 YAML 已展开各自参数，互不继承。共享数据协议不代表所有训练变量相同；尤其要区分 baseline 的 TBPTT16 和 TTT 的 TBPTT4。

| 项目                      | GR00T Sequence Baseline                                                                                                                                                          | RoboTTT3.0                                                                                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| YAML                    | [robocasa365BowlandCup_sft_gr00tn1d7_sequence.yaml](examples/sft/config/robocasa365BowlandCup_sft_gr00tn1d7_sequence.yaml) | [robocasa365BowlandCup_ttt_sequence.yaml](examples/sft/config/robocasa365BowlandCup_ttt_sequence.yaml) |
| 数据                      | BowlAndCup，104 episodes                                                                                                                                                          | BowlAndCup，104 episodes                                                                                                                                      |
| 序列                      | 最多 32 个 Anchor，stride16                                                                                                                                                          | 最多 32 个 Anchor，stride16                                                                                                                                      |
| 有效监督                    | H16 × 12                                                                                                                                                                         | H16 × 12                                                                                                                                                     |
| 原生动作张量                  | H40 × 132                                                                                                                                                                        | H40 × 132                                                                                                                                                    |
| 每卡 batch / global batch | 1 / 8，8 GPUs                                                                                                                                                                     | 1 / 8，8 GPUs                                                                                                                                                 |
| TBPTT segment           | 16                                                                                                                                                                               | 4                                                                                                                                                            |
| TTT / Register          | 关闭，0 个 Register                                                                                                                                                                  | 开启，16 个 Register                                                                                                                                             |
| 训练记忆                    | 无                                                                                                                                                                                | `supervised_noisy`                                                                                                                                           |
| 当前运行设置                  | 无恢复目录；50K 计划                                                                                                                                                                     | 从 10K 续训；将剩余 cosine 衰减压缩至 30K                                                                                                                                |

**续训配置说明：** TTT YAML 保留历史实验的 `resume_dir`、`resume_lr_migration` 和 `resume_lr_schedule`；`compressed_resume_lambda()` / `apply_resume_lr_schedule()` 负责连接断点处学习率。两份 YAML 的 `runner.logger.log_path` 都需要显式指定。

## 4. 数据怎样变成训练序列

一个 Anchor 是某个时刻的观测起点。`modality_dict()` 在该时刻读取三相机图像、状态与语言，并取从该时刻开始的 H16 动作标签。图像只取当前观测，不把未来图像作为该 Anchor 的输入。

`PackedEpisodeSampler` 分配确定性的 episode 顺序。`window_indices()` 在 episode 内选取随机合法起点，然后按 stride16 连续取 Anchor。设 episode 长度为 `L`，有效 Anchor 数与起点范围为：

```python
n = min(32, (L - 16) // 16 + 1)
last_start = L - 16 - (n - 1) * 16
s = uniform_integer(0, last_start)  # inclusive; illustrative notation
anchors = [s + 16 * i for i in range(n)]
```

`terminal_window_probability=0.0` 表示没有额外强制末尾对齐的采样分支；随机起点仍可能落在末端合法位置。短轨迹不足 32 个 Anchor 时，Dataset 复制最后一份处理后的 payload 补齐张量，但将补齐位置的 `anchor_valid` 和 `supervision` 设为 false。这些位置不应产生 loss、记忆更新或位置推进。每个新训练窗口从 `memory=None` 开始，窗口之前的历史没有自动预热。

Processor 把有效动作 H16×12 归一化并补齐为 H40×132，`action_mask` 只覆盖前 16 步、前 12 维。`SequenceCollator` 将 batch 和时间维组织成模型输入，保留每条序列的有效 Anchor 数作为 loss 分母；`segment_batch()` 再按 TBPTT 长度切片。

## 5. 训练 Forward 与监督 Loss

模型先用 `_encode()` 得到视觉语言特征和状态表示，再对各 Anchor 的动作标签独立采样噪声和时间。这里 `A_t` 已是 Processor 处理后的模型空间动作，不是原始物理单位的动作。

$$
\widetilde A_t=\tau_t A_t+(1-\tau_t)\epsilon_t,\qquad v_t^*=A_t-\epsilon_t.
$$

`_velocity()` 返回预测速度 `prediction`、更新候选 `candidate` 和监控量 `update`。训练用有效动作位置上的速度 MSE，先在每个 Anchor 内按有效元素数归一化，再除以整条序列的有效监督 Anchor 数。每个 TBPTT 片段沿用同一个整序列分母，避免片段数量改变各序列的权重。

当前 `supervised_noisy` 模式直接执行 `memory = candidate`，让下一 Anchor 使用监督前向产生的记忆。`predicted_final` 分支仍保留：先额外采样预测动作，再 `_write()` 更新 Memory。下面是当前主路径的简化表示，不是可以直接替换源码的代码。

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

“每个 Anchor 一次监督 DiT forward”描述逻辑前向次数；开启 activation checkpointing 后，反向阶段仍可能重算部分前向。

## 6. Fast Weights 与 Slow Weights 怎样更新

需要区分运行时记忆和持久模型参数。`FastState` 包含每个 TTT 层的 `w1/b1/w2/b2`；`Memory` 组织各层 FastState 及位置计数。它们通过显式参数和返回值传递。可学习的初始快权重、投影层、学习率相关参数和 gate 则属于模型参数。

| 更新 | 对象 | 驱动信号 | 发生时机 |
| --- | --- | --- | --- |
| TTT inner update | 运行时 FastState | 特征重构目标 | 有效 Anchor 的前向计算 |
| Outer AdamW | 可训练 slow weights，包括初始快权重参数 | Flow Matching loss 经计算图反传 | 所有片段与累积 microbatch 完成后 |

TTT 从隐藏特征生成 Q/K/V，内部目标为 `target = v - k`。快权重 MLP 对 K 的重构误差产生解析梯度，经 `dual_update()` 和 `scan_update()` 形成更新及读出。它不是用环境奖励更新，也不是每一步调用 AdamW。精确的归一化和学习率矩阵运算见 `reconstruction_grad()` 与 `dual_update()`。

内目标可写为下面的形式，其中 K 已经应用 RoPE，V 未应用 RoPE。这个内目标用于计算快权重更新，并未作为另一个标量 loss 直接加到外层 Flow Matching loss 上。

$$
f_W(k)=\mathrm{GELU}(kW_1+b_1)W_2+b_2,\qquad
\ell_{\mathrm{inner}}=\tfrac12\left\|\mathrm{LN}_{\gamma,\beta}(f_W(k))-(v-k)\right\|^2.
$$

同一个 Anchor 内，各 token 的内梯度在同一份传入快权重上计算；下三角更新使第 i 个 query 读取包含自身的前缀更新，最后一行更新形成下一 Anchor 的 candidate。并非所有 query 都读取最终 candidate，也不是每处理一个 token 就用新权重重新计算下一个 token 的梯度。`NativeTTTBlock.forward()` 在 attention residual 之后加入 TTT residual，再进入 FFN。

内更新本身处于可微计算图中，因此同一 TBPTT4 片段内，后续 Anchor 的 Flow Matching loss 可以经 FastState 更新反传到更早的写入操作及相关 slow weights。片段边界 `detach()` 保留记忆数值，但切断跨边界梯度。TBPTT4 不等于每 4 个 Anchor 清空记忆，也不等于每 4 个 Anchor 执行一次 AdamW。

满长度窗口含 32 个 Anchor，分成 8 个 TBPTT4 片段。当前八卡每卡一个窗口，global batch8；一次 AdamW 最多汇总 256 个有效 Anchor 的监督，短序列补齐位置不计入。FSDP 在指定的反向边界进行同步，随后裁剪梯度并更新模型。

## 7. 两种 Padding 怎样隔离

时间 Padding 和动作维度 Padding 是两个独立问题，loss mask 本身不足以隔离它们对网络特征的影响。

| 轴 | 张量与有效范围 | 当前处理 |
| --- | --- | --- |
| 动作时间 | H40 中前 H16 有效 | 每层 Self-Attention 禁止读取后 24 个动作 K/V；TTT 只处理有效前缀 |
| 动作维度 | 132 中前 12 维有效 | `_velocity()` 在 action encoder 前将后 120 维置零 |

DiT 的张量长度仍是 `16 Register + 1 State + 40 Action = 57`。TTT 的有效前缀是 `16 + 1 + 16 = 33`，其余位置的 TTT residual 补零。RoPE 使用原 57 个位置的时间坐标再取前缀，保持相邻 Anchor 的位置跨度。`dual_update()` 在有效 33-token 前缀上计算，因此提交状态自然使用有效前缀的最后一行更新。

这些动作时间与维度限制在当前 TTT 监督路径和推理路径均开启。序列中整段无效 Anchor 的 mask 是第三类约束，作用于整个 Anchor，不能与上述两个 Padding 轴混淆。

维度隔离发生在 action encoder 输入端，不代表 decoder 的后 120 维输出也被强制置零；生成噪声和模型空间输出仍保留 H40×132 形状。

## 8. 推理怎样传递记忆

评测从 [run_gr00t_native_trials.py](evaluations/run_gr00t_native_trials.py) 组织试验，经 [eval_embodied_agent.py](evaluations/eval_embodied_agent.py) 和 RLinf runner 启动环境与 rollout worker。[gr00t_native_policy.py](evaluations/gr00t_native_policy.py) 校验 checkpoint/Processor 并转换观测和动作；[rollout.py](rlinf/models/embodiment/gr00t/gr00t_n1d7/rollout.py) 按环境身份管理 Memory；`native_ttt_predict()` 执行动作预测。

一次预测先编码当前观测，再用同一份传入 Memory 完成 4 步去噪，得到 H40 动作。去噪过程中不把每次产生的候选状态连续提交。最终预测动作经过一次 `_write()`，产生下一轮 Memory；动作经 Processor 解码和环境适配后执行前 H8，再读取新观测。

推理不执行外部 AdamW，也不保留训练计算图；快权重内部更新仍通过解析运算进行。`memory_mode=carry` 在同一 episode 内保留 Memory；`reset` 模式是每次预测重置的消融。环境重新开始一个 episode 时应重置对应 Memory，避免不同 seed/episode 共享历史。

reset 恢复的是可学习初始快权重，并将相应位置计数归零，不是把所有快权重设成零。`reset` 消融仍保留当前 chunk 内的 token 前缀适应，因此不等于关闭 TTT。一次常规预测包含 4 次去噪前向和 1 次最终写入前向，Memory 位置只推进一次；rollout 在候选状态通过检查后才提交它。

| 协议 | 训练 | 推理 |
| --- | --- | --- |
| 写入来源 | 带噪标签的监督特征 | 最终预测动作特征 |
| TTT 有效范围 | 33 tokens | 33 tokens |
| 有效动作维度 | 12 | 12 |
| 更新间隔 | Anchor stride16 | 每执行 H8 后再次预测 |
| 记忆边界 | 新窗口重置，TBPTT4 截断梯度 | episode 重置；窗口内概念不适用 |

两侧有效范围已对齐，但写入来源、经历的历史长度和更新间隔仍不同。README 将其记录为现有实现的协议差异，不把它表述成完全一致。

## 9. 按职责查找源码

主线之外的文件主要负责模型装配、分布式运行和评测。下面把它们与当前未启用的数据分支分开，便于按问题定位。

| 类别        | 文件                                                                                                                                                                                                                                                        | 职责                                     |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 模型装配      | [模型](rlinf/models/embodiment/gr00t/gr00t_n1d7/__init__.py) `__init__.py`                                                                                                                            | `sequence_training` 分派                 |
| 模型装配      | [gr00t_action_model.py](rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py)                                                                                                             | 通用 GR00T RL 适配；当前构建复用 backbone 本地路径重定向 |
| 备用数据分支    | [window_dataset.py](rlinf/data/datasets/gr00t_n1d7/window_dataset.py)、[window_sampling.py](rlinf/data/datasets/gr00t_n1d7/window_sampling.py) | `independent_h40`；当前 Sequence32 配置不启用  |
| 分布式基类     | [fsdp_vla_sft_worker.py](rlinf/workers/sft/fsdp_vla_sft_worker.py)、[fsdp_sft_worker.py](rlinf/workers/sft/fsdp_sft_worker.py)                 | VLA/SFT 通用执行基础                         |
| 分布式执行     | [fsdp_model_manager.py](rlinf/hybrid_engines/fsdp/fsdp_model_manager.py)、[strategy/fsdp.py](rlinf/hybrid_engines/fsdp/strategy/fsdp.py)       | 模型、优化器、分片和保存恢复                         |
| 精度        | [mixed_precision.py](rlinf/hybrid_engines/fsdp/mixed_precision.py)、[fsdp_config.py](rlinf/utils/fsdp_config.py)                               | FSDP 包装与分模块精度                          |
| 评测配置      | [bowlandcup_gr00t_native_ttt.yaml](evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml)                                                                                                        | 八卡、seed7–14、H8 执行、Memory 模式            |
| 评测 worker | [gr00t_native_worker.py](evaluations/gr00t_native_worker.py)                                                                                                                                        | 场景/策略 seed、动作生成与终止记录                   |
| 评测循环      | [embodied_eval_runner.py](rlinf/runners/embodied_eval_runner.py)                                                                                                                                    | RLinf 通用评测循环                           |
| 环境        | [robocasa365_env.py](rlinf/envs/sim/robocasa365/robocasa365_env.py)、[robocasa365.yaml](examples/embodiment/config/env/robocasa365.yaml)       | 仿真 reset/step/观测接口与环境默认值               |
| 诊断        | [check_gr00t_native_memory.py](evaluations/check_gr00t_native_memory.py)                                                                                                                            | Memory 更新、传递和重置检查                      |
| 性能采集      | [run_robottt3_profile.sh](examples/sft/run_robottt3_profile.sh)                                                                                                                                     | 独立采集入口，默认预览                            |
| 短程验收      | [run_robottt3_acceptance.py](examples/sft/run_robottt3_acceptance.py)                                                                                                                               | 短程训练与保存恢复验收                            |

GR00T 的实际 Processor、DiT 和视觉语言 backbone 位于外部依赖，理解动作如何归一化、补齐和解码时也需要阅读它们。配置固定依赖 commit `23ace64f17aa5015259b8609d371eb61a357c776`。

| 外部文件                                                                                                                             | 职责                                 |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| [gr00t_n1d7.py](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)                       | 原生模型、动作头、`sample_time()`、动作编码/解码   |
| [processing_gr00t_n1d7.py](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py) | 图像与状态处理、动作归一化、Padding、mask         |
| [dit.py](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/dit.py)                                        | Self-Attention、Cross-Attention、FFN |
| [qwen3_backbone.py](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/modules/qwen3_backbone.py)                  | 视觉语言特征                             |
| [lerobot_episode_loader.py](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/data/dataset/lerobot_episode_loader.py)   | Episode、Parquet 与视频读取              |

## 10. 运行与复现时检查什么

启动训练需要该部署的 Python 环境、RLinf 和固定版本 Isaac-GR00T 均可导入，并确保 `PYTHONPATH` 优先加载本工作区。模型与数据路径取自训练 YAML。先确定新训还是续训、输出目录与学习率计划，再指定 `--config-name` 为上述两个名称之一；名称后不带 `.yaml`。

当前 FSDP 使用 FP32 主参数、通常为 BF16 的模型计算参数、FP32 梯度规约，并对 `FastWeightTTT` 子树显式保留 FP32；完整模型权重保存开启。具体保存/恢复还依赖 optimizer、scheduler、DataLoader 游标、RNG 与协议身份，单独加载权重不等于严格续训。以 worker 的 `save_checkpoint()` / `load_checkpoint()` 为准。

评测入口要求显式指定 checkpoint、输出、Isaac-GR00T、RoboCasa 和 robosuite 路径；正式执行使用 `--execute`，计划预览使用 `--dry-run`。当前评测按八个 seed、每个三次重复组织，共 24 条 rollout，每条最多 2400 环境动作。训练只读示范，仿真评测还需要能创建 PandaOmron 的 RoboCasa/robosuite 环境。

训练记录可从 `sequence_samples_rank*.jsonl` 查看实际 Anchor 采样，从 `sequence_metrics.jsonl` 与 TensorBoard 查看 loss、学习率、梯度范数和 `ttt_update_rms`。`ttt_update_rms` 表示快权重变化的监控量，不是记忆准确率。判断隐式记忆是否有效，需要同场景比较 carry/reset 等条件，并结合任务决策与成功率；仅有 loss 下降或 FastState 非零都不够。
