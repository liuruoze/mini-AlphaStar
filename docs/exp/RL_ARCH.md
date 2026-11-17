# RL 训练架构（代码现状，2026-11-16）

范围：`run_rl.py` → `alphastarmini/core/rl/rl_vs_inner_bot_mp.py` 这条链。
`rl_train_with_replay`（`ActorLoopPlusZ`，带回放 z 奖励）在 `run.py:76` 已注释，不在本链条内。

## 1. 总览

单模型、多进程 A3C（actor-learner）。15 个 Worker 进程，每个进程 2 个 actor 线程、1 个 Learner
线程；所有进程共享同一份 `state_dict` 和同一个共享 Adam；对手是难度 1 的内置 bot；默认不读回放、
不算 z 奖励。

| 项 | 值 | 位置 |
|---|---|---|
| Worker 进程 | 15 | `PARALLEL = 8 + 7 * 1`（可用 `MAS_PARALLEL` 覆盖） |
| 每进程 actor | 2（可用 `MAS_ACTOR_NUMS` 覆盖） | `rl_vs_inner_bot_mp.py:89` |
| actor 总数 | 30 | 2 × 15 |
| Learner | 每 Worker 1 个线程，各自 `optimizer.step()` | `Worker()` |
| 决策频率 | `STEP_MUL = 8`，固定 | `:153` |
| 学习率 | `1e-6`（`LR` 变量，不是 `THP.learning_rate` 的 1e-5） | `:110` |
| 批 | 8 条轨迹 × 8 步 | `AHP.batch_size=8`、`sequence_length=8` |
| 实体 / 可选单位 | 128 / 12 | `AHP.max_entities`、`max_selected` |
| 难度 | 1 | `:101` |
| 回放 / z 奖励 | 关 | `NO_REPLAY_LEARN=True`，`:173` |
| 对手观测 | 不用 | `USE_OPPONENT_STATE=False`，`:172` |

## 2. 进程与线程

| 角色 | 数量 | 职责 |
|---|---|---|
| 主进程 `test()` | 1 | 建模型、`share_memory()`、共享 Adam、队列、可选批推理 server、spawn Worker、回收 |
| `Worker(rank)` | 15 | 起本进程的 Learner 线程与 2 个 actor 线程；每个 actor 自带一个 SC2 环境 |
| `Learner` | 15 | 收轨迹、算损失、`backward`、把梯度写回共享参数、`shared_adam.step()` |
| `Parameter_Server` | 1 | **只做统计与存档**：收 `(native, custom)` 与点数，写 tensorboard、按块存 ckpt |
| `AsyncBatchInferenceServer` | 0/1 | 可选（`MAS_BATCH_ACTING=1`）：30 个 actor 共用一个 forward，LSTM 由 server 维护 |

数据流：

```
SC2 env ──obs──> preprocess_state_all ──> forward（本地 / 批推理）
   │                                            │
   └──<── action_to_func_call <── 6 个头 ───────┘
                │
        env.step(step_mul=8)
                │
        reward / outcome ──> Trajectory ──每 8 步成批──> Learner
                                                          │
                                    loss ── backward ──> 共享梯度 ──> shared Adam
                                                          │
                                        Parameter_Server 只读参数存盘/记曲线
```

## 3. 单步（actor）

1. `home_obs` → `preprocess_state_all`，产出实体、标量、地图三段。
2. `ActorInferenceClient.step`：local 模式直接 `agent.step_from_state`；batch 模式把状态写进
   server 的槽位，取回动作与 logits。
3. `action_to_func_call` 转成 pysc2 的 function call，`env.step(..., step_mul=8)`。
4. 奖励：胜 `+1`、负 `-1`；平局按 `killed_points > 4000` 记 `+1`，否则按比例
   （`WIN_THRESHOLD`，`:171`）。`native` 与 `custom` 两套结果同时入队。
5. 组 `Trajectory`（20 个字段，含 `state`、`memory`、`action`、`behavior_logits`、
   `teacher_logits`、`masks`、`build_order`、`game_loop` 等）。
6. 攒满 `sequence_length=8` 步，`stack_namedtuple` 后交给同进程的 Learner。

## 4. 一次更新（Learner）

1. 每 10 s 从 `global_model` 把权重拷到本进程的本地模型（`load_state_dict`）。
2. 从队列取 8 条轨迹（FIFO，`buffer_size=2`，随机采样默认关）。
3. `agent.rl_unroll` 做 teacher-forcing 的整段 forward，得到 6 个动作头的 logits 和 5 个
   baseline（winloss / build_order / built_units / upgrades / effects）。
4. 组装损失（§5），`backward()`。
5. `ensure_shared_grads` 把本地梯度写到共享参数上，`shared_adam.step()` 更新共享权重；
   `policy_version += 1`。
6. 其它 Worker 的 actor 在下一次 10 s 同步时看到新权重。

因为是 15 个 Learner 同时对同一份参数做 `step`，这是 **Hogwild** 式异步更新，没有参数聚合或
梯度平均。

## 5. 损失构成

| 项 | 权重 | 说明 |
|---|---|---|
| V-trace（winloss） | policy 10.0 / baseline 5.0 | 主 actor-critic（`WINLOSS_BASELINE_COSTS`） |
| V-trace（build_order） | 4.0 / 1.0 | 伪奖励，需 z |
| V-trace（built_units） | 6.0 / 1.0 | 伪奖励，需 z |
| V-trace（upgrades） | 6.0 / 1.0 | 伪奖励，需 z |
| V-trace（effects） | 6.0 / 1.0 | 伪奖励，需 z |
| UPGO（winloss） | 1.0 | 只在 winloss baseline 上 |
| KL 蒸馏 |全参数 2e-3 / action_type 1e-1 | 对 SL teacher 的 logits |
| 熵 | 1e-4 | 取负号做梯度下降 |

`NO_REPLAY_LEARN=True` 时，除 winloss 外的四路 V-trace 直接 `break` 掉，实际只剩
**winloss V-trace + UPGO + KL + 熵**。

## 6. 与 AlphaStar 的差异

| 维度 | AlphaStar | 本实现 |
|---|---|---|
| 对手 | league（main / exploiter / PFSP） | 内置 bot，固定难度 1；`League` 只用 1 个 main、0 exploiter |
| 伪奖励 | z 来自人类回放统计 | 代码在，默认关闭 |
| 决策频率 | 由 delay 头决定 | 固定 8，delay 头不驱动 `env.step` |
| 参数同步 | 单一 learner，批量 | 15 个 Learner Hogwild |
| 梯度裁剪 | 有 | 无（`THP.clip=0.5` 未被使用） |
| 学习率 | 3e-5 | 1e-6 |
| 观测 | 含对手状态 | `baseline_state_op` 恒 `None` |

## 7. 现存风险（2026-11-17 按修复结果标注）

1. **Hogwild 竞争**：15 个进程并发更新同一份参数，梯度基于不同时刻的权重，陈旧度可达数十步。
   （未动，计划见 `PLAN_RL_20261117.md` 阶段 3）
2. **梯度裁剪默认关闭**：损失里 V-trace、UPGO、KL 三项量级差很大（KL 2e-3 vs UPGO 1.0），
   一旦单批出现大优势，`update_norm` 会跳。D14 只加了 `MAS_RL_CLIP` 开关，默认仍为关。
3. **样本相关性强**：`buffer_size=2` 且 FIFO，等于只用最近 16 步的两批轨迹，没有打散。
   D15 只加了 `MAS_RL_USE_BUFFER` 开关，默认仍走原分支；且开启后更新门槛从 16 条变 80 条，
   不是纯采样 A/B。
4. **策略滞后**：local 模式下 actor 每 10 s 才同步一次，轨迹里的 `behavior_logits` 来自旧策略，
   与当前策略的比值（rho）天然偏离 1。（未动）
5. **KL 语义**：已修（D5），默认改为 `renorm`，`MAS_KL_MODE=legacy` 可回退。
6. **平局改判**：`killed_points > 4000` 的平局记胜（custom 口径），与 `native` 分列，
   报告时不能混用。（口径未变，属报告约定）
7. **Parameter_Server 先退不影响训练**，但胜率曲线会丢；它靠 reaper 投哨兵收尾。（未动）
8. **soak#6 未定位**：step=200 时全进程同停、零崩溃栈。这是当前唯一会阻断长跑的问题，优先于
   一切算法实验。

## 8. 优化建议（按性价比排序）

| 优先级 | 动作 | 预期 | 代价 |
|---|---|---|---|
| P0 | 固定评测口径：D1、20 种子、40k 步、native 与 custom 分报 | 现在 0/12，先要能判断"有没有学" | 低 |
| P0 | buffer 2 → 8，改随机采样 | 降低样本相关性，rho 更稳 | 低 |
| P0 | 加梯度裁剪（全局范数 0.5，用现成的 `THP.clip`） | 抑制偶发大步 | 低 |
| P1 | 把 15 个 Learner 改成 1 个集中 Learner + 30 个纯 actor | 去掉 Hogwild，更新可复现 | 中（要改 Worker 结构） |
| P1 | `MAS_KL_MODE=renorm` 设为默认，并记录每个头的 KL | 蒸馏项可解释、不为负 | 低 |
| P1 | 学习率与熵系数做小网格（1e-7 / 1e-6 / 3e-6；1e-4 / 1e-3） | 找到能出胜局的区间 | 中 |
| P2 | 打通 z 奖励路径（回放进程 + `ActorLoopPlusZ`） | 追平 AlphaStar 的伪奖励设计 | 高（需 replay 进程与对齐验证） |
| P2 | 难度课程：D1 → D2 → D3 | 对抗强度递增 | 中 |
| P2 | 启用 `AsyncBatchInferenceServer` 做统一批推理 | 30 actor 共用 forward，GPU 利用率提高 | 中（已有开关，需压测） |

## 9. 建议的实验顺序

1. **基线可复现**：`MAS_PARALLEL=4 MAS_ACTOR_NUMS=2 MAS_MAX_TIME=1800 MAS_STATIC_NUM=10`，
   跑 3 次，看 win_rate 是否出现点、loss 是否分项、`update_norm` 是否稳定。
   判据：3 次都有 ≥3 个胜率点，无 NaN，`optimizer_steps > 0`。
2. **单 Learner 改造**：同预算对比吞吐与 `update_norm`。判据：吞吐不降 20% 以上，更新次数可复现。
3. **采样与裁剪**：buffer 2→8、随机采样、梯度裁剪，2×2 对照。判据：rho 分布更集中、
   胜率不降。
4. **学习率 / 熵网格**：先 6 组，每组 30 min。判据：出现 ≥1 胜局（native）。
5. **z 奖励**：先只验证回放进程能起、z 统计非零、`build_order_baseline` 的 loss 有值，
   再谈收益。
