# mini-AlphaStar

[English](HISTORY.MD) | **中文**

本文件是 [`HISTORY.MD`](HISTORY.MD) 的中文镜像，版本顺序与条目与英文版一一对应。

## v_1.08

* 改进监督学习的训练，使其 `batch_size` 与 `sequence_length` 与强化学习中的不同，从而提升学习效果；
* 强化学习现在使用 `use_action_type_mask`，提升学习效率；
* 新增可关闭 `entity_encoder` 与 `autoregressive_embedding` 的开关，用于测试这两个模块的影响；
* 修掉一些已知问题；
* 提供一个已训练的 RL 模型，它先用 SL 方法训练，再用 RL 方法训练；

## v_1.07

* 在 `rl_unroll` 里用 `mimic_forward` 替代 `forward`，提升训练准确率；
* RL 训练支持多 GPU；
* RL 训练支持基于多 GPU 的多进程训练；
* 为 RL 损失采用新架构，GPU 显存降低 86%；
* 为 RL 采用新架构，采样速度提升 6 倍；
* 再次验证 UPGO 与 V-trace 损失；
* 用"多进程 + 多线程"训练，采样速度再提升 197%；
* 修掉 GPU 显存泄漏，并减少 CPU 内存泄漏；
* 对 level-2 的 RL 训练胜率（不计单位损失）提升到 **0.57**！

## v_1.06

* 首次成功让农民在正确位置建造水晶塔；
* 提升 SL 的选择准确率，以及 RL 的初始状态；
* 提升对内置 AI 的胜率；
* 对内置 AI 的胜率提升到 **0.8**，击杀分到 **5900**；
* 加入结果视频；
* 提升 baseline（RL 中状态价值估计的别名）的准确率与速度；
* 提升 RL 训练结果的可复现性（使用随机种子与单线程）；
* 大幅重构 RL 损失的代码写法，加入实体掩码；
* 用 `outlier_remove` 函数修掉 RL 损失计算里的大 loss 问题；
* `rl_loss` 的代码行数减少 58.2%；
* 改进 `log_prob`、KL、熵的编码；
* `rl_algo` 的代码行数减少 31.4%；

## v_1.05

* 重构大部分代码，从回放转换一直到强化学习；
* 通过新架构，转换后的回放 tensor 文件节省 70% 磁盘空间；
* 内存占用也做了优化，现在 SL 能处理的回放数量是此前的 3 倍以上；
* 借助新的 tensor 回放文件，SL 训练速度大幅提升，约为此前的 10 倍；
* 使用加权损失后 SL 结果显著提升，动作准确率升到 90% 以上；
* 通过新的 SL 训练方式，单位等动作参数的准确率从 0.05 提升到 0.95；
* 重构 RL 的计算损失，可读性提升；
* 得益于算法优化，RL 的采样速度达到此前的 10 倍。

## v_1.04

* 为监督学习的损失计算加入掩码；
* 评测中加入"含镜头 / 不含镜头"两类准确率；
* 为多 GPU 训练重构大部分代码；
* 支持多 GPU 监督训练；

## v_1.03

* 在 `USAGE.MD` 中加入"怎么跑 RL"的指南；
* 修掉若干 bug，以及 SL 训练中的一个 bug；
* 新增 `USAGE.MD`；
* 通过拆分 `map_channels` 与 `scatter_channels` 修掉一个 bug；
* 修正查看 AlphaStar Final Terran 回放所需的版本；
* 在 `selected_units_head` 中把 `z_2 = F.relu(self.fc_2(z_1))` 改为
  `z_2 = self.fc_2(F.relu(z_1))`；

## v_1.02

* 在空间编码器的每个下采样 `conv2d` 之后加入 relu；
* 把大多数 conv 与 convtranspose 的 bias 从 False 改为 True（若其后没有 bn，或位于第三方库里）；
* 在 `entity_encoder` 中加入"缺失条目掩码"；
* 把函数 `unit_tpye_to_unit_type_index` 整理为可供调用的库函数；
* 用 `unit_tpye_to_unit_type_index()` 修掉 `calculate_unit_counts_bow()` 里的一个 TODO；
* 改回使用 `All_Units_Size` 等于所有 `unit_type_index`；
* 在 `spatial_encoder` 中加入 scatter 实体图；

## v_1.01

* 在 `winloss_baseline` 中改用胜负奖励（这才是 AlphaStar 应该做的）；
* 细化伪奖励、缩放 Leven 奖励、修正 log prob（它应当是负的交叉熵）；
* 修掉因 AlphaStar 原始代码错误导致的若干问题；
* 加入 AlphaStar 回放中镜头移动次数的分析；

## v_1.00

* 对 SL 训练做了一些改进；
* 修掉 SL 训练中的一些 bug；
* 修掉因 GPU 的 cudnn 导致的 RL 训练 bug，并重整目录；
* 为单位数量奖励（汉明距离）加入时间衰减系数；

## v_0.99

* 加入分析回放统计的代码；
* 实现带 z 奖励的 RL 训练（z 来自回放中专家的统计）；
* 补齐 `entropy_loss_for_all_arguments`；
* 真正实现 `human_policy_kl_loss` 里的 `teacher_logits`，并在服务器上测试上述全部 RL 训练；
* 把 `sl_loss` 与 agent 解耦，并通过服务器上的 SL 训练测试；
* 修掉 PyTorch 1.5 导致的若干警告，并补齐大量 TODO；

## v_0.96

* 修掉 v-trace 策略梯度损失里 `clipped_rhos` 的计算问题；
* 修掉 `vtrace_pg_loss` 中另外 5 个参数的 `filter_by` 函数；
* 为全部 6 个动作参数实现 `vtrace_advantages` 方法；
* 完成全部参数的 `split_vtrace_pg_loss` 实现；
* 补齐 build_order、built_units、upgrades、effects 这些 baseline 的正确流程；
* 修掉 UPGO 中另外 5 个参数的损失计算；

## v_0.93

* 实现 `unit_counts_bow`；
* 实现 `build_order`；
* 实现 upgrade、effect、last 与 available action，以及时间编码；
* 加入 build order 与单位数量的奖励计算；
* 加入基于 Levenshtein 与汉明距离奖励的 `td_lambda_loss` 计算；

## v_0.9

* 修掉 SL 的评测问题，支持用 SL 模型微调 RL 训练；
* 修掉 SL 训练过慢的问题；
* 修掉 RL 训练问题，并准备好对内置 AI 的测试代码；

## v_0.8

* 加入 Levenshtein 与汉明距离的代码；
* 加入与内置 AI 电脑对战的函数；
* 修掉 SL 训练中选择单位的损失问题；

## v_0.7

* 我们发布了 mini-AlphaStar 项目（v_0.7），它是 DeepMind 原始 AlphaStar 程序的微型源码版本。
* "v_0.7" 表示我们认为已经实现了它 70% 以上的代码。
* "mini" 表示我们让原始 AlphaStar 的超参数可调，以便在小规模上运行。
