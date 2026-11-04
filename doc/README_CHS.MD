＃mini-AlphaStar

## 介绍

我们发布了mini-AlphaStar项目（v_0.X版），它是DeepMind原始AlphaStar程序的微型复现版本。

“v_0.X”表示我们认为已经实现了X * 10％以上的代码。

“mini”意味着我们使原始的AlphaStar超参数可调且微小化，以便可以小规模地运行。

在设计mAS时，我们参考了“奥卡姆剃刀原则”，简单就是好的。

因此我们从零开始构建mAS，除非对速度和性能影响很大的功能，能够不加就不加。

同时也尽可能不要使用过多的依赖包，让mAS最好只依赖于PyTorch（最多指定最低的版本）。

通过这样的方式，我们大大简化了mAS的学习成本，让mAS的架构和代码读起来相对容易。

另外，在代码中我们还添加了很多的注释，确保一些关键的设计能够传达给读者。

## 2026-11 改进（评估分支说明）

> 本节对应英文版 `README.md` 顶部的 branch note，是 2026-11-02 → 11-04 这四天的工作记录。
> 详细版见 [`doc/IMPROVEMENTS_2026_11.md`](IMPROVEMENTS_2026_11.md)。

本分支是**评估分支**，包含两次评估尝试：(a) 早前的 *batch acting*（中心批推理）实验；
(b) 2026-11 的 SL / RL 改进。两次的结论都是**有好有坏**，下面按实测写，不按期望写。

### 一句话结果

监督学习（SL）从"胜率 0%、每局采集 2,676 资源"，提到
**"胜率 4.3%、每局采集 9,010（官方 2021 年模型的 95%）"**，
但**没有追上官方 SL 的 17.5%**；
强化学习（RL）主循环从"从未被执行过"做到"25 分钟无故障、统计链完整、能自行退出"，
但**胜率没有上升**。

| 阶段 | 模型 | 局数 | 胜率（Wilson 95%） | 采集 | 击杀 | 军队 |
|---|---|---|---|---|---|---|
| 起点 | 旧 SL（静默只用 1/60 数据） | 20 | 0.0% | 2,676 | 145 | — |
| **本分支 SL** | `SOUP_prior3` | **230** | **4.3% [2.4, 7.8]** | **9,010** | 1,439 | **21.2** |
| 参照 | 官方 SL `sl_21-12-29` | 80 | **17.5% [10.7, 27.3]** | 9,494 | 1,811 | 14.0 |
| 参照 | 官方 RL `rl_22-02-09` | 28 | 21.4% | 9,533 | 1,787 | — |

⚠️ **参照数字本身也被修正过一次**：官方 SL 的胜率长期被引用为 **7.1%**，
但用同一套评测跑 80 局得到 **17.5%（[10.7, 27.3]）**——7.1% 只是 28 局的点估计。
因此此前所有"我们对上了 7.1%"的说法**全部作废**（见勘误表）。

### SL 阶段修掉的三个结构性缺陷

1. **数据被静默降级**：`SIMPLE_TEST = not P.on_server` 加上 `replay_files[:2]`，
   意味着忘记设一个环境变量，2.5 小时的训练就只用了 **1/60 的 replay**（`train_set = 120`）。
   程序不报错、loss 照常下降。改为冒烟必须显式 `MAS_SL_SMOKE=1`，并在首屏打印
   `[SL-config] files=… smoke=…`。
2. **动作分布被采样方式扭曲 11-19 倍**：转换脚本按固定帧步长（约每 20 帧取一帧），
   而智能体每 8 帧才下一条命令，于是大多数采样帧的标签是 `no_op`。
   训练数据是 **81.0% `no_op`、2.7% 造水晶塔、1.1% 造农民**，
   而官方模型实战是 **0% `no_op`、29.8% 造水晶塔、21.1% 造农民**。
   模型是数据的忠实镜像：实战中 99.9% 的时间在发呆。
   修法是**动作配重**（对 `action_type` 头做重要性加权
   `w[a] = clip((目标占比/数据占比)^1.0, 0.1, 50)`），实战空转从 **37% 降到 6.7%**。
3. **用验证集准确率选模型是反的**：第 3 轮 val 0.44 但只发呆 50%，第 10 轮 val 0.83 却发呆 96.8%。
   **模型选择必须看行为指标**（击杀 / 空转比例 / 胜率），不能看动作分类准确率。

### 一个零成本技巧：多版本平均

把同一次训练相邻若干轮的权重按位取平均（`experiments/_avg_models.py`，
原名 model soup），击杀上限从 2,550 提到 **6,900**，胜率从 1.9% 提到 5.1%（n=158）。
平均**更多**版本（5 个）反而更差——激进的那一轮被稀释了。

### RL 阶段：那个"看起来像死锁"的 Bug

RL 主循环（A3C）在这次之前**从未被执行过**。让它跑起来需要修 8 处，其中决定性的一处是：

```python
outcome = q_winloss.get(timeout=600)   # 队列空 -> queue.Empty -> 落进 except -> 进程退出
```

而一局要 **800-1300 秒**。于是**唯一的结果消费者在 10 分钟时把自己判死了**：
之后没有进程取结果，`q_winloss` 一路积压（6、7、8…），
`win_rate` / `mean_points` / **checkpoint 保存**全部断裂。
表面表现与死锁**完全一致**——这正是前两轮往锁和共享内存方向查、全部查错的原因。

修法：`Empty` 只表示"这一局还没打完"，父进程还活着就继续等；
真正的退出条件只有两个：收尾线程投递的结束哨兵、父进程消失。

修好后 **soak#5** 第一次跑出完整闭环：

```
1503.9 秒（25 分钟）  4/4 进程全程存活  12 局完成
[agent_4] steps=5720 | 吞吐 3.80 step/s
Parameter_Server end:
== done ==                              <- 到达墙钟后自行退出

loss/learner_0                      n=204  last=-0.0028
learner_0/loss_upgo_                n=204  -0.0029 ~ 0.1799
all_kl_loss_action_type_            n=204   0 ~ 0.0009
winrate/every_2_episodes            n=3     0.0 / 0.0 / 0.0
meanpoints/every_2_episodes         n=3     0.0344 -> 0.1781
```

### 尚未解决

- **30-60 分钟未证明**：最好一次 25 分钟。随后一次 50 分钟的复验（soak#6）
  在 11 分钟时**全局卡死**——6 个 actor 的进度同时停在同一步、心跳停止，
  但 5 个进程全都活着、`faulthandler` 零崩溃栈。这是与上述 Bug **不同**的故障，**未定位**。
- **胜率没有上升**：12 局全不赢；以 4.3% 的起点算，12 局全不赢的概率约 59%，
  所以这既不算失败证据、也不算学习证据。
- 下一步的第一件事是装 `py-spy`，在卡死时 dump 所有进程的栈（本轮缺这个工具，
  是 soak#6 没能定性的直接原因）。

### 怎么做的：人类统筹 + 多个智能体分工

这次改进**不是**单个模型自主完成的，而是**一位人类监督者指挥多个大模型智能体**：

| 角色 | 是谁 | 实际做了什么 |
|---|---|---|
| 监督 / 仓库所有者 | **刘若泽（`liuruoze`）** | 每轮定目标、提供官方 2021/2022 模型与缺失的 30 个 replay、纠正错误方向（例如在智能体猜测时指出官方模型的发布位置）、强制"反思→计划→记录"的纪律、否决没有实测支撑的结论 |
| 执行（本次改进） | **DeepSeek-V4.1-Flash** | 端到端执行实验：静态审计、训练与评测作业、埋点、以及编号 **#23-#58** 的记录 |
| 代码 review（更早的轮次） | **GLM**（#1-#22）、**Kimi 3.1 Pro**、**MiniMax H3.1**、**space-bunny** | 审 diff、质疑结论，多次发现执行者已经写进"结论"的错误（见勘误表：78 条，其中 5 条由执行者自己撤回） |

两条做法起了决定性作用，都来自监督者：**(a)** 每条结论必须带可证伪的判据和实测数字；
**(b)** 每条被撤回的结论写进勘误表，而不是悄悄删掉。
本分支里大多数"发现"之前，都有一个被勘误表记录下来的错误猜测。

工作笔记（30+ 份运行日志、计划、review 与勘误表）**有意不放进本仓库**（含机器相关的路径）。
上面所有数字都是**一台机器上的一次实测快照**（4×RTX 4090、128 核），不是可复现的基准。

### 复现命令

```bash
# 1) 训练：20 个 replay 文件 × 2 轮，开启动作配重，共 30 个 replay-epoch
MAS_ON_SERVER=1 MAS_SL_MAX_FILES=20 MAS_SL_WORKERS=0 MAS_SL_EPOCHS=2 \
MAS_SL_ACTION_REWEIGHT=1 MAS_SL_PRIOR_EXP=1.0 \
CUDA_VISIBLE_DEVICES=0 python run_sl.py --path1 ./data/replay_data_v3/

# 2) 把同一次训练的第 1、2 轮 + 另一次训练的第 3 轮 平均成一个模型（零额外训练）
python experiments/_avg_models.py \
  --a model/<run>_ep01.pth --b model/<run>_ep02.pth --more model/<other-run>_ep03.pth \
  --out model/BEST_SL.pth

# 3) 评测：4 分片 × N 局，只用 SC2 原生胜负口径
MAS_ON_SERVER=1 python experiments/_eval_winrate.py \
  --model model/BEST_SL.pth --games-per-shard 15 --shards 4 --gpus 0,1,2,3

# 4) RL 主循环：一小时、8 个 actor、显式指定 SL 初始化、心跳 60 秒
MAS_ON_SERVER=1 MAS_RL_INIT_MODEL=model/BEST_SL.pth \
MAS_MAX_TIME=3000 MAS_ACTOR_NUMS=2 MAS_PARALLEL=4 \
MAS_STATIC_NUM=4 MAS_HEARTBEAT_SEC=60 MAS_QUEUE_TIMEOUT=3600 \
MAS_GPU_LIST=1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run_rl.py
```

⚠️ `MAS_RL_INIT_MODEL` **不是可选项**：不设它时，加载器会取 `model/` 里
字典序最大的 `sl_*.pth`，本次工作中那个恰好是一个**故意训坏的** checkpoint。

### 术语对照（白话版）

| 论文术语 | 本仓库叫法 | 一句话解释 |
|---|---|---|
| action prior / prior reweight | **动作配重** | 数据里少的动作多算几遍、多的少算几遍，把比例掰回目标分布 |
| model soup / weight averaging | **多版本平均** | 同一次训练的第 1/2/3 轮参数按位取平均，零训练成本 |
| replay-epoch | **训练量（文件数 × 轮数）** | 15 文件 × 2 轮 = 30；单看"轮数"会被文件数骗 |
| no_op | **空转 / 发呆** | 模型输出"什么都不做"的比例 |
| native_outcome | **胜负（0 平 / 1 胜 / -1 负）** | "平" = 打到 21,000 帧上限，**不是势均力敌** |

## 目录

下表显示了项目中的相应模块。

模块|内容
------------ | -------------
alphastarmini.core.arch | alphaStar体系结构
alphastarmini.core.sl | 监督学习
alphastarmini.core.rl | 强化学习
alphastarmini.core.ma | 多代理联赛训练
alphastarmini.lib | 相关库函数
alphastarmini.third | 第三方函数
res | 其它有用的资料

## 要求

Pytorch >= 1.5，其他请参阅requirements.txt。

## 地点

代码库被放置在以下位置：

位置 | 网址
------------ | -------------
Github | [https://github.com/liuruoze/mini-AlphaStar](https://github.com/liuruoze/mini-AlphaStar)
Gitee | [https://gitee.com/liuruoze/mini-AlphaStar](https://gitee.com/liuruoze/mini-AlphaStar)

## 展望

还有一些部分仍需要实现，例如，z的计算，vtrace的损失部分以及需要填充的env中的某些信息。

## 引用

如果您发现此存储库有用，请引用我们的项目：
```
@misc{mini-AlphaStar,
  author = {Ruo{-}Ze Liu and Wenhai Wang and Yang Yu and Tong Lu},
  title = {mini-AlphaStar},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuruoze/mini-AlphaStar}},
}
```

## 报告

我们将在大约一个月的时间以内提交一份技术报告，以介绍其设计和使用细节。

## 论文

我们将提供一篇可能会在将来可见的论文，介绍使用它的实验和评估。