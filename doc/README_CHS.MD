# mini-AlphaStar

[English](../README.MD) | **中文**

本文件是 [README.MD](../README.MD) 的中文镜像：章节顺序、命令、文件名与数字与英文版一一对应。发现差异按缺陷处理，改动时请两边同时改。

## 介绍

我们发布了 mini-AlphaStar 项目（v_0.X 版），它是 DeepMind 原始 AlphaStar 程序的微型复现版本。

"v_0.X" 表示我们认为已经实现了 X * 10％ 以上的代码。

"mini" 意味着我们使原始的 AlphaStar 超参数可调且微小化，以便可以小规模地运行。

在设计 mAS 时，我们参考了"奥卡姆剃刀原则"，简单就是好的。

因此我们从零开始构建 mAS，除非对速度和性能影响很大的功能，能够不加就不加。

同时也尽可能不要使用过多的依赖包，让 mAS 最好只依赖于 PyTorch（最多指定最低的版本）。

通过这样的方式，我们大大简化了 mAS 的学习成本，让 mAS 的架构和代码读起来相对容易。

另外，在代码中我们还添加了很多的注释，确保一些关键的设计能够传达给读者。

## 图示

这组演示是 mini-AS 在 Simple64 上、用 50 局专家回放做监督学习后的表现，取自 v_1.09 时期，
不是下面「更新」里那个模型。放在这里是为了说明 mini-AS 训出来是什么样。

<p float="left">
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/start.gif" width="360" />
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/middle.gif" width="360" />
</p>

**左：** 开局。    **右：** 对局中期。

<p float="left">
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/attack1.gif" width="360" />
  <img src="https://github.com/liuruoze/Useful-Big-Resources/blob/main/gifs/attack2.gif" width="360" />
</p>

**左：** 智能体的第 1 次进攻。    **右：** 第 2 次进攻。

## 更新

这一段是这次推送的改动说明，范围限于监督学习（SL）：成绩到哪、模型怎么训出来、相对 v_1.09
变了哪一处。强化学习没动，也还没有可用模型，不在其中。

改动只落在训练数据上：在机器数据里再掺 5 局人类录像。模型结构、超参、评测口径都没动，成绩从
31.2% 涨到 45.6%，多了 14.4 个百分点。

读法建议：先看下面的总表，再看方法（DAgger 与规则专家），最后看训练设置和复现命令。

完整记录见 [IMPROVEMENTS_2026_11.md](IMPROVEMENTS_2026_11.md)，结论与门禁见
[CONCLUSIONS.md](../docs/exp/CONCLUSIONS.md)，命名规则见 [NAMING.md](../docs/exp/NAMING.md)。

先约定两个词，否则表读不懂。真胜指 `raw_reward == +1`，打满步数上限的截断局不算胜，
这是我们平时看的那个数。宽胜指 `outcome == 1`，或 `outcome == 0` 且
`killed_points > 4000`，用来和 2021 年的文档、以及更早的记录对齐。另外，只有条件相同的
行才能相互比较——难度、步数、种子数不一样，数字就不能相减。

先看结果。下表把五个对象排在一起：随 v_1.09 一起发布的那份权重、用新代码重训的 B0、两个配方，
以及这一轮最好的那个权重。每一行都标了它自己的条件。

表里还会用到两个概念。配方是一套固定的数据、超参和训练流程；同一配方换个种子再跑一次，叫一个运行。
对局就是那个条件下的评测局数。

对象 | 训练数据 | 条件 | 对局 | 真胜 | 宽胜 | 说明
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
v_1.09 发布权重 | 90 局人类录像（2021） | D1 / 40k | 20 | 15% | 40% | 左上 7/8 = 88%，对上其文档的 0.85
同上 | 同上 | D2 / 40k | 10 | 0% | 20% | 两次宽胜都停在步数上限
同上 | 同上 | D3 / 40k | 20 | 0% | 0% | 9 分钟 55–61 人口，被第一波打穿
Base（B0） | 同源 90 局，用 2026 年代码重训 | 30k / 仅 3038 种子 | 4 | 0% | 25% | 只跑过 3038 单种子的 4 局（不同快照），不是配方成绩
ExpertGuided（专家引导） | 30 局自战 + 30 局规则专家（DAgger） | D3 / 40k | 80 | 31.2% | 62.5%（40 局，该运行真胜 42.5%） | 4 个运行合并；宽胜只能按 `cwB_ep4` 单运行算
HumanHybrid（人机混合） | 上述 + 5 局人类录像 | D3 / 40k | 316 | 45.6% | 51.1%（180 局） | 15 个运行合并，Wilson [40.2, 51.1]；宽胜按只留 no_op 的 6 个运行算，那 6 个运行真胜同为 45.6%
HumanHybrid 最强权重 | 同上 | D3 / 40k | 39 | 64.1% | 66.7% | 含选择偏差，只能当资产，不是配方成绩

表里有两处需要说明。第一，宽胜那两格用的是仍有原始输出、可复算的子集，括号里是局数，和真胜的
全量（316 / 80 局）不是同一批对局。

第二，只有难度 3 那几行可以直接比。同一档上，v_1.09 是 0/20，`ExpertGuided` 是 31.2%，
`HumanHybrid` 是 45.6%。

结果是怎么来的？基座用 DAgger 建，再掺进 5 局人类录像，网络按这一段末尾的设置训练。这一段
剩下的部分把这三步拆开讲。

先说 DAgger，因为后面的数据构成都建立在它上面。DAgger 全名 Dataset Aggregation，直译"数据集
聚合"。普通监督学习只见过专家走过的状态，模型自己一跑就会偏离，进到训练时没见过的局面，错了
没人纠正，误差越滚越大。DAgger 的做法是让当前策略先去跑，收集它真正会遇到的状态，再请专家在
这些状态上标动作，把新标好的数据聚合回训练集。重复几轮，训练分布就慢慢贴近推理时的分布。

DAgger 需要一个"专家"。这里的专家不是人，是规则专家 v5（`experiments/_rule_expert_v5.py`）。

它的实现分两块：每帧怎么决策，建造点怎么选。它是纯 Python 写的，只用到 numpy，不碰任何学习
框架，直接读游戏给的原始观测、直接发原始动作，所以能塞进 DAgger 的采集器里跑。

决策这一块按固定优先级走。每帧先用 `summarize()` 把矿、气、人口、人口上限、空闲与否算出来，
再挑一个动作：补人口 → 造首个 Pylon → 补农民 → 造兵营 → 造兵 → 防守 → 进攻 → 采矿。

选址这块不写死坐标。用 SC2 的 `placement_grid` 校验整块建筑 footprint，Protoss 建筑还必须
落在 Pylon 能量场内、同一高度层、已探索、且不压矿；下过单的建造点会进冷却和黑名单，避免每帧
重复发被拒的命令。

还有一处是为了数据集质量。为了让 30 局不是同一盘棋，每局按种子抽一种策略（快攻、双兵营、
重兵营、运营、骚扰、龟缩、后期），各自的农民数、兵营数、进攻阈值都在区间内抖动。

回到这一轮的数据。严格说 DAgger 会一轮轮迭代，每轮都用当前策略重新采样；我们只做了第 1 轮：
30 局 bot 自战产出 21,983 帧，规则专家跑 30 局给出 9,578 帧动作，两批聚合成 31,561 帧。

专家帧里删掉了 `Cancel_Last_quick`，它在专家数据里占 8.5%，留着会让模型学会"取消—重下"的死
循环。训练时必须带 `MAS_DAGGER=1`：不带这个开关，训练器直接拒绝启动，防止把 DAgger 数据当
普通数据训。

### 规则专家的五个版本

为什么值得单列一张表？因为专家的强弱直接决定专家帧的质量，而专家帧又是 DAgger 的上限。从 v1
到 v5，改动集中在两处：决策层怎么选动作、选址层怎么挑建造点。

最后一列按"那一版能测到什么"写：能上场打的是胜率，还不能上场的是建造点去重率，所以 v1、v2 那
两格是 0.086 和 0.018，不是胜率。

版本 | 打法 | 决策方式 | 随机化 | 实盘结果
------------ | ------------- | ------------- | ------------- | -------------
v1 | 最小干预：只在明确缺口上给动作 | 输入一份手工汇总（人口、人口上限、农民、Pylon、兵营、军队、矿），只定义 5 种动作：补 Pylon / 造兵营 / 造追猎 / 造叉子 / 进攻；建造点由采集器的固定模板给 | 无 | 实跑只剩 1–2 种动作，退化成 Pylon 循环；建造点去重率 0.086
v2 | 同 v1，补上动态选址 | 决定动作仍沿用 v1；新增 `LocationPicker`：以 Nexus(19,23) 为锚点、半径 4–12 的环形候选、避开已占 ±2 格、按 tick 轮换 | 无 | 只交付接口与自测，没接采集管线；数据里去重率仍只有 0.018
v3 | 照 PiGBot 的 B2GM 开局表执行，默认 2-Gate Expand、单矿 | 按表逐步：卡人口 → 建造表 → 补农民 → 产兵 → 兜底 Pylon → 进攻；选址用地形高度层 + Pylon 能量场判定 | 无 | 卡在采气那一步，6,000 步零介入；建造失败率从 89.8% 降到 6–7%，但难度 3 仍是 0/5
v4 | 弃掉建造表，改纯 Zealot rush | 每帧都出动作：人口 / 首 Pylon → 农民 → 兵营 → 造兵 → 防守 → 进攻 → 采矿；敌人进 8 格先回防；会取消 learner 排队中的 Probe 腾矿 | 每局按种子抽一种策略：快攻 / 双兵营 / 重兵营 / 运营 / 骚扰 / 龟缩 / 后期，各自的农民数、兵营数、进攻阈值在区间内抖动 | 目标是在 learner 全程不动时也能独立打赢难度 1
v5 | 同 v4 | 同 v4 | 同 v4，只改快攻那一支：农民 15–18、固定 3 兵营、进攻阈值 9–12、不插撤退；Zealot 动作保持窗从 6–9 帧收到 3–6 帧 | 当前默认实现（`MAS_EXPERT_IMPL=v5`），这一轮的专家帧由它采集

这一轮用的就是 v5：它和 v4 的代码几乎一样，只重调了快攻分支的参数。

基座建好之后，第二步是掺人类数据。我们混入 2021-11-21 那批录像里的 5 局，共 4,545 帧，占全部
训练帧的 12.6%。这 4,545 帧里有 4,247 帧是 `no_op`（人没操作），298 帧是动作帧。只留 `no_op`
（4,247 帧，11.86%）也测过，成绩一样，所以真正起作用的是 `no_op` 这部分。

最后是训练设置。它和上一版 `ExpertGuided` 完全一致，唯一变化在数据，所以下面这些参数不用来回
对照。

项 | 取值
------------ | -------------
模型结构 | 沿用 v_1.09，128 个实体
训练器 | `run_sl.py -p1`
epoch | 4
学习率 | 1e-3
γ | 0.2
step_size | 3
batch | 24
序列长度 | 4
训练 / 验证切分 | 按局切，60 个文件训练、5 个验证
类权重 | `train_probe` 0.15、`build_pylon` 8、`build_gateway` 8、`build_nexus` 0.1、`train_zealot` 4
运行规模 | 15 个种子，每个约 31 分钟，成绩合并该配方所有运行的所有对局

为什么能确认是这批帧起的作用？做一个对照：同样是这 5 局录像，只加动作帧没有用，179 局 32.4%，
和 31.2% 的基线分不开。

想自己复现，按下面两步走。第一步，训一个运行：

```bash
DATA=$PWD/data/replay_data_a3nc_h5 MAS_ON_SERVER=1 MAS_DAGGER=1 \
MAS_SL_LR=1e-3 MAS_SL_EPOCHS=4 \
MAS_SL_CLASS_WEIGHTS="train_probe=0.15,build_pylon=8,build_gateway=8,build_nexus=0.1,train_zealot=4" \
  bash experiments/train_bvb_m30.sh <tag> 4 <seed> <gpu>
```

第二步，评测 20 个标准种子：

```bash
OFF_DIFFICULTY=3 OFF_GAME_STEPS=40000 \
  bash experiments/run_p8_eval.sh <tag> $(seq 3038 3057)
```

还有一个坑要提醒。`MAS_RL_INIT_MODEL` 必填：不设它时，加载器会取 `model/` 里字典序最大的
`sl_*.pth`，而这一轮里那个是故意训坏的。

## 版本

本版本是 **v_1.09**。主要变化如下：

* 修掉若干已知缺陷与问题；
* 改用新的 SL 设置，更容易训出可用的 SL 模型；
* 把 `STEP_MUL` 改为 8，以获得更好的 RL 训练结果；
* 在 RL 训练中记录所有损失，便于了解训练过程；
* 对 level-1 内置 bot 的胜率提高到 **0.85**（目前最好）；
* 首次提供预训练 SL 模型，供 RL 训练使用；
* 首次提供最终的 RL 模型，便于他人复现结果。

## 提示

**警告**：SC2 极其困难，AlphaStar 本身也很复杂。虽然本项目是 mini-AlphaStar，
但它几乎使用了与 AS 相同的技术，训练资源开销同样很高，**几乎不可能在笔记本上训练**。
推荐使用带 GPU 卡、内存与磁盘都充足的商用服务器。第一次接触本项目，建议先收藏（star）；
等有足够的空闲时间与训练资源时再深入研究。

## 位置

我们在两个地方存放代码并展示视频。

代码位置 | 结果视频位置 | 用途
------------ | ------------- | -------------
[Github](https://github.com/liuruoze/mini-AlphaStar) | [Youtube](https://youtu.be/mTtA0vdAULw) | 面向全球用户
[Gitee](https://gitee.com/liuruoze/mini-AlphaStar) | [Bilibili](https://www.bilibili.com/video/BV1Hm4y197Jm/) | 面向中国用户

## 目录

下表显示了项目中的相应模块。

模块 | 内容
------------ | -------------
alphastarmini.core.arch | 深度神经网络结构
alphastarmini.core.sl | 监督学习
alphastarmini.core.rl | 强化学习
alphastarmini.core.ma | 多智能体联赛训练
alphastarmini.lib | 库函数
alphastarmini.third | 第三方函数

## 要求

PyTorch >= 1.5，其余见 `requirements.txt`。

## 安装

[SCRIPT 指南](../scripts/Setup_cmd.MD) 给出用 conda 安装 PyTorch 的命令
（会自动安装 CUDA 与 cuDNN，比较省事）。

例如（安装 PyTorch 1.5 及其配套 CUDA、cuDNN）：

```
conda create -n th_1_5 python=3.7 pytorch=1.5 -c pytorch
```

然后激活该 conda 环境：

```
conda activate th_1_5
```

接着用 pip 安装其余 python 包：

```
pip install -r requirements.txt
```

## 用法

环境准备好之后，运行下面这个 python 文件即可：

```
python run.py
```

你可以在 `run.py` 里通过注释 / 取消注释来选择想跑的流程。

[USAGE 指南](USAGE_CHS.MD) 给出了一些常见问题与回答。

要得到与主页 gif 相近或更好的结果，请按下面这条顺序操作：

1. **转换回放**：下载用于训练的回放，再用 mAS 里的脚本把回放转成可训练数据；
2. **监督学习**：用这些可训练数据监督学习出一个初始模型；
3. **评测 SL 模型**：训练出的 SL 模型要在 RL 环境里评测，确认它的行为是对的；
4. **强化学习**：用训练好的 SL 模型在 SC 环境里做强化学习，观察胜率开始上升。

下面逐节详述。

## 转换回放

做监督学习，第一步是下载 SC2 回放。

[REPLAY 指南](REPLAY_CHS.MD) 说明了如何下载 SC2 回放。

[知乎指南](https://zhuanlan.zhihu.com/p/410523216) 面向不便使用（中国大陆境外的）Battle.net 的中文用户。

下载好回放后，把它们放到 `./data/Replays/filtered_replays_1`（目录名可在
`transform_replay_data.py` 里改）。

然后用 `transform_replay_data.py` 把这些回放转成 pickle 或 tensor
（输出类型可在该文件里改）。

你不必直接运行 `transform_replay_data.py`，只运行 `run.py` 也可以。把 `run.py` 里的下面这段

```
    # from alphastarmini.core.sl import transform_replay_data
    # transform_replay_data.test(on_server=P.on_server)
```

取消注释，然后直接运行 `run.py`。

**注意**：要得到 gif 里那种效果，请使用
[Useful-Big-Resources](https://github.com/liuruoze/Useful-Big-Resources/blob/main/replays) 里的回放。
这些回放由我们的专家生成，用来训出能打赢内置 bot 的智能体。

## 监督学习

拿到可训练数据之后（推荐用 tensor 数据）。把 `run.py` 里的下面这段

```
    # from alphastarmini.core.sl import sl_train_by_tensor
    # sl_train_by_tensor.test(on_server=P.on_server)
```

取消注释，然后直接运行 `run.py` 即可做监督学习。

默认学习率是 1e-4，训练轮数最好取 10（再多的轮数可能导致过拟合）。

从 v_1.05 起，mAS 支持多 GPU 监督学习（目前不推荐），用于提速。用法很直接：

```
python run_multi-gpu.py
```

多 GPU 训练有些不稳定因素（由 PyTorch 引起）。如果遇到训练不稳定的报错，请切回单 GPU 训练。

目前支持四种监督训练方式，都在 `alphastarmini.core.sl` 包里。

文件 | 内容
------------ | -------------
`sl_train_by_pickle.py` | pickle（未预处理数据）训练：慢，但占磁盘小
`sl_train_by_tensor.py` | tensor（已预处理数据）训练：快，但占磁盘极大
`sl_multi_gpu_by_pickle.py` | 多 GPU + pickle：对共享内存要求高
`sl_multi_gpu_by_tensor.py` | 多 GPU + tensor：内存与共享内存都要大

可以用 `load_pickle.py` 把生成的 pickle（在 `./data/replay_data`）转成 tensor
（在 `./data/replay_data_tensor`）。

从 v_1.06 起，我们仍推荐单 GPU 训练。最新的训练方式（例如 v_1.07 里的）
也仍然是单 GPU 类型——多 GPU 训练太吃内存。

## 评测 SL 模型

拿到监督学习模型后，应当在 SC2 环境里测它的表现。原因是 SL 数据与 RL 环境之间存在**域偏移**。

把 `run.py` 里的下面这段

```
    # from alphastarmini.core.rl import rl_eval_sl
    # rl_eval_sl.test(on_server=P.on_server)
```

取消注释，然后直接运行 `run.py` 即可评测 SL 模型。

评测与 RL 训练类似，但更新是关闭的；运行也是单线程的，避免多线程的随机性影响评测。

## 强化学习

确认 SL 模型没问题、适合做 RL 之后，就可以在这个 SL 模型的基础上做强化学习。

把 `run.py` 里的下面这段

```
    # from alphastarmini.core.rl import rl_vs_inner_bot_mp
    # rl_vs_inner_bot_mp.test(on_server=P.on_server, replay_path=P.replay_path)
```

取消注释，然后直接运行 `run.py` 即可做强化学习。

注意 RL 训练采用多进程 + 多线程（为了加速），请确保在高性能机器上运行。

例如我们跑 15 个进程，每个进程 2 个 actor 线程和 1 个 learner 线程。机器不够强就调小并行数。

学习率要低（**1e-5 以下**，因为你是在一个已训练过的模型上继续训练）。
训练迭代次数越多越好（更多迭代能降低 RL 训练的不稳定性）。

如果训练结果不如预期，请开 issue 询问或讨论（尽管我们不保证及时回复，也不保证每个问题都有解）。

## 预训练模型

预训练模型在[这里](https://github.com/liuruoze/Useful-Big-Resources/blob/main/mAS-models)。

`sl_21-12-29_08-15-43.pth` 是用我们的方法训出的**监督学习**预训练模型。

`rl_22-02-07_16-26-48.pth` 是在上一个模型基础上用我们的方法**做强化学习**训出的模型。

`rl_22-02-09_19-16-39.pth` 是在上一个 SL 模型基础上、用我们的方法做**更长时间强化学习**得到的最终模型。

## 结果

下面是 SL 训练过程的一些示意图：

![SL training process](SL_traing.png)

可以看到损失（一个主损失与六个参数损失）下降很快。

v_1.05 之后，我们还提供了 SL 训练过程的更多曲线（如准确率曲线）。

训练出的智能体行为见本页的 gif。

我们后续的论文会给出更详细的实验说明（例如不同超参数的影响）。

## 历史

[HISTORY](HISTORY_CHS.MD) 是 mini-AS 各历史版本的介绍。

## 引用

如果您发现此存储库有用，请引用我们的项目或下面的技术报告：
```
@misc{liu2021mAS,
  author = {Ruo{-}Ze Liu and Wenhai Wang and Yang Yu and Tong Lu},
  title = {mini-AlphaStar},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuruoze/mini-AlphaStar}},
}
```

[An Introduction of mini-AlphaStar](https://arxiv.org/abs/2104.06890) 是一篇介绍 mini-AS 的技术报告（非完整版）。
```
@article{liu2021mASreport,
  author    = {Ruo{-}Ze Liu and
               Wenhai Wang and
               Yanjie Shen and
               Zhiqi Li and
               Yang Yu and
               Tong Lu},
  title     = {An Introduction of mini-AlphaStar},
  journal   = {CoRR},
  volume    = {abs/2104.06890},
  year      = {2021},
}
```

## 重思

[Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452) 是我们对 AlphaStar 优缺点的思考。

附：本仓库仍有若干未实现的部分（例如 z 的计算、vtrace 的损失部分、env 中待填充的信息）。

## 论文

我们在这篇[论文](https://arxiv.org/abs/2209.11553)里给出了使用 mini-AS 的详细实验与评估。
