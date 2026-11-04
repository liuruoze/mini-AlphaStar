# mini-AlphaStar — 2026-11 improvement campaign

> 本文件是 `README.md` 顶部 branch note 的详细版，记录 2026-11-02 → 11-04 这段工作：
> 做了什么、测到什么数字、哪些结论被推翻、还有哪些没解决。
>
> **署名**
> - 监督 / 仓库所有者：**Ruo-Ze Liu (`liuruoze`)**
> - 执行：**DeepSeek-V4.1-Flash**（改动编号 #23–#53）
> - 代码 review：**GLM**（#1–#22）、**Kimi 3.1 Pro**、**MiniMax H3.1**、**space-bunny**
>
> 记录纪律（由监督者定）：每条结论必须带**可证伪判据 + 实测数字**；
> 每条被撤回的结论进 `ERRATA.md`，不静默删除。

---

## 1. 一页总结

| 阶段 | 起点 | 终点 | 参照 |
|---|---|---|---|
| SL 胜率 | 0%（n=20） | **4.3%**（n=230，Wilson95 [2.4, 7.8]） | 官方 SL **17.5%**（n=80，[10.7, 27.3]） |
| SL 采集 | 2,676 | **9,010** | 官方 9,494（**95%**） |
| SL 击杀 | 0 | **1,439** | 官方 1,811（79%） |
| SL 军队 | — | **21.2** | 官方 14.0（**151%**） |
| RL 循环 | **从未执行过** | 能跑：轨迹 / loss / KL / win_rate 全部有输出 | — |
| RL 胜率 | — | 0%（9 局） | 官方 RL 21.4% |

**一句话**：SL 从"完全不会玩"做到"会玩、能打、但赢不了"；
RL 从"跑不起来"做到"循环正确"，但 30–60 分钟稳定性**尚未达成**。

---

## 2. SL 阶段：三个结构性缺陷（不是超参问题）

### 2.1 静默数据降级（`SIMPLE_TEST`）

```python
# 原实现（sl_train_by_tensor.py）
SIMPLE_TEST = not P.on_server
...
if SIMPLE_TEST:
    replay_files = replay_files[:2]      # 只读 1 个训练 + 1 个验证
```

忘记设 `MAS_ON_SERVER=1`，2.5 小时的训练就只用了 **1/60 的 replay**
（日志铁证：`data split: train_files[0:1] (1)`、`train_set 120`）。
**程序不报错、loss 照常下降**——这是最危险的一类 bug。

修复：冒烟必须显式 `MAS_SL_SMOKE=1`；新增 `MAS_SL_MAX_FILES`；首屏打印
`[SL-config] files=… smoke=… lr=… epochs=…`。
结果：collected 2,676 → 6,072（+127%）。

### 2.2 动作分布被采样方式扭曲（11–19 倍）

数据转换按固定帧步长（约每 20 帧取一帧），而 agent 每 8 帧才下一条命令 ——
绝大多数采样帧落在两条命令之间，标签就是 `no_op`。

| 动作 | 训练数据 | 官方模型实战 | 倍数 |
|---|---|---|---|
| `no_op` | **81.0%** | 0% | — |
| `Build_Pylon` | 2.7% | 29.8% | 11× |
| `Train_Probe` | 1.1% | 21.1% | 19× |
| `Build_Gateway` | 0.7% | 12.9% | 19× |

**模型是数据的忠实镜像**：数据里 81% 是发呆，实战就 99.9% 发呆
（`army_count=0`、`used_points=600` 固定不变）。

修复：**动作配重**（importance weighting），只作用于 `action_type` 头：

```python
w[a] = clip((target_share[a] / data_share[a]) ** exp, lo=0.1, hi=50)   # exp 必须 1.0
```

- `exp=0.5`（平方根）不够：no_op 仍 30%、army 5.4、8 负；
- `exp=1.0` 才有效：实战 no_op **37% → 6.7%**、Attack 稳定在 10%（官方 9.5%）、
  army 16.4 → **24.4**、collected 7,261 → **8,981**。

⚠️ 踩过的坑：第一版把 `Smart_unit` 降权去"对齐官方词汇表"，而**我们的数据里
`Harvest_Gather` 是 0%**，Smart 是唯一采集指令 —— 降它等于关掉采矿
（collected 8,660 → 4,552）。**重加权只能在自己数据的动作空间里做。**

### 2.3 用 val accuracy 选模型是反的

| checkpoint | val action acc | 实战 no_op | 评价 |
|---|---|---|---|
| ep03 | 0.44 | **50%** | 行为最好 |
| ep10（`_best.pth` 选中的） | 0.83 | **96.8%** | 退回发呆 |

`val action acc` 奖励保守（no_op 占验证集 81%，预测它就能拿高分）。
**模型选择必须用行为指标**（killed / no_op 占比 / 胜率），不是分类准确率。

### 2.4 零成本技巧：权重空间平均

同一 run 的相邻 epoch 权重处在同一线性模式连通区域，插值不产生"四不像"，
而是互补能力叠加（`experiments/_avg_models.py`）：

| 组合 | 效果 |
|---|---|
| 3 个（ep01+ep02+ep03） | kill 尾部 2,550 → **6,900**、胜率 1.9% → **5.1%**（n=158） |
| 5 个 | **更差**：尾部 2,800、army 15.8（激进的那一轮被稀释） |

### 2.5 SL 的可复现最优配置

```bash
MAS_ON_SERVER=1 MAS_SL_MAX_FILES=20 MAS_SL_WORKERS=0 MAS_SL_EPOCHS=2 \
MAS_SL_ACTION_REWEIGHT=1 MAS_SL_PRIOR_EXP=1.0 \
CUDA_VISIBLE_DEVICES=0 python run_sl.py --path1 ./data/replay_data_v3/
```

**训练量按「replay-epochs（文件数 × 轮数）」控制，30 最优**：
20 文件 × 2 轮 = 30 → 有胜局；20×3 = 45 → 0/51；60×2 = 110 → 更差（no_op 56-79%）。

---

## 3. RL 阶段：六个让循环跑起来的加固

RL 主循环（`rl_vs_inner_bot_mp`，A3C）在这之前**一次都没执行过**。启动它需要先修六处：

| # | 问题 | 症状 | 修法 |
|---|---|---|---|
| **#47** | 权重静默选错 | `models.sort()[-1]` 选到**微调失败**的 checkpoint；冠军名是大写 `SL`，`"sl" in …` 为 False，永远选不到 | `MAS_RL_INIT_MODEL` 显式指定 + 打印参数量/权重 L2 范数/missing 键数 |
| **#48a** | `STATIC_NUM=100` 硬编码 | 30–50 分钟跑不满 100 局 → **win_rate 曲线根本不出现** | `MAS_STATIC_NUM` 可配 |
| **#48b** | 统计表维度为 0 | `int(train_iters/STATIC_NUM)=0` → shape (0,N) → 首次赋值 IndexError（冒烟必崩） | `max(1, …)` |
| **#49** | 无可观测性 | 多进程安静时无法区分"在跑/死锁/已崩" | `MAS_HEARTBEAT_SEC` 心跳 |
| **#50** | spawn 子进程常量不一致 | `param.py` 的 `on_server` 硬编码 False → 子进程 `TRAIN_ITERS=1` → **`Parameter_Server` 收 1 局就退** | `SIMPLE_TEST` 同时认 `MAS_ON_SERVER` |
| **#51** | 无条件调试打印 | `print(...) if 1 else None` → 4 分钟 14,213 行（60 行/秒），关键信息全淹没 | `MAS_RL_DEBUG=1` 门控 |
| **#52** | 单局帧上限随训练时长缩小 | `max_frames_per_episode = 22.4 × MAX_TIME`，短跑时 < 一局所需 → 每局静默 `break`、**不投递 outcome** → 队列永远空 | 默认 0（不限），`MAS_MAX_FRAMES_PER_EP` 可覆盖 |

### 3.1 跑通后的实测

```
loss/learner_0                    n=256  last=0.148  (min -1.78, max 1.23)
learner_0/loss_upgo_              n=255  0.0028 ~ 0.2015
learner_0/all_kl_loss_action_type n=255  0 ~ 0.0014
winrate/every_4_episodes          n=2    0.0 / 0.0
meanpoints/every_4_episodes       n=2    0.283 / 0.239
```

**KL ≤ 0.0014 是符合预期的**：策略从 SL 权重初始化，teacher 就是同一个 SL agent，
所以 student≈teacher。反过来说，这也证明**权重真的加载了**（随机初始化会到 1 以上）。

---

## 4. 未解决的两个问题（交接给下一位）

### 4.1 长跑 #1：t=1201 s 静默猝死

```
最后一行：[hb] t= 1201s  proc=5/5  q_winloss=0  v_steps=67776  mem_avail=100.7GB
之后：日志同刻截断，python 进程消失，8 个 SC2 成孤儿
内存：available 100 GB -> 不是 OOM
```

无 traceback → 不在 Python 层，而在**进程组被统一终止**。

### 4.2 长跑 #2：34 分钟 0 局

进程活着、6 个 SC2 活着、主进程 CPU 只用了 10 s（阻塞在 join），
但 `native_outcome` 一直是 0 —— **episode 未被终止/记录**。

### 4.3 一个影响所有观测的问题（#53）

`python run_rl.py > log/x.log` 是**普通文件 + 各自的文件偏移**，
而 Python 主进程 + 6–8 个 SC2 子进程**同时往同一个 fd 写** → 互相覆盖。
所以"心跳停在第 N 秒"很可能不是线程死了，而是**后续内容被覆盖**。

**修复（下一位请直接采用）**：
```bash
python -u run_rl.py >> log/x.log 2>&1     # O_APPEND 原子追加
# 或更好：每个进程/每个 rank 写自己的文件（SummaryWriter 已经是这样做的）
```

### 4.4 建议的下一步

1. 用 `>>` + 每进程独立日志，重跑 40 分钟，先确认**日志不再自我覆盖**；
2. 在 actor 里每 100 步打印 `episode_frames / total_frames`，确认 episode 终止条件；
3. 若仍不终止，显式设 `MAS_MAX_FRAMES_PER_EP=21000`（= 一局上限）作为兜底；
4. 吞吐 13.5 step/s（8 actor）偏低：开 `MAS_BATCH_ACTING=1` 或增大 `buffer_size`（现在每 16 条轨迹就 update 一次）。

---

## 5. 否证清单（负面结果也是结果）

| 假设 | 结果 | 数字 |
|---|---|---|
| 数据量越多越好 | ❌ | 55 文件版：collected 9,242↑ 但 army 13.4↓、killed 823↓ |
| 训练越久越好 | ❌ | 20 文件 × 10 轮：no_op 回到 96.8% |
| `Smart_unit` 替换是 killed=0 的根因 | ❌ | 关掉后 20 局 0 胜、collected 5,977（更低） |
| `SEQ_LEN=8`（完整时序）能救长程行为 | ❌ | 12 分钟/epoch，val acc 无优势，中止 |
| Attack 加权能提升收割 | ❌ | 1.5×：collected 4,441；3.0×：负局率 33%→42% |
| `units/location` 权重 10/5（作者注释值） | ❌ | army **3.7**、collected **3,159** |
| 按交战密度筛 replay | ❌ **灾难** | 32 局 0 胜 4 平 **28 负**，killed 84（正常 1,222） |
| 多版本平均越多越稳 | ❌ | 5 个版本：尾部 6,900 → 2,800 |
| 从最优平均再微调 1 轮 | ❌ | 19 局 1 胜 **6 负**（败局率 32% vs 5.2%） |
| 攻击坐标总在自家附近 | ❌ | 官方均值 4.8 比我们 8.0 **更近**，照样打出 7,875 击杀 |
| `unit_tags` 送的是下标所以指令全失效 | ❌ | 强行加 index→tag 映射后**更差**（collected 7,000→3,206） |
| 官方 SL 胜率 = 7.1% | ❌ | 同 harness 80 局实测 **17.5%**；7.1% 是 28 局点估计 |
| `val action acc` 可以选模型 | ❌ **反向** | ep03 val 0.44 / no_op 50% **优于** ep10 val 0.83 / no_op 96.8% |
| `SEQ_LEN×0.5` 是我们改坏的 | ❌ | `git blame`：原作者 2022-05-02 引入，上游一致 |

> 这 14 条里，有 5 条曾被写成结论，后来被自己或 reviewer 撤回（见 `ERRATA.md`）。

---

## 6. 改动索引

| 编号 | 文件 | 内容 |
|---|---|---|
| #23 | `sl_train_by_tensor.py` / `run_sl.py` | 显式冒烟、可控文件数/workers、首屏 `[SL-config]` |
| #24 | `sl_train_by_tensor.py` | 每 epoch 独立 `_epNN.pth` + `_best.pth` |
| #25 | `sl_train_by_tensor.py` | `MAS_SL_EPOCHS`；SEQ_LEN 保持官方设计 |
| #26 | `sl_loss_multi_gpu.py` | `Smart_unit→Harvest` 替换开关（默认关） |
| #27 | `sl_train_by_tensor.py` | `MAS_SL_SEQ_LEN` |
| #28 | `rl_eval_sl.py` | `MAS_ACT_STATS` 实战动作分布 |
| #29 | `rl_eval_sl.py` | `MAS_DUMP_STATE` 输入形状 |
| #30 | `sl_loss_multi_gpu.py` | `MAS_SL_NOOP_WEIGHT` 样本降权 |
| #31 | `sl_loss_multi_gpu.py` | `MAS_SL_ATTACK_WEIGHT` |
| #32 | `experiments/_eval_winrate.py` | 评测入口 BLAS 线程=1（**吞吐 ×5.9**） |
| #33 | `experiments/_soup.py` → `_avg_models.py` | 权重插值 / N-way 平均 |
| #34 | `sl_train_by_tensor.py` | `sorted(os.listdir())`（原来划分是随机的） |
| #35 | `experiments/_rank_replays_by_combat.py` | 按动作密度排序（判据错了，工具留用） |
| #36 | `sl_loss_multi_gpu.py` | `MAS_SL_UNITS_W/TARGET_W/LOC_W` |
| #37-#39 | `rl_eval_sl.py` | tag 核对、index→tag A/B、全量动作直方图 |
| #40-#41 | `sl_loss_multi_gpu.py` | **动作配重**（`build_action_prior_weights`） |
| #42 | `rl_eval_sl.py` | 攻击坐标统计 |
| #43 | `sl_loss_multi_gpu.py` | `MAS_SL_TGT_*` 允许扫参 |
| #44 | `rl_eval_sl.py` | `MAS_TIME_STATS` 时间线（埋点未生效） |
| #45-#46 | `sl_train_by_tensor.py` | 续训开关 + 修 `RESTORE`/`LOAD_CHECKPOINT` 守卫 |
| **#47-#52** | `lib/utils.py` / `rl_vs_inner_bot_mp.py` / `rl_loss.py` | **RL 循环的六处加固（见 §3）** |
| #53 | 运行方式 | `>>` 追加模式（多进程日志自覆盖，见 §4.3） |

---

## 7. 术语对照

论文术语对读日志的人不友好，本仓库统一用白话（旧名保留兼容）：

| 原术语 | 现在叫法 | 一句话 |
|---|---|---|
| action prior / prior reweight | **动作配重** | 数据里少的动作多算几遍、多的少算几遍，把比例掰回目标分布 |
| model soup / weight averaging | **多版本平均** | 同一个 run 的第 1/2/3 轮参数按位取平均，零训练成本 |
| replay-epoch | **训练量（文件数 × 轮数）** | 15 文件 × 2 轮 = 30；单看"轮数"会被文件数骗 |
| no_op | **空转 / 发呆** | 模型输出"什么都不做"的比例 |
| native_outcome | **胜负（0 平 / 1 胜 / -1 负）** | "平"= 打到 21,000 帧上限，**不是势均力敌** |
