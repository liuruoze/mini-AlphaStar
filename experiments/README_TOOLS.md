# experiments/ 工具速查（2026-11-20 交接版）

> 背景：2026-11-19 全天实验的结论是「同配置换 seed，胜率 12.5%–62.5%（n=40，σ≈16pp）」。
> 因此**任何单跑对比都不可信**，下面的工具都是为了把"看起来更好"变成可判定的结论。

## 1. 训练

```bash
# 单跑（SL，4 epoch）
bash experiments/train_bvb_m30.sh <tag> 4 <seed> 3        # 第三个参数=训练 seed，第四个=step_size
# 需要类权重：
env MAS_SL_CLASS_WEIGHTS="train_probe=0.15,build_pylon=8,build_gateway=8,build_nexus=0.1,train_zealot=4" \
  bash experiments/train_bvb_m30.sh <tag> 4 <seed> 3
```

包装脚本 `round_runner.sh train <data> <tag> <lr> <epochs> [gpu] [weights]` 会顺手跑 preflight，
但它把 seed 写死成 20261108；要换 seed 就必须直接调 `train_bvb_m30.sh`。

### 架构开关（都在环境变量里，默认=历史行为）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAS_LSTM_HIDDEN` | 128 | LSTM 隐藏维度 |
| `MAS_LSTM_LAYERS` | 1 | LSTM 层数 |
| `MAS_LSTM_LN` | off | `off` / `out` / `pre` / `gate`（门级 LN） |
| `MAS_LN_GAMMA` | 1.0 | `out` 模式下 LayerNorm 的初始 γ |
| `MAS_SL_ENT_WEIGHT` | 0 | 动作熵正则权重（0.3–0.5 可阻止塌缩，但会压死策略） |
| `MAS_SL_ATTACK_WEIGHT` | 1.0 | Attack_pt 样本权重 |
| `MAS_SL_NOOP_WEIGHT` | 1.0 | no_op 样本权重 |
| `MAS_SL_ENT_VERBOSE` | 0 | 1 = 逐 batch 打印熵（默认每 200 次） |

**注意**：权重的结构与训练时必须是同一套开关，否则 `load_state_dict` 会静默丢参数。
每个 run 的 `model/sl/<tag>/train_manifest.json` 里记录了这些开关。

## 2. 评测

```bash
python experiments/rank_checkpoints.py --weights <ckpt> --seeds 3038-3077 --jobs 8 --out-dir <dir>
```
- 一定要用**同一组种子**做配对比较；
- 默认 40 局；`σ≈16pp`，所以只看 `summarize_runs.py` 的区间，不要看裸胜率。

## 3. 汇总（必用）

```bash
python experiments/summarize_runs.py --all ~/bvb/arch_ab
python experiments/summarize_runs.py base=.../base3_out new=.../b0s5_out
```
输出：win/decided/attempted、Wilson 区间、按 seed 的配对差值、以及所有 run 的均值±标准差。
**分母用 attempted（含崩溃局）**：只统计"有 outcome"的局会把胜率抬高（b0s5 从 62.5% 变 80.6%）。

## 4. 离线探针（不需要 SC2，秒级）

```bash
# 模型实际预测了哪些动作、熵、置信度
python experiments/probe_action_dist.py <ckpt> --file 7 --nframes 288
# 数据集标签分布
python experiments/probe_label_hist.py --files 30
```
判读经验：**acc > 0.6 且熵 < 0.5、置信度 > 0.85** ⇒ 已经塌缩到少数动作，直接判负，省一次 30 分钟评测。

## 5. 单测

```bash
python experiments/test_ln_lstm.py     # 13 项：off 与 nn.LSTM 等价、out 与手算 LN 一致、梯度、NaN、退化形状
```

## 6. 监控

```bash
bash experiments/status_snapshot.sh <tag1> [tag2 ...]
```
一条命令输出：时间 / 训练与评测进程数 / 每个 tag 的最新 val acc 与 checkpoint 数 / GPU / 内存。

## 7. 今日得到的三条硬约束

1. **SL 的 run 间 σ≈16pp（n=40）**：少于 3 个 seed 的 A/B 不要下结论；
2. **val action acc 与胜率反相关**：数据集里 Attack_pt 占 53.6%、总共只有 11 类动作，
   模型押多数类就能拿到 0.5+ 的准确率；acc 只能当"是否塌缩"的诊断；
3. **LN 的三种放置都会塌缩**，降 lr 只推迟不阻止；熵 0.3–0.5 能阻止塌缩但策略不敢决断（0 胜）。
