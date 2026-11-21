# hybrid 三臂说明（2026-11-22）

## 三条臂是什么

| 臂 | `MAS_HYBRID_MODE` | 谁在决定 | 用途 |
|---|---|---|---|
| `sl` | 不设 `MAS_HYBRID` | 纯 SL 策略，无指挥官 | 基线 |
| `doctrine` | `doctrine` | `doctrine.decide_bits` 直接算 | 教条本身的上界（无判断误差） |
| `llm_v3` | `llm_v3` | 32B 逐条回答五个"是/否" | 教条 + LLM 执行 |

三者共用同一套位序与命令映射：

```
BIT_NAMES = [补水晶, 补农民, 补兵营, 量产狂热者, 全面进攻]
PRIORITY  = pylon > probe > attack > gateway > zealot   （2026-11-21 用户裁定：进攻提到兵营之前）
```

## 为什么要第三条臂

之前只能比 `sl` 与 `llm_v3`。若 `llm_v3` 不如 `sl`，无法判断是**教条不好**还是**模型执行不准**。
加了 `doctrine` 之后：

- `doctrine` ≈ `llm_v3` → 模型没拖后腿，问题在教条；
- `doctrine` 明显好于 `llm_v3` → 差的是模型的判断误差（离线实测 6/120 题判错，≈5%）；
- 两者都不如 `sl` → 教条本身不如策略自发行为。

## 关键环境变量

| 变量 | 作用 | 备注 |
|---|---|---|
| `MAS_HYBRID=1` | 总开关 | **漏了会静默退回纯 SL**（踩过） |
| `MAS_HYBRID_MODE` | `rule` / `doctrine` / `llm_v3` / `tree` | 默认 `rule` |
| `MAS_HYBRID_BIAS` | 偏置权重（如 `3.0`） | 设了就走软偏置；**不设则走硬遮罩** |
| `MAS_HYBRID_INTERVAL` | 决策间隔（**游戏秒**） | 默认 8；仿真约 16×，故实际约 0.5 墙钟秒 |
| `LLM_BASE_URL` | 推理服务地址 | 默认 `http://127.0.0.1:8769/v1` |
| `MAS_MACRO_STATS` | 偏置计数落盘路径 | `{"calls","flip","sum_abs_delta"}` |
| `MAS_V3_DRYRUN=1` | 只读观测、不调 LLM | 诊断用 |
| `EVAL_MODEL` | **必须显式指定** | 不设会回落到字典序最大的弱权重 |

## 两种干预必须互斥（2026-11-21 修复）

```python
if _bw and _cmd is not None:    # 偏置
elif _bw:                        # 偏置模式 + 无指令 = 不干预
else:                            # 遮罩模式
```

旧代码把"偏置模式下指挥官无指令"也送进了 else，结果在那些 tick 上施加了**硬遮罩**。
实测污染率 9.4%（6418 次决策中 603 次），使 bias 臂实际是 bias+mask 混合。
修复后 716 次决策遮罩计数为 0，且"零偏置"与纯 SL 逐位一致（5400/122/37）。

## 自测

```bash
python experiments/hybrid/run_selftests.py
# 32 项：归一化 14、位映射 8、教条臂 6、遮罩/偏置算术 4
```

## 常用命令

```bash
# 单局（d3，教条臂）
DISPLAY=:1024 SC2PATH=~/StarCraftII MAS_ON_SERVER=0 \
EVAL_MODEL=./model/sl/cwH5h_s20261207/sl_26-11-15_09-37-33_best.pth \
MAS_HYBRID=1 MAS_HYBRID_MODE=doctrine MAS_HYBRID_BIAS=3.0 \
EVAL_DIFFICULTY=3 EVAL_MAX_LOOPS=21000 EVAL_OUT=/tmp/one \
  python run_rl_eval.py

# 汇总与统计
python experiments/hybrid/collect_ab.py /tmp/ab6
python experiments/hybrid/analyze_ab.py /tmp/ab6/summary.csv
```

## 已知限制

- 击杀数不能直接跨臂比较：赢局会提前结束，必须按时间归一或只看胜率；
- `EVAL_MAX_LOOPS` 的单位是**游戏步**（22.4 步 = 1 游戏秒）；
- 同 seed 可复现，但**不同难度梯度非单调**（d1≈d2 >> d3）。
