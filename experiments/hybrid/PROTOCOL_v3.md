# 评测协议与复现指南（v3）

> 目的：回答「LLM 能否按项目作者的狂热者快攻战术做宏观决策」。
> 结论摘要见 `BENCH_RESULTS_20261120.md`；本文件说明**怎么跑、每个文件是什么**。

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `prompt_spec_v3.md` | 协议正文：system 原文、user 模板、指标、判定门槛、修订记录 |
| `doctrine.py` | 六条规则的可执行版本 + 五位编码（`decide` / `decide_bits`） |
| `score.py` | **唯一的评分实现**（方案 B：全对 1 / 错一位 0.5 / 其余 0） |
| `make_eval_set_v3.py` | 生成并冻结题集；`--seed` 可产平行题集 |
| `eval_set_v3.json` | 冻结的 120 题（sha256 见文件与 `.sha256`） |
| `eval_set_v3b.json` | 平行题集（不同种子，用于泛化检验） |
| `audit_eval_set_v3.py` | 结构/物理/因果/时间/经济/边界翻转等 17 项检查 |
| `audit_deep_v3.py` | **双重录入**：按规范原文另写一套规则重算标签再比对 |
| `analyze_rules_coverage.py` | 每个临界值命中多少题、组合分布、盲区 |
| `baselines_v3.py` | 平凡解基线（什么都不做 / 全都做 / 单一启发式） |
| `bench_llm.py` | 整题模式：一次要 5 位输出 |
| `bench_split.py` | 拆问模式：每条规则单独问一次，是/否，再拼成 5 位 |
| `make_final_table.py` | 把所有落盘结果汇总成论文主表 |
| `sc2_terms_*.{json,md,py}` | 神族术语对照（英文/官方简体/官方繁体/玩家俗称）+ 别名解析 |
| `llm_server.py` / `llm_client.py` | OpenAI 兼容的本地推理服务与客户端 |

## 2. 复现步骤

```bash
# ① 生成题集（已冻结，一般不需要重跑）
python experiments/hybrid/make_eval_set_v3.py
python experiments/hybrid/make_eval_set_v3.py --seed 20261121 \
       --out experiments/hybrid/eval_set_v3b.json      # 平行题集

# ② 体检（三项都必须 0 失败）
python experiments/hybrid/audit_eval_set_v3.py
python experiments/hybrid/audit_deep_v3.py
python experiments/hybrid/analyze_rules_coverage.py

# ③ 基线
python experiments/hybrid/baselines_v3.py

# ④ 起服务（示例：32B，多卡自动切分）
LLM_DEVICE_MAP=auto python experiments/hybrid/llm_server.py \
    --model-path ~/models/Qwen/Qwen/Qwen2___5-32B-Instruct --port 8769

# ⑤ 跑模型
python experiments/hybrid/bench_llm.py   --model qwen32b --base-url http://127.0.0.1:8769/v1
python experiments/hybrid/bench_split.py --model qwen32b --base-url http://127.0.0.1:8769/v1

# ⑥ 汇总
python experiments/hybrid/make_final_table.py
```

## 3. 关键设计决定（为什么这么做）

| 决定 | 理由 |
|---|---|
| **五位独立判断**（不是五选一） | 基地造农民、兵营造兵、农民造建筑互不冲突，可以同时进行 |
| **评分方案 B** | 「部分对就给 0.5」会让"什么都不做"拿 0.517；B 下是 0.158 |
| **题集含 31 条边界项** | 每个阈值配 4–8 条，模型若忽略上下文会在这些位上暴露 |
| **物理/时间/经济三重约束** | 早期版本出现过 4 分钟 200 人口、有兵无兵营等不可能局面 |
| **冻结 + sha256** | 跑模型前锁死题集，避免"改题迁就结果" |
| **双重录入校验** | 规范文本、生成脚本、评分脚本三处实现必须一致 |

## 4. 已知限制

1. **只测单图（Simple64）、单对手（内置 AI 难度 1）**；
2. **只测宏观选择**，不含指位与微操；
3. 题集覆盖 5–15 分钟、单基地的局面；
4. 标准答案来自作者口述的六条规则，**不是胜负**——得分高只说明"听话"，不说明"能赢"；
5. 平行题集 v3b 与主集同分布，不能替代真实对局分布。

## 5. 变更记录

| 日期 | 版本 | 变化 |
|---|---|---|
| 2026-11-20 | v1 | 33 条人工标注拟合的单选战术（弃用） |
| 2026-11-20 | v2 | 作者修订的单选战术（弃用） |
| 2026-11-20 | v3 | 多选五位判断；评分方案 B；题集冻结 |
| 2026-11-20 | v3 | 19:30 补「上限已满不补水晶」；19:33 补「农民需余量≥1」、删空闲农民字段 |
