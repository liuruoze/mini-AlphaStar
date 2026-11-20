# -*- coding: utf-8 -*-
"""把 ~/bvb/bench/*.jsonl 汇总成一张表（论文主表 + 基线）。

用法：
  python experiments/hybrid/make_final_table.py [--dir ~/bvb/bench]
"""

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S          # noqa: E402

BITS = ["补水晶", "补农民", "补兵营", "量产", "进攻"]

# 文件名 -> 展示名（按"整题/拆问"分组）
LABELS = {
    "qwen_full": ("Qwen2.5-7B", "整题"),
    "qwen14b_full": ("Qwen2.5-14B", "整题"),
    "qwen32b_whole": ("Qwen2.5-32B", "整题"),
    "glm4-9b_full": ("GLM-4-9B", "整题"),
    "qwen3-8b_full": ("Qwen3-8B(关思考)", "整题"),

    "qwen7b_full": ("Qwen2.5-7B", "拆问"),
    "qwen14b_split": ("Qwen2.5-14B", "拆问"),
    "qwen3-8b_split": ("Qwen3-8B(关思考)", "拆问"),
    "glm4-9b_split": ("GLM-4-9B", "拆问"),
    "qwen32b_plain": ("Qwen2.5-32B", "拆问"),
    "qwen32b_plain2": ("Qwen2.5-32B(重复)", "拆问"),
    "qwen32b_simple": ("Qwen2.5-32B 改写规则", "拆问"),
    "qwen32b_shots": ("Qwen2.5-32B 原文+示例", "拆问"),
    "qwen32b_shots6": ("Qwen2.5-32B 改写+示例", "拆问"),
    "qwen32b_shots8": ("Qwen2.5-32B 改写+更多示例", "拆问"),
    "qwen32b_emph": ("Qwen2.5-32B 强调兵营", "拆问"),
    "qwen32b_cap": ("Qwen2.5-32B 能力问法", "拆问"),
    "qwen32b_v3b": ("Qwen2.5-32B 平行题集", "拆问"),
    "qwen32b_jsonfmt": ("Qwen2.5-32B JSON 局面", "拆问"),
    "qwen3-32b_plain": ("Qwen3-32B(关思考)", "拆问"),
    "qwen14b_cond": ("Qwen2.5-14B 核对式问法", "拆问"),
    "qwen14b_vote3b": ("Qwen2.5-14B 三票制", "拆问"),
}


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    preds = [r.get("pred") for r in rows]
    truths = [r["label"] for r in rows]
    legal = [(p, t) for p, t in zip(preds, truths) if p and len(p) == 5]
    if not legal:
        return None
    sc = S.score_all([p for p, _ in legal], [t for _, t in legal])[0] * len(legal) / len(truths)
    exact = sum(1 for p, t in zip(preds, truths) if p == t) / float(len(truths))
    per = S.per_bit([p for p, _ in legal], [t for _, t in legal])
    return sc, exact, per, len(truths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/bvb/bench"))
    a = ap.parse_args()

    table = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        key = os.path.basename(f)[:-6]
        if key not in LABELS:
            continue
        r = load(f)
        if r:
            name, mode = LABELS[key]
            table.append((mode, name, *r))

    print("| 模式 | 模型/配置 | 得分 B | 全对率 | 逐位（%s） | 题数 |" % "/".join(BITS))
    print("|---|---|---:|---:|---|---:|")
    for mode, name, sc, exact, per, n in sorted(table, key=lambda x: (x[0] != "整题", -x[2])):
        print("| %s | %s | %.3f | %.1f%% | %s | %d |"
              % (mode, name, sc, 100 * exact, "/".join("%.0f" % (100 * x) for x in per), n))

    # 基线
    d = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
    items = d["items"]
    truths = [it["label"] for it in items]
    print("\n基线（同一题集）：")
    for nm, pred in [("什么都不做", "00000"), ("全都做", "11111"),
                     ("只看兵≥8", None)]:
        if pred is None:
            preds = ["0000" + ("1" if it["state"]["army"] >= 8 else "0") for it in items]
        else:
            preds = [pred] * len(items)
        sc = S.score_all(preds, truths)[0]
        print("  %-10s %.3f" % (nm, sc))


if __name__ == "__main__":
    main()
