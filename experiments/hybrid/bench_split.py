# -*- coding: utf-8 -*-
"""把 5 位拆成 5 次独立单问：每次只问"现在该不该做这件事？"

动机：一次要求 5 个条件同时成立时，小模型会坍缩成恒定输出（见 BENCH_RESULTS_20261120.md）。
拆开后每次只需判断一个条件，输出也从 5 位降到 是/否。

用法：
  python experiments/hybrid/bench_split.py --model qwen --base-url http://127.0.0.1:8765/v1 --tag split
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import llm_client          # noqa: E402
import score as S          # noqa: E402

DATA = os.path.join(HERE, "eval_set_v3.json")

# 每条规则单独成问：system 里只给这一条，避免变相做整题
RULES = [
    ("补水晶", "补水晶：没有水晶，或者人口余量（上限−已用）≤ 9 且人口上限未满 200。"),
    ("补农民", "补农民：农民 ≤ 14 且人口余量 ≥ 1。"),
    ("补兵营", "补兵营：兵营 < 3 且至少有一个水晶（没有水晶就没有能量场，兵营放不下）。"),
    ("量产狂热者", "量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2。"),
    ("全面进攻", "全面进攻：兵 ≥ 8。"),
]

SYS_TMPL_ACTION = """你是星际争霸2神族的战术指挥官，本局打狂热者快攻。现在只判断一件事：

%s

结合下面的局面，回答“现在是否应该执行这件事”。只输出一个字：是 或 否，不要解释。"""

# 换一种问法：只问条件是否成立（纯逻辑核对，不许权衡后果）
SYS_TMPL_COND = """这是一道核对题。下面给出一个条件和一份局面数据，判断该条件此刻是否成立：

%s

只输出一个字：是（成立）或 否（不成立），不要解释。"""

SYS_TMPL = SYS_TMPL_ACTION


def parse_yes(raw):
    t = (raw or "").strip()
    if "<｜end▁of▁thinking｜>" in t:
        t = t.split("<｜end▁of▁thinking｜>")[-1]
    t = t.strip()
    if t.startswith("是") or t[:1] in ("是", "Y", "y", "1"):
        return "1"
    if t.startswith("否") or t[:1] in ("否", "N", "n", "0", "不"):
        return "0"
    # 兜底：句子里先出现哪个
    i_yes, i_no = t.find("是"), t.find("否")
    if i_yes >= 0 and (i_no < 0 or i_yes < i_no):
        return "1"
    if i_no >= 0:
        return "0"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="split")
    ap.add_argument("--phrasing", default="action", choices=["action", "condition"])
    a = ap.parse_args()

    d = json.load(open(a.data, encoding="utf-8"))
    items = d["items"][:a.limit] if a.limit else d["items"]
    llm = llm_client.build(a.backend, base_url=a.base_url, model_name=a.model)
    tmpl = SYS_TMPL_COND if a.phrasing == "condition" else SYS_TMPL_ACTION

    out_path = a.out or os.path.join(os.path.expanduser("~"), "bvb", "bench",
                                     "%s_%s.jsonl" % (a.model.replace("/", "_"), a.tag))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows, preds, truths, lat = [], [], [], []
    for n, it in enumerate(items, 1):
        bits, raws, t0 = "", [], time.time()
        for name, rule in RULES:
            sysmsg = tmpl % rule
            r = llm.chat([{"role": "system", "content": sysmsg},
                          {"role": "user", "content": it["prompt"]}],
                         max_new_tokens=a.max_tokens, temperature=0.0)
            b = parse_yes(r)
            bits += (b if b else "?")
            raws.append((r or "")[:40])
        lat.append(time.time() - t0)
        rows.append({"id": it["id"], "label": it["label"], "pred": bits, "raw": raws})
        preds.append(bits if "?" not in bits else None)
        truths.append(it["label"])
        if n % 20 == 0:
            print("  ...%d/%d" % (n, len(items)), flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    legal = [(p, t) for p, t in zip(preds, truths) if p]
    total, exact, _ = S.score_all([p for p, _ in legal], [t for _, t in legal])
    total_all = total * len(legal) / float(len(truths))
    per = S.per_bit([p for p, _ in legal], [t for _, t in legal])
    lat.sort()
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95) - 1]

    print("\n=== %s（拆成 5 次单问）===" % a.model)
    print("题数 %d｜解析失败 %d" % (len(truths), len(truths) - len(legal)))
    print("得分 B: %.3f" % total_all)
    print("全对率: %.1f%%" % (100 * exact))
    print("逐位: " + "  ".join("%s %.0f%%" % (n, 100 * x) for n, x in zip(
        ["补水晶", "补农民", "补兵营", "量产", "进攻"], per)))
    print("单题总耗时 p50 %.2fs｜p95 %.2fs（含 5 次请求）" % (p50, p95))
    print("结果: %s" % out_path)

    from collections import Counter
    print("答“是”的比例: " + "  ".join(
        "%s %.0f%%" % (n, 100 * sum(1 for p, _ in legal if p[i] == "1") / len(legal))
        for i, n in enumerate(["补水晶", "补农民", "补兵营", "量产", "进攻"])))
    print("最常见的 3 种回答: %s" % Counter(p for p, _ in legal).most_common(3))


if __name__ == "__main__":
    main()
