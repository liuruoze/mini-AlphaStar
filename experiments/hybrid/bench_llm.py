# -*- coding: utf-8 -*-
"""在冻结题集上跑一个模型，输出得分与诊断。

用法：
  python experiments/hybrid/bench_llm.py --model qwen --base-url http://127.0.0.1:8769/v1 \
         --out ~/bvb/bench/qwen7b_v3.jsonl
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
BIT_NAMES = ["补水晶", "补农民", "补兵营", "量产", "进攻"]


def parse_bits(raw):
    """剥思考块 → 去空白标点 → 取第一个长度 5 的 0/1 串。"""
    if not raw:
        return None
    t = raw
    # 去掉思考块（DeepSeek/Qwen 的  thinking...）
    if "<｜end▁of▁thinking｜>" in t:
        t = t.split("<｜end▁of▁thinking｜>")[-1]
    t = re.sub(r"[^01]", "", t)
    if len(t) < 5:
        return None
    for i in range(len(t) - 4):
        cand = t[i:i + 5]
        if cand.count("1") + cand.count("0") == 5:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    d = json.load(open(a.data, encoding="utf-8"))
    items = d["items"][:a.limit] if a.limit else d["items"]
    system = d["system"]
    llm = llm_client.build(a.backend, base_url=a.base_url, model_name=a.model)

    out_path = a.out or os.path.join(os.path.expanduser("~"), "bvb", "bench",
                                     "%s_%s.jsonl" % (a.model.replace("/", "_"), a.tag or "v3"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    preds, truths, rows, lat = [], [], [], []
    for i, it in enumerate(items, 1):
        t0 = time.time()
        raw = llm.chat([{"role": "system", "content": system},
                        {"role": "user", "content": it["prompt"]}],
                       max_new_tokens=a.max_tokens, temperature=0.0)
        dt = time.time() - t0
        pred = parse_bits(raw)
        rows.append({"id": it["id"], "label": it["label"], "pred": pred,
                     "raw": (raw or "")[:200], "sec": round(dt, 3)})
        preds.append(pred)
        truths.append(it["label"])
        lat.append(dt)
        if i % 20 == 0:
            print("  ...%d/%d" % (i, len(items)), flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 只把合法预测交给评分，格式失败单独统计
    legal = [(p, t) for p, t in zip(preds, truths) if p]
    total, exact, _ = S.score_all([p for p, _ in legal], [t for _, t in legal])
    n_bad = len(truths) - len(legal)
    # 格式失败按 0 分计入总分
    total_all = total * len(legal) / float(len(truths))
    per = S.per_bit([p for p, _ in legal], [t for _, t in legal])

    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95) - 1]

    print("\n=== %s on %s ===" % (a.model, os.path.basename(a.data)))
    print("题数 %d｜格式失败 %d" % (len(truths), n_bad))
    print("得分 B（格式失败算 0）: %.3f" % total_all)
    print("全对率: %.1f%%" % (100 * exact))
    print("逐位准确率: " + "  ".join("%s %.0f%%" % (n, 100 * x) for n, x in zip(BIT_NAMES, per)))
    print("延迟 p50 %.2fs｜p95 %.2fs" % (p50, p95))
    print("结果: %s" % out_path)

    # 错在哪
    print("\n错误分布（错几位）：")
    w = {}
    for p, t in legal:
        k = sum(1 for x, y in zip(p, t) if x != y)
        w[k] = w.get(k, 0) + 1
    for k in sorted(w):
        print("  错 %d 位: %d 题" % (k, w[k]))
    if w.get(5):
        print("  （全错 %d 题，注意是否恒答同一个组合）" % w[5])
    from collections import Counter
    print("最常见的 3 种回答: %s" % Counter(p for p, _ in legal).most_common(3))


if __name__ == "__main__":
    main()
