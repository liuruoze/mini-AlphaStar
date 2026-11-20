#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对"8 分钟绝对量区分胜负"做两个对抗性检验，看结论是否会变。

检验 1（共线性）：军队与农民高度相关。按农民数分层后，军队还有区分力吗？
检验 2（幸存者偏差方向）：把没打到 8 分钟的局按 army=0 计入，结论会变强还是变弱？

用法:
  python experiments/macro_adversarial.py <trace_dir>:<out_dir> [...]
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load(td, od):
    games = []
    for f in sorted(glob.glob(os.path.join(td, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f, errors="ignore") if l.strip()]
        txt = os.path.join(od, os.path.basename(f).replace(".jsonl", ".txt"))
        win = None
        if os.path.exists(txt):
            m = re.search(r"native_outcome:\s*(-?\d+)", open(txt, errors="ignore").read())
            if m:
                win = int(int(m.group(1)) == 1)
        if win is not None:
            games.append({"rows": rows, "win": win})
    return games


def at(rows, t):
    near = [r for r in rows if abs(r["t_s"] - t) <= 20]
    if not near:
        return None
    r = near[0]
    return float(r.get("army", 0) or 0), float(r.get("workers", 0) or 0)


def auc(a, b):
    if not a or not b:
        return None
    gt = sum(1 for x in a for y in b if x > y)
    eq = sum(1 for x in a for y in b if x == y)
    return (gt + 0.5 * eq) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    args = ap.parse_args()

    games = []
    for pair in args.pairs:
        td, od = pair.split(":", 1)
        games.extend(load(td, od))

    pts = []
    for g in games:
        v = at(g["rows"], 480)
        if v:
            pts.append({"army": v[0], "workers": v[1], "win": g["win"]})
    print("样本：%d 局（胜 %d / 负 %d）" % (len(pts), sum(p["win"] for p in pts),
                                          sum(1 for p in pts if not p["win"])))

    # 检验 1：按农民中位数分层
    med = st.median([p["workers"] for p in pts])
    print("\n[检验1] 按农民数分层（中位 %.0f）" % med)
    for label, sel in (("高农民(>=%d)" % med, [p for p in pts if p["workers"] >= med]),
                       ("低农民(<%d)" % med, [p for p in pts if p["workers"] < med])):
        a = [p["army"] for p in sel if p["win"]]
        b = [p["army"] for p in sel if not p["win"]]
        if len(a) >= 2 and len(b) >= 2:
            print("  %-14s n=%d/%d  army %.1f vs %.1f  AUC=%.2f"
                  % (label, len(a), len(b), st.mean(a), st.mean(b), auc(a, b)))
        else:
            print("  %-14s 样本不足 n=%d/%d" % (label, len(a), len(b)))

    # 检验 2：把没到 8 分钟的局按 army=0 计入
    print("\n[检验2] 幸存者偏差方向：短局按 army=0 计入")
    a2, b2 = [], []
    for g in games:
        v = at(g["rows"], 480)
        army = v[0] if v else 0.0
        (a2 if g["win"] == 1 else b2).append(army)
    print("  含短局：n=%d/%d  army %.1f vs %.1f  AUC=%.2f"
          % (len(a2), len(b2), st.mean(a2), st.mean(b2), auc(a2, b2)))
    a1 = [p["army"] for p in pts if p["win"]]
    b1 = [p["army"] for p in pts if not p["win"]]
    print("  仅长局：n=%d/%d  army %.1f vs %.1f  AUC=%.2f"
          % (len(a1), len(b1), st.mean(a1), st.mean(b1), auc(a1, b1)))


if __name__ == "__main__":
    main()
