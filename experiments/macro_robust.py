#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对"某个时点的某个量能否区分胜负"做稳健性推断：bootstrap 区间、Cliff's delta、AUC、Holm 校正。

为什么需要：2026-11-20 的结论翻转过两次，原因都是"统计口径"而不是数据本身。
这个脚本把口径固定成一套，并显式给出不确定性，避免下一次再换口径就换结论。

用法:
  python experiments/macro_robust.py <trace_dir>:<out_dir> [more...] --time 480 --metric army
  python experiments/macro_robust.py ... --grid 180,300,480,600   # 一次看多个时点并做 Holm 校正
"""

import argparse
import glob
import json
import os
import random
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


def value_at(rows, t, metric, win_s=20):
    near = [r for r in rows if abs(r["t_s"] - t) <= win_s]
    if not near:
        return None
    r = near[0]
    if metric == "army":
        return float(r.get("army", 0) or 0)
    if metric == "apu":
        return min(r["food_used"] / max(r["food_cap"], 1), 1.0)
    if metric == "cap":
        return float(r.get("food_cap", 0))
    if metric == "workers":
        return float(r.get("workers", 0))
    raise ValueError(metric)


def boot_ci(a, b, n=10000, seed=0):
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n):
        sa = [rnd.choice(a) for _ in a]
        sb = [rnd.choice(b) for _ in b]
        diffs.append(st.mean(sa) - st.mean(sb))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def perm_p(a, b, n=20000, seed=0):
    rnd = random.Random(seed)
    obs = abs(st.mean(a) - st.mean(b))
    pool = list(a) + list(b)
    na, hits = len(a), 0
    for _ in range(n):
        rnd.shuffle(pool)
        if abs(st.mean(pool[:na]) - st.mean(pool[na:])) >= obs - 1e-12:
            hits += 1
    return hits / n


def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def auc(a, b):
    """a=胜，b=负；返回 P(a>b) 的秩统计量（0.5 = 无区分）。"""
    gt = sum(1 for x in a for y in b if x > y)
    eq = sum(1 for x in a for y in b if x == y)
    return (gt + 0.5 * eq) / (len(a) * len(b))


def holm(pvals):
    """返回 Holm 校正后的 p（与输入同序）。"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, prev = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)
        out[i] = prev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--time", type=int, default=480)
    ap.add_argument("--metric", default="army")
    ap.add_argument("--grid", default=None, help="给定时做多时点比较并 Holm 校正")
    args = ap.parse_args()

    games = []
    for pair in args.pairs:
        td, od = pair.split(":", 1)
        games.extend(load(td, od))
    if not games:
        print("没有可用数据")
        return

    if args.grid:
        times = [int(t) for t in args.grid.split(",")]
        raw = []
        print("%-6s %-8s %12s %12s %8s %8s" % ("t", "metric", "win", "loss", "p", "p_holm"))
        for t in times:
            a, b = [], []
            for g in games:
                v = value_at(g["rows"], t, args.metric)
                if v is None:
                    continue
                (a if g["win"] == 1 else b).append(v)
            if len(a) < 2 or len(b) < 2:
                print("%-6d %-8s 样本不足 (n=%d/%d)" % (t, args.metric, len(a), len(b)))
                raw.append((t, None))
                continue
            p = perm_p(a, b)
            raw.append((t, p))
            print("%-6d %-8s %5.1f±%-5.1f %5.1f±%-5.1f %8.3f" %
                  (t, args.metric, st.mean(a), st.pstdev(a), st.mean(b), st.pstdev(b), p))
        ps = [p for _, p in raw if p is not None]
        if ps:
            adj = holm(ps)
            k = 0
            print("\nHolm 校正：")
            for t, p in raw:
                if p is None:
                    continue
                print("  t=%-4ds p=%.3f -> %.3f %s" % (t, p, adj[k], "显著" if adj[k] < 0.05 else "不显著"))
                k += 1
        return

    a, b = [], []
    for g in games:
        v = value_at(g["rows"], args.time, args.metric)
        if v is None:
            continue
        (a if g["win"] == 1 else b).append(v)
    print("样本：胜 %d / 负 %d（t=%ds, metric=%s）" % (len(a), len(b), args.time, args.metric))
    if len(a) < 2 or len(b) < 2:
        print("样本不足，无法推断")
        return
    lo, hi = boot_ci(a, b)
    print("均值：%.2f vs %.2f   差值 %.2f  bootstrap 95%% CI [%.2f, %.2f]"
          % (st.mean(a), st.mean(b), st.mean(a) - st.mean(b), lo, hi))
    print("中位数：%.1f vs %.1f" % (st.median(a), st.median(b)))
    print("置换检验 p=%.4f   Cliff's delta=%.2f   AUC=%.2f"
          % (perm_p(a, b), cliffs_delta(a, b), auc(a, b)))


if __name__ == "__main__":
    main()
