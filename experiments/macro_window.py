#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在**固定时间点**上比较胜负局的 APU/军队，排除"打赢的局更长"这个混杂因素。

背景：整局的 APU 均值会被局时长影响（赢的局打到 12 分钟，输的局 6 分钟就投），
所以"胜局 APU 高"可能只是"胜局更长"。这里把时钟对齐到 180/300/480 秒再比。

用法:
  python experiments/macro_window.py <trace_dir>:<out_dir> [<trace_dir>:<out_dir> ...]
      [--times 180,300,480]
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


def load(d, od):
    games = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f, errors="ignore") if l.strip()]
        if len(rows) < 3:
            continue
        txt = os.path.join(od, os.path.basename(f).replace(".jsonl", ".txt"))
        win = None
        if os.path.exists(txt):
            m = re.search(r"native_outcome:\s*(-?\d+)", open(txt, errors="ignore").read())
            if m:
                win = int(int(m.group(1)) == 1)
        games.append((rows, win))
    return games


def at(rows, t):
    best = min(rows, key=lambda r: abs(r["t_s"] - t))
    apu = min(best["food_used"] / max(best["food_cap"], 1), 1.0)
    return apu, best.get("army", 0) or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="trace_dir:out_dir")
    ap.add_argument("--times", default="180,300,480")
    args = ap.parse_args()
    times = [int(t) for t in args.times.split(",")]

    for pair in args.pairs:
        td, od = pair.split(":", 1)
        games = load(td, od)
        if not games:
            print("%s: 没有数据" % td)
            continue
        for label, sel in (("win", [g for g in games if g[1] == 1]),
                           ("loss", [g for g in games if g[1] == 0]),
                           ("all", games)):
            if not sel:
                continue
            parts = []
            for t in times:
                vals = [at(rows, t) for rows, _ in sel]
                parts.append("t%d APU=%.2f army=%.1f" % (t, st.mean(v[0] for v in vals), st.mean(v[1] for v in vals)))
            print("%-22s %-4s n=%-2d  %s" % (os.path.basename(td), label, len(sel), " | ".join(parts)))


if __name__ == "__main__":
    main()
