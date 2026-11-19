#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总多个评测目录，给出胜率、Wilson 区间，以及按种子的配对比较。

为什么需要它：2026-11-19 的 6 个基线 seed 胜率从 12.5% 到 62.5%（n=40），
标准差约 17pp。任何单跑之间的差异都在噪声里。这个脚本把"看起来更好"变成
"配对差值 + 区间"，避免再用单跑下结论。

用法:
  python experiments/summarize_runs.py a=~/bvb/arch_ab/b0s3_out b=~/bvb/arch_ab/b0s4_out
  python experiments/summarize_runs.py --all ~/bvb/arch_ab
"""

import argparse
import glob
import math
import os
import re
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def parse_dir(path):
    """返回 {seed: outcome}，outcome ∈ {1 win, 0 loss, None 无结果（崩溃/未完成）}

    注意：无结果的局必须计入分母。昨天的教训是"只统计有 outcome 的局"会把
    崩溃局悄悄排除，胜率虚高（s5 因此从 62.5% 变成 80.6%）。
    """
    out = {}
    for f in glob.glob(os.path.join(path, "*.txt")):
        txt = open(f, errors="ignore").read()
        m = re.search(r"native_outcome:\s*(-?\d+)", txt)
        s = re.search(r"sc2_seed:\s*(\d+)", txt)
        if not m or not s:
            continue
        v = int(m.group(1))
        out[int(s.group(1))] = 1 if v == 1 else (0 if v == -1 else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="name=path 或直接 path")
    ap.add_argument("--all", default=None, help="汇总某个根目录下所有子目录")
    args = ap.parse_args()

    items = []
    if args.all:
        for d in sorted(glob.glob(os.path.join(args.all, "*_out"))):
            items.append((os.path.basename(d).replace("_out", ""), d))
    for a in args.dirs:
        if "=" in a:
            k, v = a.split("=", 1)
            items.append((k, os.path.expanduser(v)))
        else:
            items.append((os.path.basename(a.rstrip("/")), os.path.expanduser(a)))

    if not items:
        ap.error("没有输入目录")

    data = {}
    print("%-14s %4s %4s %4s %7s  %-16s" % ("run", "win", "dec", "all", "rate", "95% CI (Wilson)"))
    for name, d in items:
        r = parse_dir(d)
        n_all = len(r)                      # 所有有结果行的局（含崩溃=未计入）
        k = sum(1 for v in r.values() if v == 1)
        n_dec = sum(1 for v in r.values() if v is not None)
        lo, hi = wilson(k, n_all)
        data[name] = {s: v for s, v in r.items() if v is not None}
        print("%-14s %4d %4d %4d %6.1f%%  [%5.1f, %5.1f]" % (name, k, n_dec, n_all, 100.0 * k / max(n_all, 1), 100 * lo, 100 * hi))

    names = [n for n, _ in items]
    if len(names) >= 2:
        base = names[0]
        print("\n配对比较（同一 sc2_seed，基准 = %s）" % base)
        print("%-14s %6s %6s %6s %8s" % ("vs", "共同", "基准胜", "对手胜", "差值"))
        for other in names[1:]:
            common = sorted(set(data[base]) & set(data[other]))
            if not common:
                print("%-14s %6d    -      -        -" % (other, 0))
                continue
            a = sum(data[base][s] for s in common)
            b = sum(data[other][s] for s in common)
            print("%-14s %6d %6d %6d %+7.1fpp" % (other, len(common), a, b, 100.0 * (b - a) / len(common)))

        # 全部 run 的均值/标准差（衡量噪声底）
        rates = [100.0 * sum(v.values()) / max(len(v), 1) for v in data.values()]
        if len(rates) >= 2:
            mean = sum(rates) / len(rates)
            sd = math.sqrt(sum((x - mean) ** 2 for x in rates) / (len(rates) - 1))
            print("\n%d 个 run 的胜率：均值 %.1f%%，样本标准差 %.1fpp，极差 %.1f–%.1f%%"
                  % (len(rates), mean, sd, min(rates), max(rates)))


if __name__ == "__main__":
    sys.exit(main())
