#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split evaluation results by spawn corner (左上 / 右下) and report 真胜 / 宽胜.

Usage:
    python experiments/stat_corner.py C1_h5noop C2_h5noop
    python experiments/stat_corner.py HumanHybrid_c1:HumanHybrid ...

The 20 standard seeds are not an even split: 8 spawn at left-top and 12 at
right-bottom (see docs/exp/NAMING.md §7.5). Pooling them hides large gaps —
one weight can be 50% left-top and 83% right-bottom, and the official weight is
88% vs 8%. Always report per corner.

Environment:
    OFF_OUT   output directory, defaults to ~/mini-AlphaStar-off/outputs
"""

import argparse
import glob
import math
import os
import re
import sys

DEFAULT_OUT = os.environ.get("OFF_OUT") or os.path.expanduser("~/mini-AlphaStar-off/outputs")

# Measured with experiments/_probe_spawn.py over seeds 3038-3057.
# Seeds outside this range have no measured corner: they go to an explicit 未知
# bucket instead of being silently counted as 右下, which would fake the split.
LEFT_TOP = {3038, 3040, 3045, 3046, 3051, 3053, 3054, 3055}
KNOWN_SEEDS = set(range(3038, 3058))
CORNERS = ("左上", "右下")

FIELDS = re.compile(
    r"outcome:\s*([+-]?\d+).*?raw_reward:\s*([+-]?\d+).*?killed_points:\s*([\d.]+)"
)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def load_corner_map(path):
    """Optional override: lines of 'seed <n>: ... -> <corner>' from _probe_spawn.py."""
    mapping = {}
    if not path:
        return mapping
    if not os.path.exists(path):
        sys.exit("角位映射文件不存在: %s" % path)
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = re.match(r"seed (\d+):.*?->\s*(\S+)", ln)
            if m:
                mapping[int(m.group(1))] = m.group(2)
    return mapping


def load(out_dir, tag):
    exact = re.compile(r"^" + re.escape(tag) + r"s(\d+)_\d+\.txt$")
    rows = {}
    for path in sorted(glob.glob(os.path.join(out_dir, tag + "s*_*.txt"))):
        m = exact.match(os.path.basename(path))
        if not m:
            continue
        seed = int(m.group(1))
        with open(path, encoding="utf-8", errors="ignore") as fh:
            head = fh.readline()
        mm = FIELDS.search(head)
        if not mm:
            continue
        outcome, raw, killed = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
        rows[seed] = (seed, raw == 1, outcome == 1 or (outcome == 0 and killed > 4000), killed)
    return [rows[s] for s in sorted(rows)]


def corner_of(seed, override):
    """Return 左上 / 右下, or None when the corner was never measured."""
    if seed in override:
        return override[seed]
    if seed not in KNOWN_SEEDS:
        return None
    return "左上" if seed in LEFT_TOP else "右下"


def emit(label, rows):
    n = len(rows)
    if n == 0:
        print("%-16s 无数据" % label)
        return
    strict = sum(1 for r in rows if r[1])
    lenient = sum(1 for r in rows if r[2])
    lo, hi = wilson(strict, n)
    kills = sorted(r[3] for r in rows)
    print("%-16s n=%-2d 真胜=%2d (%5.1f%%, [%4.1f,%4.1f])  宽胜=%2d (%5.1f%%)  击杀中位=%.0f" % (
        label, n, strict, 100.0 * strict / n, 100 * lo, 100 * hi,
        lenient, 100.0 * lenient / n, kills[n // 2],
    ))


def main():
    ap = argparse.ArgumentParser(description="按角位分层统计评测结果")
    ap.add_argument("specs", nargs="+", help="tag 或 tag:标签")
    ap.add_argument("--out", default=DEFAULT_OUT, help="结果目录，默认 $OFF_OUT 或 ~/mini-AlphaStar-off/outputs")
    ap.add_argument("--map", dest="mapfile", default="", help="可选：_probe_spawn.py 输出，覆盖内置角位表")
    args = ap.parse_args()

    if not os.path.isdir(args.out):
        sys.exit("结果目录不存在: %s" % args.out)
    override = load_corner_map(args.mapfile)

    for spec in args.specs:
        tag, label = (spec.split(":", 1) + [spec])[:2]
        rows = load(args.out, tag)
        emit(label, rows)
        for corner in CORNERS:
            emit("  " + corner, [r for r in rows if corner_of(r[0], override) == corner])
        unknown = [r for r in rows if corner_of(r[0], override) is None]
        if unknown:
            emit("  未知角位", unknown)
            print("  注意：%d 局的种子在 3038–3057 之外，角位未实测；"
                  "先用 _probe_spawn.py 补测再用 --map 传入" % len(unknown))
        print("-" * 76)


if __name__ == "__main__":
    main()
