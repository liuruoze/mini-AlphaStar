#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pool evaluation results by tag and report 真胜 / 宽胜 with Wilson intervals.

Usage:
    python experiments/pool_tags.py C1_h5noop C2_h5noop
    python experiments/pool_tags.py --out /path/to/outputs HumanHybrid_c1 HumanHybrid_c2

Every result file is one episode: outputs/<tag>s<seed>_<seed>.txt.
The tag must match up to the literal "s<seed>_" suffix, so tags that only differ
in their tail cannot silently absorb each other's files.

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

# outcome / raw_reward / killed_points are printed on the first line of every result file.
FIELDS = re.compile(
    r"outcome:\s*([+-]?\d+).*?raw_reward:\s*([+-]?\d+).*?killed_points:\s*([\d.]+)"
)


def wilson(k, n, z=1.96):
    """Wilson score interval; returns (lo, hi) as fractions."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def load(out_dir, tag):
    """Return [(seed, strict_win, lenient_win)] for one tag, deduplicated by seed."""
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
        strict = raw == 1
        lenient = outcome == 1 or (outcome == 0 and killed > 4000)
        rows[seed] = (seed, strict, lenient)
    return [rows[s] for s in sorted(rows)]


def line(label, rows):
    n = len(rows)
    if n == 0:
        return "%-18s 无数据" % label
    strict = sum(1 for r in rows if r[1])
    lenient = sum(1 for r in rows if r[2])
    lo, hi = wilson(strict, n)
    return "%-18s n=%-3d 真胜=%2d (%5.1f%%, [%4.1f,%4.1f])  宽胜=%2d (%5.1f%%)" % (
        label, n, strict, 100.0 * strict / n, 100 * lo, 100 * hi,
        lenient, 100.0 * lenient / n,
    )


def main():
    ap = argparse.ArgumentParser(description="按 tag 池化评测结果")
    ap.add_argument("tags", nargs="+", help="评测 tag（可多个，会额外给出合计）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="结果目录，默认 $OFF_OUT 或 ~/mini-AlphaStar-off/outputs")
    args = ap.parse_args()

    if not os.path.isdir(args.out):
        sys.exit("结果目录不存在: %s" % args.out)

    pooled = []
    for tag in args.tags:
        rows = load(args.out, tag)
        print(line(tag, rows))
        pooled.extend(rows)

    if len(args.tags) > 1:
        print("-" * 72)
        print(line("池化", pooled))
        print("覆盖 %d 个不同种子（同一批种子被多个 tag 重复评测时按局计，不去重）"
              % len({r[0] for r in pooled}))


if __name__ == "__main__":
    main()
