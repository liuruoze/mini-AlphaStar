#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 A/B 的指令日志：每个 key 触发多少次、接受/排队/丢弃、以及对局结果。

用法:
  python experiments/hybrid/hybrid_stats.py <on_out_dir>
"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict


def outcome(txt):
    if not os.path.exists(txt):
        return None
    m = re.search(r"native_outcome:\s*(-?\d+)", open(txt, errors="ignore").read())
    return None if not m else int(int(m.group(1)) == 1)


def main():
    d = sys.argv[1]
    logs = sorted(glob.glob(os.path.join(d, "*.hybrid.jsonl")))
    if not logs:
        print("没有 .hybrid.jsonl")
        return
    keys, res = Counter(), Counter()
    per_game = []
    for f in logs:
        rows = [json.loads(l) for l in open(f, errors="ignore") if l.strip()]
        txt = f.replace(".hybrid.jsonl", "")
        w = outcome(txt)
        k = Counter(r.get("key") for r in rows if r.get("key"))
        r2 = Counter(r.get("res") for r in rows if r.get("res"))
        keys.update(k)
        res.update(r2)
        per_game.append((os.path.basename(txt)[:40], w, len(rows), sum(1 for r in rows if r.get("key"))))

    print("局数 %d" % len(logs))
    print("\n%-14s %8s" % ("策略", "次数"))
    for k, v in keys.most_common():
        print("%-14s %8d" % (k, v))
    print("\n%-14s %8s" % ("提交结果", "次数"))
    for k, v in res.most_common():
        print("%-14s %8d" % (k, v))

    wins = sum(1 for _, w, _, _ in per_game if w == 1)
    decided = sum(1 for _, w, _, _ in per_game if w in (0, 1))
    print("\n胜率 %d/%d = %.1f%%" % (wins, decided, 100.0 * wins / max(decided, 1)))
    print("\n%-42s %3s %6s %6s" % ("game", "win", "ticks", "cmds"))
    for name, w, n, c in per_game:
        print("%-42s %3s %6d %6d" % (name, str(w), n, c))


if __name__ == "__main__":
    main()
