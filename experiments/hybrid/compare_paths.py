#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比同一 seed 下 SL 臂与 hybrid 臂的执行轨迹。

SL 臂：`MAS_STATE_LOG` 写的 <path>，字段 t/used/cap/workers/army/gw/pylons/minerals
hybrid 臂：`.hybrid.jsonl` 的 `why` 字段（v3:xxx|w%d a%d g%d py%d）

用法：python compare_paths.py <seed> [<seed> ...]
"""

import glob
import json
import re
import sys

WHY = re.compile(r"(\w+):(\w+)\|w(\d+) a(\d+) g(\d+) py(\d+)")


def load_sl(seed):
    pts = []
    for p in ("/tmp/st_sl_%d.jsonl" % seed,):
        if not glob.glob(p):
            continue
        for line in open(p, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            pts.append((r["t"], r["workers"], r["army"], r["gw"], r["pylons"]))
    return sorted(pts)


def load_hyb(seed):
    pts = []
    for p in glob.glob("/tmp/ab6/d3_hyb_s%d.hybrid.jsonl" % seed):
        for line in open(p, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            m = WHY.search(r.get("why") or "")
            if m:
                pts.append((r["t"], int(m.group(3)), int(m.group(4)),
                            int(m.group(5)), int(m.group(6))))
    return sorted(pts)


def first_time(pts, idx, thresh):
    for t, *v in pts:
        if v[idx] >= thresh:
            return t
    return None


def at_time(pts, t0):
    best = None
    for p in pts:
        if p[0] <= t0:
            best = p
        else:
            break
    return best


def main(seeds):
    for s in seeds:
        sl, hy = load_sl(s), load_hyb(s)
        print("== seed %d" % s)
        for name, pts in (("SL", sl), ("v3", hy)):
            if not pts:
                print("   %-3s 无数据" % name)
                continue
            a8 = first_time(pts, 1, 8)
            g1 = first_time(pts, 2, 1)
            p1 = first_time(pts, 3, 1)
            row = at_time(pts, 600)
            print("   %-3s 首水晶 %-5s 首兵营 %-5s 8兵 %-5s ｜ t=600s: 农%d 兵%d 营%d 晶%d" % (
                name,
                "%.0fs" % p1 if p1 else "—",
                "%.0fs" % g1 if g1 else "—",
                "%.0fs" % a8 if a8 else "—",
                row[1] if row else -1, row[2] if row else -1,
                row[3] if row else -1, row[4] if row else -1))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [52, 56, 57, 66, 51, 54])
