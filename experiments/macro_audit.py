#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""审计宏观指标分析本身：采样是否可信、胜负是否配错、固定时点比较有没有幸存者偏差。

检查项：
  A1 每个 trace 的样本数、时间跨度、缺失字段（army=-1 之类）
  A2 胜负配对：jsonl 与 .txt 的局数/结果是否一一对应
  A3 幸存者偏差：各时间点上，胜负两组各自有多少局还有采样
  A4 固定时点比较：带 n 与置换检验 p 值
  A5 诊断"APU 高但军队少"：直接看 food_workers / army 的构成

用法:
  python experiments/macro_audit.py <trace_dir>:<out_dir> [...]
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


def perm_p(a, b, n=5000, seed=0):
    if len(a) < 2 or len(b) < 2:
        return None
    rnd = random.Random(seed)
    obs = abs(st.mean(a) - st.mean(b))
    pool = list(a) + list(b)
    na, hits = len(a), 0
    for _ in range(n):
        rnd.shuffle(pool)
        if abs(st.mean(pool[:na]) - st.mean(pool[na:])) >= obs - 1e-12:
            hits += 1
    return hits / n


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
        games.append({"name": os.path.basename(f), "rows": rows, "win": win})
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--times", default="180,300,480,600")
    args = ap.parse_args()
    times = [int(t) for t in args.times.split(",")]

    for pair in args.pairs:
        td, od = pair.split(":", 1)
        games = load(td, od)
        print("\n=== %s ===" % td)
        if not games:
            print("无数据")
            continue

        # A1
        ns = [len(g["rows"]) for g in games]
        spans = [g["rows"][-1]["t_s"] - g["rows"][0]["t_s"] for g in games if g["rows"]]
        miss_army = sum(1 for g in games for r in g["rows"] if r.get("army", -1) < 0)
        miss_work = sum(1 for g in games for r in g["rows"] if r.get("workers", -1) < 0)
        print("A1 局数=%d 样本/局 min=%d med=%d max=%d 时长 min=%.0fs med=%.0fs"
              % (len(games), min(ns), st.median(ns), max(ns), min(spans), st.median(spans)))
        print("   army 字段缺失=%d  workers 缺失=%d" % (miss_army, miss_work))

        # A2
        paired = [g for g in games if g["win"] is not None]
        print("A2 胜负可配对=%d/%d  胜=%d 负=%d"
              % (len(paired), len(games), sum(g["win"] == 1 for g in paired),
                 sum(g["win"] == 0 for g in paired)))

        # A3 + A4
        for t in times:
            grp = {}
            for label, want in (("win", 1), ("loss", 0)):
                sel = []
                for g in paired:
                    if g["win"] != want:
                        continue
                    near = [r for r in g["rows"] if abs(r["t_s"] - t) <= 20]
                    if near:
                        r = near[0]
                        sel.append((min(r["food_used"] / max(r["food_cap"], 1), 1.0),
                                    r.get("army", 0), r.get("workers", -1), r.get("food_cap", 0)))
                grp[label] = sel
            w, l = grp["win"], grp["loss"]
            if not w or not l:
                print("A3 t=%-3ds 只有一侧有数据 (win=%d loss=%d)" % (t, len(w), len(l)))
                continue
            pw = perm_p([x[0] for x in w], [x[0] for x in l])
            pa = perm_p([x[1] for x in w], [x[1] for x in l])
            # 固定窗口 RUR：开局到 t 的采集速率（每分钟），避免"局越长总量越大"
            def rur(g):
                rows = g["rows"]
                near = [r for r in rows if abs(r["t_s"] - t) <= 20]
                if not near or "collected_minerals" not in near[0]:
                    return None
                r = near[0]
                mins = max(r["t_s"] / 60.0, 0.1)
                return (r.get("collected_minerals", 0) + r.get("collected_vespene", 0)) / mins
            rw = [rur(g) for g in paired if g["win"] == 1]
            rl = [rur(g) for g in paired if g["win"] == 0]
            rw = [x for x in rw if x is not None]
            rl = [x for x in rl if x is not None]
            pr = perm_p(rw, rl) if rw and rl else None
            pc = perm_p([x[3] for x in w], [x[3] for x in l])
            print("A3 t=%-3ds n=%d/%d  APU %.2f vs %.2f (p=%.3f)  army %.1f vs %.1f (p=%.3f)  cap %.0f vs %.0f (p=%.3f)"
                  % (t, len(w), len(l), st.mean(x[0] for x in w), st.mean(x[0] for x in l), pw,
                     st.mean(x[1] for x in w), st.mean(x[1] for x in l), pa,
                     st.mean(x[3] for x in w), st.mean(x[3] for x in l), pc))
            if pr is not None:
                print("        RUR@%ds %.0f vs %.0f /min (p=%.3f)" % (t, st.mean(rw), st.mean(rl), pr))

        # A5
        t = 300
        sel = []
        for g in paired:
            near = [r for r in g["rows"] if abs(r["t_s"] - t) <= 20]
            if near:
                sel.append((near[0]["food_used"], near[0]["food_cap"],
                            near[0].get("workers", -1), near[0].get("army", 0)))
        if sel:
            print("A5 t=300s 平均 food_used=%.0f cap=%.0f 其中农民=%.1f 军队=%.1f"
                  % (st.mean(x[0] for x in sel), st.mean(x[1] for x in sel),
                     st.mean(x[2] for x in sel), st.mean(x[3] for x in sel)))


if __name__ == "__main__":
    main()
