#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从评测的 15 秒时序采样（.jsonl）计算宏观指标 APU / PBR / RUR / TR。

指标定义（写死，避免口径漂移）:
  APU = mean(food_used / max(food_cap, 1))           平均人口利用率
  PBR = mean(food_used >= 0.95 * food_cap)            人口封顶时长占比
  RUR = (collected_minerals + collected_vespene) 的每分钟增量
  TR  = 已出现的 Protoss 科技建筑种类数 / 7           科技完成比例

用法:
  python experiments/macro_metrics.py <dir_or_glob> [...] [--json out.json] [--selftest]
  # 目录下同时有 <name>.jsonl 与 <name>.txt（后者提供 native_outcome）
"""

import argparse
import glob
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TECH_BUILDINGS = [
    "Forge", "CyberneticsCore", "TwilightCouncil", "RoboticsFacility",
    "Stargate", "TemplarArchives", "FleetBeacon",
]


def _unit_name_map():
    """unit_type id -> name。pysc2 里 U.Protoss.Forge 是枚举成员，id 取 .value。"""
    try:
        from pysc2.lib import units as U
        out = {}
        for race in (getattr(U, "Protoss", None), getattr(U, "Terran", None), getattr(U, "Zerg", None)):
            if race is None:
                continue
            for m in race:
                try:
                    out[int(m.value)] = m.name
                except Exception:
                    continue
        return out
    except Exception:
        return {}


_NAMES = _unit_name_map()


def _tech_seen(own_counts):
    seen = set()
    for uid in own_counts:
        nm = _NAMES.get(int(uid))
        if nm in TECH_BUILDINGS:
            seen.add(nm)
    return len(seen)


def load_game(jsonl_path):
    rows = []
    with open(jsonl_path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def outcome_of(txt_path):
    if not os.path.exists(txt_path):
        return None
    txt = open(txt_path, errors="ignore").read()
    import re
    m = re.search(r"native_outcome:\s*(-?\d+)", txt)
    return None if not m else (1 if int(m.group(1)) == 1 else 0)


def metrics(rows):
    if len(rows) < 2:
        return None
    # 采样时偶尔会读到 food_used 略大于 food_cap（正在排队的单位），利用率按 1.0 截断
    apu = [min(r["food_used"] / max(r["food_cap"], 1), 1.0) for r in rows]
    pbr = [1.0 if r["food_used"] >= 0.95 * r["food_cap"] and r["food_cap"] > 0 else 0.0 for r in rows]
    dur_min = max(rows[-1]["t_s"] - rows[0]["t_s"], 1.0) / 60.0
    gain = 0.0
    for key in ("collected_minerals", "collected_vespene"):
        if key in rows[0] and key in rows[-1]:
            gain += rows[-1][key] - rows[0][key]
    tech = max((_tech_seen(r.get("own", {})) for r in rows), default=0)
    return {
        "APU": statistics.mean(apu),
        "PBR": statistics.mean(pbr),
        "RUR": gain / dur_min,
        "TR": tech / len(TECH_BUILDINGS),
        "tech_types": tech,
        "minutes": dur_min,
        "max_army": max(r.get("army", 0) or 0 for r in rows),
        "max_workers": max(r.get("workers", 0) or 0 for r in rows),
        "samples": len(rows),
    }


def mannwhitney(a, b, n_perm=5000, seed=0):
    """不依赖 scipy 的置换检验（双侧）。样本量都很小时够用。"""
    import random
    rnd = random.Random(seed)
    if not a or not b:
        return None
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool = list(a) + list(b)
    na = len(a)
    hits = 0
    for _ in range(n_perm):
        rnd.shuffle(pool)
        d = abs(sum(pool[:na]) / na - sum(pool[na:]) / (len(pool) - na))
        if d >= obs - 1e-12:
            hits += 1
    return hits / n_perm


def selftest():
    tmp = os.path.join("/tmp", "macro_selftest.jsonl")
    rows = []
    for i in range(8):
        rows.append({"loop": int(i * 336), "t_s": float(i * 15), "food_used": 10 + i,
                     "food_cap": 15 + i, "minerals": 100, "vespene": 0, "army": i,
                     "workers": 12, "collected_minerals": 100.0 * i, "collected_vespene": 0.0,
                     "own": {"0": 3}})
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    m = metrics(load_game(tmp))
    assert abs(m["APU"] - statistics.mean([r["food_used"] / r["food_cap"] for r in rows])) < 1e-9
    assert 0.0 <= m["PBR"] <= 1.0 and m["RUR"] > 0 and m["TR"] == 0.0
    p = mannwhitney([1, 2, 3], [9, 10, 11])
    assert p is not None and p < 0.1
    print("selftest OK: APU=%.3f PBR=%.3f RUR=%.1f/min TR=%.2f p=%.3f"
          % (m["APU"], m["PBR"], m["RUR"], m["TR"], p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out-dir", default=None, help="评测输出目录；用来把 <name>.jsonl 对上 <name>.txt 的胜负")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
        else:
            files.extend(sorted(glob.glob(p)))
    if not files:
        ap.error("没有找到 .jsonl")

    games = []
    for f in files:
        rows = load_game(f)
        m = metrics(rows)
        if m is None:
            continue
        m["name"] = os.path.basename(f).replace(".jsonl", "")
        txt = f.replace(".jsonl", ".txt")
        if args.out_dir and not os.path.exists(txt):
            txt = os.path.join(args.out_dir, os.path.basename(txt))
        m["win"] = outcome_of(txt)
        games.append(m)

    wins = [g for g in games if g["win"] == 1]
    losses = [g for g in games if g["win"] == 0]
    keys = ["APU", "PBR", "RUR", "TR", "max_army", "max_workers"]

    print("%-40s %5s %6s %6s %7s %6s %7s" % ("game", "win", "APU", "PBR", "RUR/m", "TR", "army"))
    for g in games:
        print("%-40s %5s %6.3f %6.3f %7.1f %6.2f %7d"
              % (g["name"][:40], str(g["win"]), g["APU"], g["PBR"], g["RUR"], g["TR"], g["max_army"]))

    print("\n%-10s %14s %14s %8s" % ("metric", "wins(n=%d)" % len(wins), "losses(n=%d)" % len(losses), "p"))
    for k in keys:
        a = [g[k] for g in wins]
        b = [g[k] for g in losses]
        if not a or not b:
            continue
        print("%-10s %14s %14s %8s"
              % (k, "%.3f±%.3f" % (statistics.mean(a), statistics.pstdev(a)),
                 "%.3f±%.3f" % (statistics.mean(b), statistics.pstdev(b)),
                 ("%.3f" % mannwhitney(a, b)) if len(a) >= 2 and len(b) >= 2 else "-"))

    if args.json:
        json.dump(games, open(args.json, "w"), ensure_ascii=False, indent=2)
        print("\nwritten -> %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
