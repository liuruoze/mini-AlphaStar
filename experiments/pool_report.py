#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配方级汇总：把一个配方下多个 run（每个 run 可能用过多个 OFF_TAG 别名）的评测结果合并，
给出「逐模型」与「配方合并」的真胜率，并附 Wilson 95% 置信区间。

用法::

    python experiments/pool_report.py --seeds 3038-3097 \\
        --group "h5=171:h5v,h5v3|172:..."  # 不建议手写

更简单的用法（推荐）：每个 run 一行，用 ';' 分隔；配方名用 '=' 开头重新声明::

    python experiments/pool_report.py --seeds 3038-3097 \\
        --group "h5=A:cwH5_s20261171,cwH5_s20261171_e,h5v,h5v3" \\
        --group "h5=B:cwH5_s20261172,cwH5_s20261172_e,h5v2"

同一配方可以多次 --group（同名前缀），脚本会把它们合并。

判胜两种口径都打印：
  native  : raw_reward == +1
  official: outcome == 1 或 (outcome == 0 且 killed_points > 4000)
"""
import argparse
import glob
import math
import os
import re
import sys

OUT_CANDIDATES = [
    os.path.expanduser("~/mini-AlphaStar-off/outputs"),
    os.path.expanduser("~/mini-AlphaStar/outputs"),
]


def find_out_dir():
    for d in OUT_CANDIDATES:
        if os.path.isdir(d):
            return d
    return OUT_CANDIDATES[0]


OUT = find_out_dir()


def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def parse_line(path):
    try:
        with open(path, "r", errors="ignore") as f:
            head = f.readline()
    except Exception:
        return None
    if "raw_reward" not in head:
        return None

    def g(key, cast=float):
        m = re.search(r"%s:\s*([-+0-9.eE]+)" % key, head)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:
            return None

    d = {
        "raw": g("raw_reward", int),
        "outcome": g("outcome", int),
        "killed": g("killed_points"),
        "steps": g("steps"),
        "food": g("food_used"),
        "army": g("army_count"),
        "our": g("our_units"),
    }
    if d["raw"] is None:
        return None
    d["native"] = 1 if d["raw"] == 1 else 0
    off = 0
    if d["outcome"] == 1:
        off = 1
    elif d["outcome"] == 0 and (d["killed"] or 0) > 4000:
        off = 1
    d["official"] = off
    return d


def find_file(tag, seed):
    pats = ["%ss%d_%d.txt" % (tag, seed, seed), "%s_%d.txt" % (tag, seed)]
    pats += [os.path.basename(p) for p in glob.glob(os.path.join(OUT, "%s_*_%d.txt" % (tag, seed)))]
    pats += [os.path.basename(p) for p in glob.glob(os.path.join(OUT, "%s_*%d_%d.txt" % (tag, seed, seed)))]
    for p in pats:
        fp = p if os.path.isabs(p) else os.path.join(OUT, p)
        if os.path.isfile(fp) and os.path.getsize(fp) > 0:
            return fp
    return None


def collect_model(tags, seeds):
    """同一模型的多个 tag 别名按 seed 合并（一个 seed 只取第一个有结果的别名）。"""
    seen = {}
    used = {t: 0 for t in tags}
    for t in tags:
        for s in seeds:
            if s in seen:
                continue
            fp = find_file(t, s)
            if not fp:
                continue
            d = parse_line(fp)
            if d is None:
                continue
            d["seed"] = s
            seen[s] = d
            used[t] += 1
    return [seen[s] for s in sorted(seen)], used


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / float(n)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="如 3038-3097 或 3038,3040")
    ap.add_argument("--group", action="append", required=True,
                    help='如 "h5=A:tag1,tag2"（配方名=run名:别名列表）')
    ap.add_argument("--detail", action="store_true", help="打印每个 run 的逐种子胜负")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    recipes = {}   # recipe -> list of (runname, tags)
    for g in args.group:
        if "=" not in g:
            raise SystemExit("--group 需要 '配方=run:tag,tag' 形式: %s" % g)
        rname, rest = g.split("=", 1)
        rname = rname.strip()
        if ":" in rest:
            run, tags = rest.split(":", 1)
        else:
            run, tags = rest, rest
        tags = [t.strip() for t in tags.split(",") if t.strip()]
        recipes.setdefault(rname, []).append((run.strip(), tags))

    print("输出目录: %s\n种子: %d 个 (%d..%d)\n" % (OUT, len(seeds), seeds[0], seeds[-1]))

    for rname, runs in recipes.items():
        print("=" * 78)
        print("配方 %s —— %d 个 run" % (rname, len(runs)))
        print("-" * 78)
        print("%-26s %6s %6s %8s %8s   %s" % ("run", "n", "native", "rate", "official", "Wilson95"))
        tot_n = tot_w = tot_off = 0
        rates = []
        for run, tags in runs:
            rows, used = collect_model(tags, seeds)
            n = len(rows)
            w = sum(r["native"] for r in rows)
            o = sum(r["official"] for r in rows)
            p, lo, hi = wilson(w, n)
            print("%-26s %6d %6d %7.1f%% %8.1f%%   [%.1f%%, %.1f%%]  %s"
                  % (run, n, w, 100 * p, 100 * (o / n if n else 0), 100 * lo, 100 * hi,
                     "" if n else "(无输出文件)"))
            if args.detail and rows:
                print("     " + " ".join("%d:%s" % (r["seed"], "+" if r["native"] else "-") for r in rows))
            if n:
                rates.append(p)
                tot_n += n
                tot_w += w
                tot_off += o
        if tot_n:
            p, lo, hi = wilson(tot_w, tot_n)
            print("-" * 78)
            print("合并      n=%d  native=%d (%.1f%%)  official=%d (%.1f%%)  Wilson95=[%.1f%%, %.1f%%]"
                  % (tot_n, tot_w, 100 * p, tot_off, 100 * tot_off / tot_n, 100 * lo, 100 * hi))
            if len(rates) > 1:
                mean = sum(rates) / len(rates)
                var = sum((x - mean) ** 2 for x in rates) / (len(rates) - 1)
                print("run 级：均值 %.1f%%  标准差 %.1f%%（n_run=%d，说明 run 间方差）"
                      % (100 * mean, 100 * math.sqrt(var), len(rates)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
