#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank RL checkpoints on a fixed seed set (D3 / 18k by default).

Why: a run writes several `.pth` (one per process) and the in-training win_rate
is not comparable across runs. Picking the best checkpoint needs a *fixed*
seed set and a single scoring rule.

Usage
-----
    python experiments/rank_checkpoints.py \
        --weights model/rl_26-11-18_16-16-18.pth model/rl_26-11-18_16-16-29_p72562.pth \
        --seeds 3038-3047 --jobs 8

    # rank everything a run wrote
    python experiments/rank_checkpoints.py --weights 'model/rl_26-11-18_19-35-*.pth'

Output: one line per weight, sorted by win rate; Wilson 95% interval; a JSON
copy next to the first weight's directory as `checkpoint_ranking.json`.

Notes
-----
* Each episode is a separate process, so a crashed game does not lose the batch.
* `EVAL_OUT` files are kept (default `outputs/rank/<tag>-s<seed>.txt`).
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import threading

DEFAULT_SEEDS = list(range(3038, 3058))

# 官方模型/我们的权重都对出生点敏感（官方 SL 只见过左上）：把已知映射写进来，
# 这样排名表能同时给出「左上 / 右下」两列，避免用混合口径下结论。
LEFT_TOP = {3038, 3040, 3045, 3046, 3051, 3053, 3054, 3055}
BASE = os.path.expanduser("~/mini-AlphaStar")


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def parse_result(path):
    """Return native outcome for one episode, or None."""
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    kv = {}
    for part in txt.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            kv[k.strip().split()[-1]] = v.strip()
    try:
        return int(float(kv["native_outcome"]))
    except Exception:
        return None


def run_one(weight, seed, out_dir, difficulty, loops, gpu, results, lock):
    tag = os.path.basename(weight).replace(".pth", "")
    out = os.path.join(out_dir, "%s-s%d.txt" % (tag, seed))
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "EVAL_MODEL": os.path.abspath(weight),
        "EVAL_DIFFICULTY": str(difficulty),
        "EVAL_MAX_LOOPS": str(loops),
        "EVAL_EPISODES": "1",
        "EVAL_SEED": str(seed),
        "EVAL_SEED_PER_EP": "1",
        "EVAL_OUT": out,
        "MAS_ON_SERVER": "1",
        "DISPLAY": env.get("DISPLAY", ":1024"),
        "SC2PATH": os.path.expanduser("~/StarCraftII"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    log = out + ".log"
    try:
        with open(log, "wb") as fh:
            subprocess.run([sys.executable, "run_rl_eval.py"], cwd=BASE, env=env,
                           stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
    except Exception as exc:  # timeout / spawn failure
        print("[rank] %s s%d failed: %s" % (tag, seed, exc), flush=True)
    res = parse_result(out)
    with lock:
        results.setdefault(weight, {})[seed] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="paths or globs (quote globs)")
    ap.add_argument("--seeds", default="3038-3057")
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--loops", type=int, default=18000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(BASE, "outputs", "rank"))
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要跑的任务数与种子范围，不开进程")
    args = ap.parse_args()

    weights = []
    for w in args.weights:
        hits = sorted(glob.glob(w)) if any(c in w for c in "*?[") else [w]
        weights.extend([h for h in hits if os.path.exists(h)])
    weights = sorted(set(weights))
    if not weights:
        print("[rank] no weights matched", file=sys.stderr)
        return 2

    seeds = parse_seeds(args.seeds)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    results, lock = {}, threading.Lock()
    tasks = [(w, s) for w in weights for s in seeds]
    if args.dry_run:
        # 2026-11-20：先看清"要跑多少局"再开跑，避免 40 个种子只跑 10 局的静默偏差
        print("[dry-run] weights=%d seeds=%d tasks=%d jobs=%d"
              % (len(weights), len(seeds), len(tasks), args.jobs))
        print("[dry-run] seeds: %d..%d" % (min(seeds), max(seeds)))
        print("[dry-run] out-dir: %s" % args.out_dir)
        return 0
    print("[rank] weights=%d seeds=%d tasks=%d jobs=%d"
          % (len(weights), len(seeds), len(tasks), args.jobs), flush=True)
    gpus = len([d for d in os.environ.get("OFF_GPUS", "0,1,2,3").split(",") if d.strip()])
    gpus = max(1, gpus)

    idx = [0]
    idx_lock = threading.Lock()

    def worker(wid):
        while True:
            with idx_lock:
                if idx[0] >= len(tasks):
                    return
                w, s = tasks[idx[0]]
                idx[0] += 1
            run_one(w, s, out_dir, args.difficulty, args.loops, wid % gpus, results, lock)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(max(1, args.jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = []
    for w in weights:
        per = results.get(w, {})
        played = {sd: v for sd, v in per.items() if v is not None}
        # 2026-11-20：分母改成"尝试过的局数"（含崩溃/无结果的局）。
        # 只统计有 outcome 的局会把胜率抬高：实测 b0s5 由 62.5%（25/40）变成 80.6%（25/31）。
        n = len(per)
        k = sum(1 for v in played.values() if v == 1)
        n_dec = len(played)
        lo, hi = wilson(k, n)
        lt = [v for sd, v in played.items() if sd in LEFT_TOP]
        rb = [v for sd, v in played.items() if sd not in LEFT_TOP]
        rows.append((k / n if n else 0.0, w, k, n, n_dec, lo, hi,
                     sum(1 for v in lt if v == 1), len(lt),
                     sum(1 for v in rb if v == 1), len(rb)))
    rows.sort(reverse=True)

    print("\n%-36s %4s %4s %4s %7s   %-14s %-8s %-8s"
          % ("weight", "win", "dec", "all", "rate", "95% CI", "left-top", "right-bot"))
    for rate, w, k, n, n_dec, lo, hi, lk, ln, rk, rn in rows:
        print("%-36s %4d %4d %4d %6.1f%%   [%4.1f, %4.1f]   %d/%-6d %d/%-6d"
              % (os.path.basename(w)[:36], k, n_dec, n, 100 * rate, lo, hi, lk, ln, rk, rn))

    payload = [{"weight": w, "wins": k, "n": n, "n_decided": n_dec, "rate": rate, "ci": [lo, hi],
                "left_top": [lk, ln], "right_bottom": [rk, rn],
                "per_seed": results.get(w, {})}
               for rate, w, k, n, n_dec, lo, hi, lk, ln, rk, rn in rows]
    dump = os.path.join(os.path.dirname(weights[0]) or ".", "checkpoint_ranking.json")
    try:
        with open(dump, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("\n[done] %s" % dump)
    except Exception as exc:
        print("\n[done] (json not written: %s)" % exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
