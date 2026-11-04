#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E0（第 76 轮，GLM，改动 #21）：native 胜率评测编排器（Kimi-17 的可执行部分）。

**它解决什么**

此前"策略好不好"没有任何可执行答案：
`win_rate_list` 在 RL 训练里永远为空（统计间隔 > 局数），
`rl_eval_sl.test()` 写死 1 局、难度 1、固定种子、共享输出文件。

**设计**

- 复用 `rl_eval_sl` 的整套对局机制（与训练同一套动作转换，避免"评测路径 ≠ 训练路径"）；
- 只把 4 个参数环境变量化（局数/难度/种子/输出文件），并支持 `EVAL_MODEL` 指定 checkpoint；
- 编排器起 K 个分片进程（`CUDA_VISIBLE_DEVICES` 轮转分卡、种子各不同），
  每片打 `--games-per-shard` 局；结束后汇总 `results: [loss, draw, win]`，
  给出 native 胜率与 **Wilson 95% 置信区间**。

**判读纪律**：只报 native 口径（SC2 原生胜负）；平局不计入胜率分子（另列）。
样本 20 局时，95% 区间宽度约 ±20 个百分点 —— 所以这个数字只能回答
"是 0% / 很低 / 中等 / 很高"这种量级问题，回答不了"差 5 个点"。

跑法：
    python experiments/_eval_winrate.py --model model/sl_26-11-02_13-58-18.pth \
        --games-per-shard 5 --shards 4 --tag sl_base
"""
from __future__ import print_function

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    """Wilson 分数区间。n=0 时返回 (0,0) 并由调用方跳过。"""
    if n == 0:
        return 0.0, 0.0
    p = k / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="要评测的 .pth 绝对/相对路径")
    ap.add_argument("--games-per-shard", type=int, default=5)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--difficulty", type=int, default=1, help="1=very_easy ... 9=cheat_insane")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--gpus", default="0,1,2,3")
    args = ap.parse_args()

    os.chdir(ROOT)                       # run_rl_eval 用 ./model ./outputs 相对路径
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    procs = []
    t0 = time.time()
    for i in range(args.shards):
        env = os.environ.copy()
        env.update({
            # ---- 第 80 轮（DS，#32）：修 BLAS 线程超订 ----
            # 实测：未设这些变量时，**每个**评测进程起 131 个线程、吃 901% CPU（9 核）。
            # 模型只有 1.1 GB，多线程对微张量纯属同步开销；3 分片时看不出来（核多），
            # 8 分片时 8x9=72 核互相抢，单局从 311 s 退化到 >700 s（并发反而更慢）。
            # 这与 MEMORY.md 里"入口脚本须在 import numpy 前设 1"是同一条坑，
            # 之前只在训练入口修过，评测入口一直漏着。
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)],
            "EVAL_MODEL": args.model,
            "EVAL_EPISODES": str(args.games_per_shard),
            "EVAL_DIFFICULTY": str(args.difficulty),
            "EVAL_SEED": str(args.seed_base + i),
            "EVAL_OUT": "./outputs/eval_%s_shard%d.txt" % (args.tag, i),
            "SC2PATH": os.path.expanduser("~/StarCraftII"),
        })
        logf = open("log/eval_%s_shard%d.log" % (args.tag, i), "w")
        p = subprocess.Popen([sys.executable, "run_rl_eval.py"],
                             env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
        procs.append((i, p, logf))
        print("[shard %d] gpu=%s seed=%s pid=%d -> log/eval_%s_shard%d.log"
              % (i, env["CUDA_VISIBLE_DEVICES"], env["EVAL_SEED"], p.pid, args.tag, i),
              flush=True)

    r = re.compile(r"results:\s*\[(\d+),\s*(\d+),\s*(\d+)\]")
    totals = [0, 0, 0]          # [loss, draw, win]，与 rl_eval_sl 的 results 含义一致
    for i, p, logf in procs:
        p.wait()
        logf.close()
        txt = open("log/eval_%s_shard%d.log" % (args.tag, i), errors="ignore").read()
        ms = r.findall(txt)
        if ms:
            a, b, c = (int(x) for x in ms[-1])
            totals[0] += a
            totals[1] += b
            totals[2] += c
        else:
            print("[shard %d] ⚠️ 没找到 results 行（可能中途崩溃），看分片日志" % i, flush=True)

    loss, draw, win = totals
    n = loss + draw + win
    lo, hi = wilson(win, n)
    summary = {
        "tag": args.tag,
        "model": args.model,
        "difficulty": args.difficulty,
        "games": n,
        "native_win": win,
        "native_draw": draw,
        "native_loss": loss,
        "win_rate": (win / float(n)) if n else 0.0,
        "wilson95": [lo, hi],
        "wall_s": time.time() - t0,
    }
    out_json = os.path.join("log", "eval_%s.json" % args.tag)
    with open(out_json, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 胜率汇总（native 口径，难度 %d）=====" % args.difficulty, flush=True)
    print(" checkpoint : %s" % args.model, flush=True)
    print(" 局数       : %d（%d 分片 × %d 局）" % (n, args.shards, args.games_per_shard), flush=True)
    print(" 胜/平/负   : %d / %d / %d" % (win, draw, loss), flush=True)
    if n:
        print(" 胜率       : %.1f%%   Wilson95%%: [%.1f%%, %.1f%%]"
              % (100.0 * win / n, 100.0 * lo, 100.0 * hi), flush=True)
    print(" 明细 JSON  : %s" % out_json, flush=True)
    print(" 用时       : %.0f s" % summary["wall_s"], flush=True)
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
