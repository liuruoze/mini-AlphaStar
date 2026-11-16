#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 run 总览：把所有 run 的 val、配置、是否已评测、胜率汇总成一张表。

解决的问题：今天为了找"哪些 run 的 val≤0.35、哪些还没评测"翻了十几次日志。

用法：
  python3 experiments/runs_overview.py                 # 最近 30 个 run，按 val 升序
  python3 experiments/runs_overview.py --n 60 --all    # 全部
  python3 experiments/runs_overview.py --only-good     # 只列 val<=0.55（可用候选）
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

M = os.path.expanduser("~/mini-AlphaStar")
SL = os.path.join(M, "model/sl")
OUT = os.path.expanduser("~/mini-AlphaStar-off/outputs")
PY = sys.executable

VAL_RE = re.compile(r"best val_action_acc=([0-9.]+) at epoch (\d+)")


def read_val(tag):
    log = os.path.join(M, "log", "train_%s.log" % tag)
    if not os.path.exists(log):
        return None, None
    # 口径：用"最终 val"（`[SL-best] 最终：best val_action_acc=X`）。
    # 注意 val_action_acc 是**越大越好**，best 是各 epoch 里的最大值；
    # 不能用 min（第一版就写错了：把 cwB_s20261113 的 0.192(ep1) 当成最终值，而它其实 0.703 崩了）。
    final_v, final_ep = None, None
    try:
        txt = open(log, "rb").read().decode("utf-8", "ignore")
        m = re.search(r"最终：best val_action_acc=([0-9.]+) at epoch (\d+)", txt)
        if m:
            final_v, final_ep = float(m.group(1)), int(m.group(2))
        else:
            ms = VAL_RE.findall(txt)
            if ms:
                # 没有最终行（训练中断）：取最后一条 SL-best
                final_v, final_ep = float(ms[-1][0]), int(ms[-1][1])
    except Exception:
        pass
    return final_v, final_ep


def read_manifest(tag):
    p = os.path.join(SL, tag, "train_manifest.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except Exception:
        return {}


def count_eval(tag, seeds):
    """统计已评测的 seed 数。除标准命名外，也认 `tag_xxx` 这种别名前缀
    （今天同一个模型用过 cwH5_s20261171_e / h5v / h5v3 三套 tag）。"""
    n = 0
    for s in seeds:
        cands = ["%ss%d_%d.txt" % (tag, s, s), "%s_%d.txt" % (tag, s)]
        # 别名：<tag>_<后缀><seed>_<seed>.txt
        cands += [os.path.basename(p) for p in glob.glob(os.path.join(OUT, "%s_*%d_%d.txt" % (tag, s, s)))]
        # 直接调 run_eval_batch（N=1）的产物是 <tag>_<别名>_<seed>.txt
        cands += [os.path.basename(p) for p in glob.glob(os.path.join(OUT, "%s_*_%d.txt" % (tag, s)))]
        for c in cands:
            p = c if os.path.isabs(c) else os.path.join(OUT, c)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                n += 1
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only-good", action="store_true", help="只列 val<=0.55 的 run")
    ap.add_argument("--seeds", default="3038-3077")
    args = ap.parse_args()

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))

    rows = []
    for d in glob.glob(os.path.join(SL, "*")):
        tag = os.path.basename(d)
        if not os.path.isdir(d) or tag.startswith("_"):
            continue
        has_best = bool(glob.glob(os.path.join(d, "*_best.pth")))
        if not has_best:
            continue
        val, ep = read_val(tag)
        man = read_manifest(tag)
        rows.append({
            "tag": tag, "val": val, "ep": ep,
            "lr": man.get("optim", {}).get("lr", "?"),
            "ep_cfg": man.get("epochs", "?"),
            "data": os.path.basename(man.get("dataset", {}).get("path", "") or ""),
            "n_eval": count_eval(tag, seeds),
        })

    rows.sort(key=lambda r: (r["val"] is None, r["val"] if r["val"] is not None else 9))
    if not args.all:
        rows = rows[:args.n]
    if args.only_good:
        rows = [r for r in rows if r["val"] is not None and r["val"] <= 0.55]

    print("%-24s %-8s %-5s %-8s %-4s %-26s %s" % ("tag", "best_val", "ep", "lr", "ep_cfg", "data", "已评测"))
    print("-" * 92)
    for r in rows:
        v = "%.4f" % r["val"] if r["val"] is not None else "?"
        print("%-24s %-8s %-5s %-8s %-4s %-26s %d/%d%s"
              % (r["tag"], v, r["ep"] or "?", r["lr"], r["ep_cfg"] or "-", r["data"][:26] or "-",
                 r["n_eval"], len(seeds), "" if r["n_eval"] else "   <-- 未评测"))
    print("\n提示：val<=0.35 是最优带；>=0.60 基本必崩（同配方 7/7 命中）。"
          "已评测数用 `eval_table.py --seeds %s <tag>` 看胜率。" % args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
