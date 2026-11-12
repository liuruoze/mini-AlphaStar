#!/usr/bin/env python3
"""量化 AI 决策的重复率：相邻帧的动作是否完全相同。

标签分段（max_entities=128 布局，Label.getSize()=6454）：
    0:564      action_type (564)
    564:572    delay (8)
    572:574    queued (2)
    574:694    其它 (120)
    694:2230   selected_units (12 x 128)
    2230:2358  target_unit (128)
    2358:6454  target_location (64x64)

用法: python experiments/replay_gen/bvb_action_dup.py --n 6
"""
import argparse
import glob
import os

import numpy as np
import torch

A_AT = (0, 564)
A_SEL = (694, 2230)
A_TU = (2230, 2358)
A_TL = (2358, 6454)


def sigs(y):
    at = y[:, A_AT[0]:A_AT[1]].argmax(1)
    sel = y[:, A_SEL[0]:A_SEL[1]].reshape(-1, 12, 128).argmax(2)
    tu = y[:, A_TU[0]:A_TU[1]].argmax(1)
    tl = y[:, A_TL[0]:A_TL[1]].argmax(1)
    full = torch.cat([at.unsqueeze(1), sel, tu.unsqueeze(1), tl.unsqueeze(1)], 1)
    coarse = torch.stack([at, tu, tl], 1)
    return full, coarse


def run_lengths(flags):
    """flags[i] = 与前一行相同；返回连续相同段的长度分布"""
    runs = []
    cur = 0
    for f in flags:
        if f:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--dir", default="data/bvb_all30_nonop")
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    step = max(1, len(fs) // args.n)
    pick = fs[::step][:args.n]

    print("%-34s %6s %8s %8s %9s %9s" %
          ("file", "frames", "dup_full", "dup_coarse", "uniq_full", "max_run"))
    agg = []
    for p in pick:
        x, y = torch.load(p, map_location="cpu")
        full, coarse = sigs(y)
        eq_full = (full[1:] == full[:-1]).all(1)
        eq_coarse = (coarse[1:] == coarse[:-1]).all(1)
        n = y.shape[0]
        nf = int(eq_full.sum())
        nc = int(eq_coarse.sum())
        uniq = len({tuple(r.tolist()) for r in full})
        runs = run_lengths(eq_full.tolist())
        print("%-34s %6d %7.1f%% %7.1f%% %9d %9d" %
              (os.path.basename(p)[:34], n, 100.0 * nf / max(n - 1, 1),
               100.0 * nc / max(n - 1, 1), uniq, max(runs) if runs else 0))
        agg.append((n, nf, nc, uniq))
        del x, y, full, coarse
    if agg:
        tn = sum(a[0] for a in agg)
        print()
        print("合计 frames=%d  相邻重复(full)=%.1f%%  (coarse)=%.1f%%  平均唯一签名占比=%.1f%%"
              % (tn, 100.0 * sum(a[1] for a in agg) / tn,
                 100.0 * sum(a[2] for a in agg) / tn,
                 100.0 * sum(a[3] for a in agg) / tn))


if __name__ == "__main__":
    main()
