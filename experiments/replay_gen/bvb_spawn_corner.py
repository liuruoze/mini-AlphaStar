#!/usr/bin/env python3
"""统计每局开局时我方（alliance=1）单位的出生象限。

entity 编码（max_entities=128）：
  alliance 272:277，x 282:290（8 位 one-hot），y 290:298（8 位 one-hot）
  坐标范围 0..63，<32 = 左/上，>=32 = 右/下

用法: python experiments/replay_gen/bvb_spawn_corner.py --dir data/bvb_all30_nonop
"""
import argparse
import glob
import os

import numpy as np
import torch

SCALAR = 8020
N_ENT = 128
EMB = 1856
W = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float32)


def dec(b):
    return (b * W).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/bvb_all30_nonop")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    print("%-34s %6s %6s %6s  %s" % ("file", "x", "y", "units", "corner"))
    tally = {}
    for p in fs:
        x, _ = torch.load(p, map_location="cpu")
        e = x[:, SCALAR:SCALAR + N_ENT * EMB].reshape(x.shape[0], N_ENT, EMB).float()
        f = min(args.frame, e.shape[0] - 1)
        fr = e[f]
        real = (fr.abs().sum(1) > 1e-6)
        al = fr[:, 272:277].argmax(1)
        mine = real & (al == 1)
        if int(mine.sum()) == 0:
            print("%-34s  (no alliance=1 unit at frame %d)" % (os.path.basename(p)[:34], f))
            continue
        xs = dec(fr[mine, 282:290].numpy())
        ys = dec(fr[mine, 290:298].numpy())
        mx, my = float(xs.mean()), float(ys.mean())
        corner = ("左" if mx < 32 else "右") + ("上" if my < 32 else "下")
        tally[corner] = tally.get(corner, 0) + 1
        print("%-34s %6.1f %6.1f %6d  %s" % (os.path.basename(p)[:34], mx, my, int(mine.sum()), corner))
        del x, e
    print()
    print("象限统计:", tally)


if __name__ == "__main__":
    main()
