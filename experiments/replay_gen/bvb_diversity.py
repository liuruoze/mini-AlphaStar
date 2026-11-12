#!/usr/bin/env python3
"""量化「打法多样性」：每局动作类型分布的两两余弦相似度 + 阈值聚类数。

--subsample N：每局随机抽 N 个动作再建直方图，
消除「每局动作数不同（人类 ~57 vs AI ~2230）导致直方图噪声不同」的混杂。

用法:
  python experiments/replay_gen/bvb_diversity.py --dir data/bvb_all30_nonop --name ai30 --subsample 57
"""
import argparse
import glob
import os

import numpy as np
import torch


def one(p, subsample=0, rng=None):
    x, y = torch.load(p, map_location="cpu")
    at = y[:, :564].argmax(1).numpy()
    if subsample and at.size > subsample:
        at = rng.choice(at, size=subsample, replace=False)
    h = np.bincount(at, minlength=564).astype(np.float64)
    h /= max(h.sum(), 1)
    n = int(y.shape[0])
    del x, y
    return h, n


def n_clusters(C, thr):
    """单链接聚类：相似度 > thr 视为同簇"""
    M = C.shape[0]
    parent = list(range(M))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(M):
        for j in range(i + 1, M):
            if C[i, j] > thr:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(M)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", default="ds")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    fs = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    H, N = [], []
    for p in fs:
        h, n = one(p, args.subsample, rng)
        H.append(h)
        N.append(n)
    H = np.stack(H)
    M = len(fs)
    tag = "%s%s" % (args.name, ("(sub%d)" % args.subsample) if args.subsample else "")
    print("=== %s：%d 局，%d 帧 ===" % (tag, M, sum(N)))

    C = H @ H.T
    iu = np.triu_indices(M, 1)
    pair = C[iu]
    print("两两余弦: 均值=%.3f 中位=%.3f max=%.3f" % (pair.mean(), np.median(pair), pair.max()))
    np.fill_diagonal(C, 0.0)
    near = C.max(1)
    print("每局最近邻相似度: 中位=%.3f" % np.median(near))
    print("聚类数（相似度阈值）: " + "  ".join(
        ">%.1f → %d 簇" % (t, n_clusters(C, t)) for t in (0.4, 0.5, 0.6, 0.7)))


if __name__ == "__main__":
    main()
