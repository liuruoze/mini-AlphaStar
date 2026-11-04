"""多版本平均（白话版：把几次"模拟考"的参数平均成一个模型）。

原名叫 model soup（权重汤），直译不好懂，这里统一叫"多版本平均"。

动机（来自本轮实测）：
  · ep02（30 rep-epochs）：killed 1,338（官方 112%）、collected 5,986、唯一有胜局；
  · ep03（45 rep-epochs）：collected 8,120、但 killed 788、0 胜。
两者是同一次训练（同 init）的**相邻 epoch**，线性模式连通性通常成立，
50/50 插值有机会同时拿到"会打"和"会运营"。

用法：
  python experiments/_avg_models.py --a model/A.pth --b model/B.pth --alpha 0.5 --out model/soup.pth
"""
from __future__ import print_function

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--more", default="",
                    help="额外的模型路径，逗号分隔（与 --a/--b 一起做等权平均）")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="两模型时：out = (1-alpha)*a + alpha*b；多模型时忽略（等权）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [args.a] + ([args.b] if args.b else [])
    if args.more:
        paths += [x.strip() for x in args.more.split(",") if x.strip()]

    if len(paths) > 2:
        # 等权平均 N 个模型
        ms = [torch.load(p_, map_location="cpu") for p_ in paths]
        out, n_f, n_i = {}, 0, 0
        for k in ms[0].keys():
            vals = [m[k] for m in ms if k in m]
            if any(v.shape != vals[0].shape for v in vals):
                out[k] = vals[0]
                continue
            if vals[0].is_floating_point():
                acc = vals[0].clone()
                for v in vals[1:]:
                    acc += v
                out[k] = acc / float(len(vals))
                n_f += 1
            else:
                out[k] = vals[0]
                n_i += 1
        torch.save(out, args.out)
        print("[平均] N=%d 等权  浮点键 %d / 整型键 %d -> %s"
              % (len(paths), n_f, n_i, args.out))
        return

    a = torch.load(paths[0], map_location="cpu")
    b = torch.load(paths[1], map_location="cpu")

    ka, kb = set(a.keys()), set(b.keys())
    if ka != kb:
        print("[平均] 键不一致：仅 A 有 %d，仅 B 有 %d；按交集处理"
              % (len(ka - kb), len(kb - ka)))

    out, n_f, n_i = {}, 0, 0
    for k in a.keys():
        if k not in b:
            out[k] = a[k]
            continue
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            print("[平均] 形状不一致，跳过：%s %s vs %s" % (k, tuple(ta.shape), tuple(tb.shape)))
            out[k] = ta
            continue
        if ta.is_floating_point():
            out[k] = (1.0 - args.alpha) * ta + args.alpha * tb
            n_f += 1
        else:
            # 整型（如 num_batches_tracked）取 A
            out[k] = ta
            n_i += 1

    torch.save(out, args.out)
    print("[平均] alpha=%.2f  浮点键 %d / 整型键 %d -> %s" % (args.alpha, n_f, n_i, args.out))


if __name__ == "__main__":
    main()
