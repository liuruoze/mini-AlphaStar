#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按标签频率生成逆频率类权重表（JSON），供 MAS_SL_CLASS_WEIGHTS_JSON 使用。

动机：手写权重（build_pylon=8 等）是拍脑袋的，且只覆盖少数动作。
这里用 w_c = (1/freq_c)^alpha 归一化到均值 1，alpha=0.5 相当于"平方根逆频率"，
既压制多数类，又不会把稀有动作抬到失控。

用法:
  python experiments/make_class_weights.py --data data/replay_data_a3nc_h5 --alpha 0.5 \
      --out data/a3nc_h5_classw.json [--max-files N]
"""

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from alphastarmini.lib.hyper_parameters import Label_Size as LS, LabelIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "replay_data_a3nc_h5"))
    ap.add_argument("--alpha", type=float, default=0.5, help="0=不加权，1=完全逆频率")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=0, help="0 表示全部文件")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        ap.error("没有任何 .pt 文件：%s" % args.data)

    a0 = int(LabelIndex.action_type_encoding)
    n_cls = LS[LabelIndex.action_type_encoding]
    counts = torch.zeros(n_cls, dtype=torch.float64)

    total = 0
    for f in files:
        x = torch.load(f, map_location="cpu")
        ids = x[1][:, a0:a0 + n_cls].argmax(dim=-1)
        counts += torch.bincount(ids, minlength=n_cls).double()
        total += ids.numel()

    present = counts > 0
    freq = counts[present] / total
    w = (1.0 / freq) ** args.alpha
    # 归一化到"最常见类 = 1.0"：这样权重表读起来是"稀有类被放大多少倍"，
    # 也避免按均值归一化时把常见类压到 0.1（那等于整体缩小学习信号）。
    w = w / w.min()

    table = {str(int(i)): round(float(v), 4) for i, v in zip(torch.nonzero(present).flatten().tolist(), w.tolist())}
    meta = {"data": args.data, "files": len(files), "frames": total,
            "alpha": args.alpha, "classes_present": int(present.sum())}
    json.dump({"meta": meta, "weights": table}, open(args.out, "w"), ensure_ascii=False, indent=2)

    top = sorted(table.items(), key=lambda kv: -kv[1])[:5]
    print("%d files, %d frames, %d classes present" % (len(files), total, present.sum()))
    print("weight range %.3f–%.3f；最大的 5 类: %s" % (w.min(), w.max(), top))
    print("written -> %s" % args.out)


if __name__ == "__main__":
    main()
