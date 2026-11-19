#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计数据集里 action_type 标签的分布（不加载模型，纯读张量）。

用途：确认标签是否被单一动作（No_Op）主导——如果是，SL 的"准确率"就主要在对多数类计分，
策略塌缩到少数动作也能拿到不低的分数，这解释了"离线指标好、上线崩"。
"""

import os
import os
import sys
import glob
import argparse
from collections import Counter

# 允许从任意工作目录运行：以脚本所在目录的上一级作为仓库根
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from alphastarmini.lib.hyper_parameters import Label_Size as LS, LabelIndex
from pysc2.lib.actions import RAW_FUNCTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=10)
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "replay_data_a3nc_h5"))
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    if args.files > 0:
        files = files[:args.files]

    a0 = int(LabelIndex.action_type_encoding)
    n_at = LS[LabelIndex.action_type_encoding]
    cnt = Counter()
    total = 0
    per_file = []

    for f in files:
        x = torch.load(f, map_location="cpu")
        lab = x[1]
        if args.stride > 1:
            lab = lab[::args.stride]
        ids = lab[:, a0:a0 + n_at].argmax(dim=-1)
        c = Counter(ids.tolist())
        cnt.update(c)
        total += ids.numel()
        per_file.append((f.split("/")[-1], ids.numel(), len(c)))

    print("files=%d frames=%d unique_action_types=%d" % (len(files), total, len(cnt)))
    print("%-6s %-8s %-7s %s" % ("id", "count", "share", "name"))
    for k, v in cnt.most_common(10):
        name = RAW_FUNCTIONS[k].name if k < len(RAW_FUNCTIONS) else "?"
        print("%-6d %-8d %-6.2f%% %s" % (k, v, 100.0 * v / total, name))

    head = sum(v for _, v in cnt.most_common(3))
    print("top3 占比 %.2f%%" % (100.0 * head / total))
    print("每文件: " + ", ".join("%s:%d帧/%d类" % (n[:18], t, u) for n, t, u in per_file[:5]))


if __name__ == "__main__":
    main()
