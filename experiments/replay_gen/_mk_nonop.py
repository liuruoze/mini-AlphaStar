# -*- coding: utf-8 -*-
"""把 v3 风格的 replay 张量过滤成 v3-nonop 风格（去掉 no_op 帧）。

规则（2026-11-06 逆向并验证）：
    保留 `argmax(label[:, 0:564]) != 0` 的行 —— 即 action_type 不是 no_op 的帧。
    验证：对 data/replay_data_v3 的 60 个文件逐文件比对，保留行数与
          data/replay_data_v3_nonop 完全一致（3,554 = 3,554，0 个文件不一致）。

用法：
    python experiments/replay_gen/_mk_nonop.py --src data/replay_data_ext70 --dst data/replay_data_ext70_nonop
"""
import argparse
import glob
import os

import torch

ACTION_DIM = 564


def filter_one(src_path, dst_path):
    feats, labels = torch.load(src_path, map_location="cpu")
    keep = labels[:, 0:ACTION_DIM].argmax(dim=1) != 0
    n_in = int(labels.shape[0])
    n_keep = int(keep.sum())
    if n_keep == 0:
        return n_in, 0, None
    torch.save((feats[keep].contiguous(), labels[keep].contiguous()), dst_path)
    return n_in, n_keep, dst_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-keep", type=int, default=1,
                    help="保留帧数低于该值的文件不落盘（默认 1，即空文件丢弃）")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.pt")))
    tot_in = tot_keep = n_files = n_empty = 0
    for p in files:
        dst = os.path.join(args.dst, os.path.basename(p))
        n_in, n_keep, _ = filter_one(p, dst)
        tot_in += n_in
        tot_keep += n_keep
        if n_keep < args.min_keep:
            if os.path.exists(dst):
                os.remove(dst)
            n_empty += 1
        else:
            n_files += 1
    ratio = (tot_keep / tot_in * 100) if tot_in else 0.0
    print(f"[NONOP] src={args.src}")
    print(f"[NONOP] files_in={len(files)} files_out={n_files} dropped={n_empty}")
    print(f"[NONOP] frames_in={tot_in} frames_keep={tot_keep} keep_rate={ratio:.2f}%")
    print(f"[NONOP] dst={args.dst}")


if __name__ == "__main__":
    main()
