#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按帧序列比较两个权重"在做什么"，而不是"预测得准不准"。

动机：2026-11-20 发现 b0s5（63.8%）与 b0s6（25%）的 val acc 几乎一样（0.289 vs 0.270），
动作占比也接近（build 25% vs 30%、attack 47% vs 48%），但胜率差 2.5 倍。
静态直方图看不出差别，所以这里看**时序**：

  - switch_rate : 相邻帧预测不同的比例（越低越"黏"，越高越"抖"）
  - run_median  : 同一动作连续出现的长度中位数
  - build_per100: 每 100 帧里建造类动作的帧数
  - first_build : 第一次预测建造类动作的帧号（None = 整段没建造）
  - top1_share  : 最常见动作占比

用法:
  python experiments/probe_sequence.py <ckpt> --file 7 [--chunk 8] [--max-frames 400]
  python experiments/probe_sequence.py <ckptA> <ckptB> --file 7     # 并排比较
"""

import argparse
import glob
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from pysc2.lib.actions import RAW_FUNCTIONS


def load(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()


def scan(ckpt, data, file_idx, chunk, max_frames):
    files = sorted(glob.glob(os.path.join(data, "*.pt")))
    src = files[file_idx]
    x = torch.load(src, map_location="cpu")
    fs, ls = Feature.getSize(), Label.getSize()
    feat, lab = x[0], x[1]
    n = feat.shape[0] if max_frames <= 0 else min(max_frames, feat.shape[0])
    n = (n // chunk) * chunk

    model = ArchModel()
    load(model, ckpt)

    preds = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            f = feat[i:i + chunk].reshape(chunk, fs)
            state = Feature.feature2state(f)
            out = model.forward(state, batch_size=1, sequence_length=chunk, return_logits=True)
            # out[0] 是 logits（out[1] 是采样出的动作，不是分布）
            preds.extend(out[0].action_type.argmax(dim=-1).tolist())

    # 统计
    runs, cur, length = [], None, 0
    for p in preds:
        if p == cur:
            length += 1
        else:
            if cur is not None:
                runs.append(length)
            cur, length = p, 1
    runs.append(length)

    cnt = Counter(preds)
    top1, top1_n = cnt.most_common(1)[0]
    build_ids = [i for i in range(len(RAW_FUNCTIONS)) if RAW_FUNCTIONS[i].name.lower().startswith("build")]
    build_frames = sum(v for k, v in cnt.items() if k in build_ids)
    first_build = next((i for i, p in enumerate(preds) if p in build_ids), None)

    runs_sorted = sorted(runs)
    return {
        "file": os.path.basename(src),
        "frames": len(preds),
        "switch_rate": 1.0 - (len(runs) / max(len(preds), 1)),
        "run_median": runs_sorted[len(runs_sorted) // 2],
        "build_per100": 100.0 * build_frames / max(len(preds), 1),
        "first_build": first_build,
        "top1": RAW_FUNCTIONS[top1].name if top1 < len(RAW_FUNCTIONS) else str(top1),
        "top1_share": 100.0 * top1_n / max(len(preds), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "replay_data_a3nc_h5"))
    ap.add_argument("--file", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=400, help="0 = 整个文件")
    args = ap.parse_args()

    rows = []
    for c in args.ckpts:
        r = scan(c, args.data, args.file, args.chunk, args.max_frames)
        r["name"] = os.path.basename(c).replace(".pth", "")
        rows.append(r)

    print("%-38s %6s %7s %7s %8s %10s %s" %
          ("ckpt", "frames", "switch", "run_med", "build/100", "first_build", "top1"))
    for r in rows:
        print("%-38s %6d %6.2f %7d %8.1f %10s %s %.0f%%" %
              (r["name"][:38], r["frames"], r["switch_rate"], r["run_median"],
               r["build_per100"], str(r["first_build"]), r["top1"], r["top1_share"]))


if __name__ == "__main__":
    main()
