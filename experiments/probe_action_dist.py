#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线探查：比较不同权重在同一批数据上的 action_type 分布与置信度。

目的：验证"策略塌缩"假设——某个改动是不是让模型恒定输出少数几类动作
（离线看熵和直方图，不需要跑 SC2，几十秒出结果）。

用法：
  MAS_LSTM_HIDDEN=128 MAS_LSTM_LAYERS=1 python experiments/probe_action_dist.py <ckpt> [n_frames]
  MAS_LSTM_HIDDEN=256 MAS_LSTM_LAYERS=2 python experiments/probe_action_dist.py <ckpt>
"""

import sys
import glob
import argparse
from collections import Counter

sys.path.insert(0, ".")

import torch

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP


def load_state(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--nframes", type=int, default=64)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--file", type=int, default=-1, help="数据集文件序号，-1 表示最后一个")
    args = ap.parse_args()

    bsz = max(1, args.nframes // args.seq)
    n = bsz * args.seq

    files = sorted(glob.glob("data/replay_data_a3nc_h5/*.pt"))
    src = files[args.file]
    x = torch.load(src, map_location="cpu")
    n = min(n, x[0].shape[0])
    bsz = max(1, n // args.seq)
    n = bsz * args.seq
    feat, lab = x[0][:n], x[1][:n]
    is_final = torch.zeros(n, 1)
    batch = torch.cat([feat, lab, is_final], dim=-1).reshape(bsz, args.seq, -1)

    model = ArchModel()
    missing, unexpected = load_state(model, args.ckpt)
    model.eval()

    fs, ls = Feature.getSize(), Label.getSize()
    f3 = batch[:, :, :fs].reshape(bsz * args.seq, fs)
    l3 = batch[:, :, fs:fs + ls].reshape(bsz * args.seq, ls)

    state = Feature.feature2state(f3)
    with torch.no_grad():
        out = model.forward(state, batch_size=bsz, sequence_length=args.seq, return_logits=True)
    action_logits, action_pred = out[0], out[1]

    logits = action_logits.action_type
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values
    ent = -(probs * (probs + 1e-9).log()).sum(dim=-1)

    from alphastarmini.lib.hyper_parameters import Label_Size as LS, LabelIndex
    gt = lab.reshape(-1, ls)
    a0 = int(LabelIndex.action_type_encoding)
    gt_ids = gt[:, a0:a0 + LS[LabelIndex.action_type_encoding]].argmax(dim=-1)

    cnt = Counter(pred.tolist())
    top = cnt.most_common(5)
    uniq = len(cnt)

    print("ckpt=%s  hidden=%d layers=%d" % (args.ckpt.split("/")[-1], AHP.lstm_hidden_dim, AHP.lstm_layers))
    print("missing_keys=%d unexpected_keys=%d" % (len(missing), len(unexpected)))
    if missing[:3]:
        print("  例: missing", missing[:3])
    print("frames=%d  unique_pred_action_types=%d/%d" % (n, uniq, logits.shape[-1]))
    from pysc2.lib.actions import RAW_FUNCTIONS
    nm = lambda i: (RAW_FUNCTIONS[i].name if i < len(RAW_FUNCTIONS) else "?")
    print("pred top10:", [(nm(k), v) for k, v in cnt.most_common(10)])
    print("mean_conf=%.3f  mean_entropy=%.3f  acc=%.3f"
          % (conf.mean().item(), ent.mean().item(), (pred == gt_ids).float().mean().item()))
    gtc = Counter(gt_ids.tolist())
    print("gt   top10:", [(nm(k), v) for k, v in gtc.most_common(10)])


if __name__ == "__main__":
    main()
