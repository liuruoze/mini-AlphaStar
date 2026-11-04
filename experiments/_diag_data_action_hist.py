"""第 82 轮（DS，#40 前置诊断）：dump 训练数据的**全量**动作类型直方图。

动机（DS12 82.4）：实测我们与官方模型的实战动作分布差异巨大
  官方：Build_Pylon 29%、Train_Probe 19-23%、Build_Gateway 10-16%、Harvest_Gather 5-10%、Train_Stalker 少量
  我们：no_op 36-38%、Train_Zealot 21%、Build_Pylon 7%、Train_Probe 6%、Harvest_Gather 0
要判断这是"数据里就没有"还是"模型没学到"，必须先看数据侧的分布。

用法：
  PYTHONPATH=. python experiments/_diag_data_action_hist.py --n-files 3
"""
from __future__ import print_function

import argparse
import glob
import os

import torch

from alphastarmini.lib.hyper_parameters import Label_Size as LS
from pysc2.lib.actions import RAW_FUNCTIONS as F

NAMES = {
    0: "no_op", 2: "Attack_pt", 12: "Smart_unit", 13: "Move_pt",
    35: "Build_Pylon_pt", 36: "Build_Assimilator_pt", 37: "Build_Gateway_pt",
    38: "Build_CyberneticsCore_pt", 39: "Build_Stargate_pt",
    47: "Train_Zealot", 49: "Train_Zealot_quick", 50: "Train_Stalker",
    51: "Train_Stalker_quick", 52: "Train_Sentry", 53: "Train_Sentry_quick",
    64: "Train_Probe_quick", 102: "Harvest_Gather_unit",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/replay_data_v3")
    ap.add_argument("--n-files", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.path, "*.pt")))[:args.n_files]
    total = {}
    n_frames = 0
    for fp in files:
        try:
            feat, lab = torch.load(fp)
        except Exception as e:
            print("[skip] %s: %r" % (fp, e))
            continue
        n = lab.shape[0]
        n_frames += n
        at = lab[:, :LS.action_type_encoding]
        # one-hot -> id
        ids = at.argmax(dim=1)
        cnt = torch.bincount(ids.long(), minlength=LS.action_type_encoding)
        for i, c in enumerate(cnt.tolist()):
            if c:
                total[i] = total.get(i, 0) + c
        print("[done] %-42s frames=%d" % (os.path.basename(fp), n), flush=True)

    print("\n=== 数据侧动作分布（%d 文件 / %d 帧）===" % (len(files), n_frames))
    tot = sum(total.values())
    for i, c in sorted(total.items(), key=lambda x: -x[1])[:24]:
        print("  %-26s id=%-4d %7d  %6.2f%%" % (NAMES.get(i, "?"), i, c, 100.0 * c / tot))
    print("\n合计动作数: %d" % tot)

    # 与官方实战分布的对照（官方 2 局的全量直方图，n=2625/局）
    print("\n=== 关键动作对照 ===")
    key = [0, 2, 12, 35, 37, 49, 50, 64, 102]
    print("%-26s %10s %10s" % ("动作", "数据占比", "官方实战"))
    off = {35: 0.298, 64: 0.211, 37: 0.129, 2: 0.095, 49: 0.150, 102: 0.075, 36: 0.020, 50: 0.014, 0: 0.0}
    for i in key:
        print("%-26s %9.2f%% %9.2f%%" % (NAMES.get(i, "?"), 100.0 * total.get(i, 0) / tot,
                                         100.0 * off.get(i, float("nan"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
