#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 79 轮（DS）诊断：replay_data_v3 里 action_type 的类别分布。

目的：验证 78.19 的假设 —— 行为克隆的类别不平衡（采集/移动占绝大多数，
交战稀疏），以及 Simple64 上 `Smart_unit -> Harvest_Gather_unit` 的强制改写在
数据里到底造成多大影响。

输出：
  1. top-20 动作及其占比
  2. 关键动作各自的占比：no_op / Attack_pt / Smart_unit / Harvest_Gather_unit / Move*
  3. 交战类动作（名字含 Attack / Effect_*Attack）的合计占比

用法：
  python experiments/_diag_action_dist.py [--n-files 3]
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

from pysc2.lib.actions import RAW_FUNCTIONS as F
from alphastarmini.lib.hyper_parameters import Label_Size as LS


def id2name():
    """action id -> 动作名。

    ⚠️ one-hot 的列号**就是 pysc2 RAW_FUNCTIONS 的 id**（已实测核对：
    no_op=0, Attack_pt=2, Smart_unit=12, Move_pt=13, Harvest_Gather_unit=102），
    不是 alphastar_available_actions.py 里 FUNCTION_LIST 的列表下标
    （那里把 0/1 两项注释掉了，用下标会整体错位）。
    """
    m = {}
    for name in F._func_dict:          # pysc2 内部字典：name -> Function
        fn = F._func_dict[name]
        m[int(fn.id.value)] = name
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="./data/replay_data_v3/")
    ap.add_argument("--n-files", type=int, default=3, help="统计前 N 个文件（按名字排序）")
    args = ap.parse_args()

    names = id2name()
    files = sorted(f for f in os.listdir(args.path) if f.endswith(".pt"))[:args.n_files]
    print("统计 %d 个文件: %s" % (len(files), files))
    print("action_type 编码维度 LS.action_type_encoding =", LS.action_type_encoding)

    cnt = collections.Counter()
    total = 0
    for fn in files:
        feat, lab = torch.load(os.path.join(args.path, fn), map_location="cpu")
        # lab: [T, 6454]，前 action_type_encoding 维是 one-hot
        at = lab[:, :LS.action_type_encoding]
        idx = at.argmax(dim=-1)
        # 同时确认 one-hot 的"有效位"数量（全 0 表示该帧无动作标签）
        valid = at.sum(dim=-1) > 0.5
        cnt.update(idx[valid].tolist())
        total += int(valid.sum())
        print("  %-44s frames=%d 有效标签=%d (%.1f%%)"
              % (fn, lab.shape[0], int(valid.sum()),
                 100.0 * int(valid.sum()) / max(lab.shape[0], 1)))

    print("\n=== 有效动作标签总数: %d ===" % total)
    print("\n--- top 20 ---")
    for aid, c in cnt.most_common(20):
        print("  %-34s id=%-4d %8d  %6.2f%%" % (names.get(aid, "?"), aid, c, 100.0 * c / total))

    print("\n--- 关键动作 ---")
    key = {"no_op": 0, "Attack_pt": 2, "Smart_unit": 12,
           "Move_pt": 13, "Harvest_Gather_unit": 102}
    for nm, aid in key.items():
        c = cnt.get(aid, 0)
        print("  %-24s id=%-4d %8d  %6.2f%%" % (nm, aid, c, 100.0 * c / total))

    attack_like = sum(c for aid, c in cnt.items()
                      if "Attack" in names.get(aid, ""))
    print("  %-24s %8d  %6.2f%%" % ("**含 Attack 的全部动作**", attack_like,
                                    100.0 * attack_like / total))

    print("\n--- 按 goal 归类的粗分布（看 combat 类总量）---")
    goal_cnt = collections.Counter()
    for aid, c in cnt.items():
        nm = names.get(aid, "")
        if "Attack" in nm or "Effect_" in nm:
            g = "combat(攻击/技能)"
        elif "Harvest" in nm or "Smart" in nm or "Move" in nm or "Gather" in nm:
            g = "move/economy(移动/采集)"
        elif nm.startswith("Build") or nm.startswith("Train"):
            g = "build/train(建造/造兵)"
        else:
            g = "other(含 no_op)"
        goal_cnt[g] += c
    for g, c in goal_cnt.most_common():
        print("  %-26s %8d  %6.2f%%" % (g, c, 100.0 * c / total))


if __name__ == "__main__":
    main()
