#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把专家自主打的 nonop/*.pt 汇总成纯专家数据集，并删除 Cancel_Last 帧。

用法:
  python build_expert_ds.py --src-glob "$HOME/bvb/onpolicy/e1_*/nonop/*.pt" \
                            --dst "$HOME/mini-AlphaStar/data/replay_data_e1" \
                            --tag e1
- 每帧 action_type = argmax(lab[:, :LS.action_type_encoding])，其值即 pysc2 RAW_FUNCTIONS id
- 删除所有名字里含 "Cancel_Last" 的动作帧（与 a3nc 口径一致）
- 若某文件无需删帧，直接软链；需要删帧才写新文件
"""
import argparse
import collections
import glob
import json
import os
import sys

import torch
from pysc2.lib.actions import RAW_FUNCTIONS as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphastarmini.lib.hyper_parameters import Label_Size as LS


def name2id():
    m = {}
    for name in F._func_dict:
        fn = F._func_dict[name]
        m[name] = int(fn.id.value)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-glob", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tag", default="e1")
    args = ap.parse_args()

    n2i = name2id()
    cancel_ids = {v for k, v in n2i.items() if "Cancel_Last" in k}
    print("[ds] cancel ids:", sorted(cancel_ids))

    files = sorted(glob.glob(args.src_glob))
    print("[ds] src files:", len(files))
    if not files:
        raise SystemExit("[ds] FAIL no src")

    os.makedirs(args.dst, exist_ok=True)
    for old in glob.glob(os.path.join(args.dst, "*.pt")):
        if os.path.islink(old):
            os.unlink(old)

    tot_raw = tot_kept = 0
    kept_names = []
    per_game = []
    for p in files:
        feat, lab = torch.load(p, map_location="cpu")
        at = lab[:, :LS.action_type_encoding]
        idx = at.argmax(dim=-1)
        mask = torch.ones(lab.shape[0], dtype=torch.bool)
        for cid in cancel_ids:
            mask &= (idx != cid)
        n_raw = int(lab.shape[0])
        n_keep = int(mask.sum())
        tot_raw += n_raw
        tot_kept += n_keep
        dst = os.path.join(args.dst, "expert_" + os.path.basename(p))
        if n_keep == n_raw:
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(os.path.abspath(p), dst)
        else:
            torch.save((feat[mask], lab[mask]), dst)
        kept_names.append(dst)
        per_game.append({"file": os.path.basename(p), "frames": n_raw, "kept": n_keep})
        print("  %-52s %5d -> %5d" % (os.path.basename(p), n_raw, n_keep))

    json.dump({"dataset": args.tag, "note": "纯专家自主作战（learner 全 no-op + 专家 replace），已去 Cancel_Last",
               "expert_files": len(files), "expert_frames_raw": tot_raw,
               "expert_frames": tot_kept, "dagger": True, "per_game": per_game},
              open(os.path.join(args.dst, "manifest.json"), "w"), ensure_ascii=False, indent=2)
    print("[ds] files=%d frames %d -> %d (drop %.2f%%)"
          % (len(files), tot_raw, tot_kept, 100.0 * (tot_raw - tot_kept) / max(tot_raw, 1)))


if __name__ == "__main__":
    main()
