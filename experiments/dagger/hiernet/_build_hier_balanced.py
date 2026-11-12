"""生成「动作配额均衡」版专家数据聚合数据集（下一轮 3 run 的可复跑材料）。

夜间 S3 的专家帧分布被重复下单污染：
    Build_Assimilator_unit 1115 (40.6%) + Train_Probe_quick 922 (33.5%) = 74.1%
    而 Build_Gateway 110 (4.0%) / Build_Cyber 43 (1.6%) / Train_Zealot 283 (10.3%)
⇒ 建造/产兵示范太少，模型学不到战局行为。

本脚本只做**配额下采样**（不复制、不上采样、不改标签），把两个过多项压到
上限内，其余动作全部保留；再与 30 局 AI 原帧物理复制组成独立数据集。
**本轮不训练**，只产出数据 + 门禁报告，供下一轮 3 run 验证「仅修分布」的效果。
"""
import collections
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
from pysc2.lib import actions as A

ROOT = "/home/liuruoze/mini-AlphaStar"
SRC_EXPERT = os.path.join(ROOT, "data/hier_night_nonop")
SRC_AI = os.path.join(ROOT, "data/bvb_m30_sub")
DST = os.path.join(ROOT, "data/replay_data_m30_dagger_hier_balanced")
QUOTA = {                       # 占「原始专家帧总数」的最大比例
    "Build_Assimilator_unit": 0.08,
    "Train_Probe_quick": 0.20,
}
SEED = 3038
CHUNK = 200

NAMES = {i: f.name for i, f in enumerate(A.RAW_FUNCTIONS)}


def main():
    rng = np.random.RandomState(SEED)
    fs = sorted(glob.glob(os.path.join(SRC_EXPERT, "*.pt")))
    buckets = collections.defaultdict(list)
    total = 0
    for p in fs:
        x, y = torch.load(p, map_location="cpu")
        acts = y[:, :564].argmax(1).tolist()
        for i, a in enumerate(acts):
            buckets[NAMES.get(a, str(a))].append((x[i], y[i]))
            total += 1
    print("[bal] src expert frames=%d actions=%d" % (total, len(buckets)), flush=True)

    before = {k: len(v) for k, v in buckets.items()}
    kept = []
    after = {}
    for name, rows in buckets.items():
        cap = int(total * QUOTA[name]) if name in QUOTA else len(rows)
        if len(rows) > cap:
            idx = rng.choice(len(rows), size=cap, replace=False)
            rows = [rows[int(i)] for i in idx]
        after[name] = len(rows)
        kept.extend(rows)
    rng.shuffle(kept)
    print("[bal] after frames=%d" % len(kept), flush=True)
    print("[bal] before:", dict(sorted(before.items(), key=lambda t: -t[1])[:10]), flush=True)
    print("[bal] after :", dict(sorted(after.items(), key=lambda t: -t[1])[:10]), flush=True)

    # 门禁：过多项已压到配额内，且建造+产兵占比 >= 30%
    build_train = sum(v for k, v in after.items()
                      if k.startswith(("Build_", "Train_")) and k not in QUOTA)
    ratio = build_train / max(1, len(kept))
    ok = (after.get("Build_Assimilator_unit", 0) <= int(total * 0.08) + 1
          and after.get("Train_Probe_quick", 0) <= int(total * 0.20) + 1
          and ratio >= 0.30)
    print("[bal] gate: quota_ok=%s other_build_train_ratio=%.3f PASS=%s"
          % (True, ratio, ok), flush=True)
    if not ok:
        return 2

    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    ai = sorted(glob.glob(os.path.join(SRC_AI, "*.pt")))
    for p in ai:
        shutil.copy2(p, os.path.join(DST, os.path.basename(p)))

    n_file = 0
    for s in range(0, len(kept), CHUNK):
        part = kept[s:s + CHUNK]
        xb = torch.stack([r[0] for r in part]).contiguous().float()
        yb = torch.stack([r[1] for r in part]).contiguous().float()
        torch.save((xb, yb), os.path.join(DST, "expert_bal_%03d.pt" % n_file))
        n_file += 1

    ai_frames = sum(torch.load(p, map_location="cpu")[0].shape[0] for p in ai)
    man = {
        "dataset": "m30_dagger_hier_balanced", "dagger": True,
        "expert": "HierNet-SC2 lv10-0.94 (exec) + v3 raw exec",
        "expert_frame_policy": "quota_downsample_only (no duplication, no upsampling)",
        "quota": QUOTA, "seed": SEED,
        "source_nodagger": "bvb_m30_sub", "orig_files": len(ai),
        "orig_frames": int(ai_frames),
        "expert_files": n_file, "expert_frames": len(kept),
        "expert_action_before": before, "expert_action_after": after,
        "gate_other_build_train_ratio": round(ratio, 4),
        "layout": "128",
        "trained": False,
        "note": "仅修专家帧动作分布；未训练，待下一轮 3 run 验证",
    }
    json.dump(man, open(os.path.join(DST, "manifest.json"), "w",
                        encoding="utf-8"), ensure_ascii=False,
              indent=2, sort_keys=True)
    print("[bal] DST=%s ai_frames=%d expert_frames=%d" %
          (DST, ai_frames, len(kept)), flush=True)
    print("[bal] PASS (data only, NOT trained)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
