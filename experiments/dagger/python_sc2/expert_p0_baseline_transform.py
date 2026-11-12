#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-1a: freeze the CURRENT (index0) transform baseline for ONE replay.

Runs the real project transform (transform_replay_data.test) but overrides
output paths so nothing lands in data/replay_data_tensor_new_small/.

Writes: <out>/Simple64_....pt  and  baseline_index0.json with
  file md5 / size / N / features sha256 / labels sha256 / action_type argmax dist.

No production file is modified.
"""
import hashlib
import json
import os
import random
import sys

M = os.path.expanduser("~/mini-AlphaStar")
sys.path.insert(0, M)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

REPLAY_DIR = os.environ.get("P0_REPLAY_DIR",
                            os.path.expanduser("~/bvb/p0_baseline/replays"))
OUT_DIR = os.environ.get("P0_OUT_DIR",
                         os.path.expanduser("~/bvb/p0_baseline/pt_index0"))
RESULT = os.environ.get("P0_RESULT",
                        os.path.expanduser("~/bvb/p0_baseline/baseline_index0.json"))

import param as P  # noqa: E402

P.on_server = True
P.replay_path = REPLAY_DIR.rstrip("/") + "/"

from alphastarmini.core.sl import transform_replay_data as TRD  # noqa: E402


def stat_pt(path):
    import torch
    import numpy as np
    feat, lab = torch.load(path, map_location="cpu")
    fn = feat.numpy()
    ln = lab.numpy()
    # action_type segment is label[:, 0:564]
    at = ln[:, :564].argmax(axis=1)
    dist = {int(k): int(v) for k, v in zip(*np.unique(at, return_counts=True))}
    return {
        "pt_path": path,
        "pt_md5": hashlib.md5(open(path, "rb").read()).hexdigest(),
        "pt_bytes": os.path.getsize(path),
        "n_frames": int(ln.shape[0]),
        "label_dim": int(ln.shape[1]),
        "feature_dim": int(fn.shape[1]),
        "features_sha256": hashlib.sha256(fn.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(ln.tobytes()).hexdigest(),
        "action_type_argmax_dist": dist,
        "action_type_argmax_top10": sorted(dist.items(), key=lambda x: -x[1])[:10],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # transform_replay_data.py has no random.seed of its own; it calls
    # random.uniform() 6 times per action step to apply keep-thresholds.
    # Without a seed the same replay gives a different .pt every run, so any
    # equivalence check MUST pin this.
    seed_raw = os.environ.get("P0_SEED", "")
    if seed_raw != "":
        random.seed(int(seed_raw))
        print("[p0-1a] random.seed(%s) applied" % seed_raw, flush=True)
    else:
        print("[p0-1a] WARNING: no P0_SEED -> transform is NON-deterministic",
              flush=True)
    TRD.FLAGS.save_path_tensor = OUT_DIR.rstrip("/") + "/"
    TRD.FLAGS.save_path = OUT_DIR.rstrip("/") + "/"
    TRD.DATA_FROM = 0
    TRD.DATA_NUM = 1
    TRD.FLAGS.max_replays = 1
    print("[p0-1a] replay_dir=%s out=%s" % (REPLAY_DIR, OUT_DIR), flush=True)
    print("[p0-1a] replays=%s" % sorted(os.listdir(REPLAY_DIR)), flush=True)
    TRD.test(on_server=True)
    pts = [f for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".pt")]
    if not pts:
        raise SystemExit("[p0-1a] FAIL: no .pt produced in %s" % OUT_DIR)
    st = stat_pt(os.path.join(OUT_DIR, pts[0]))
    out = {"mode": "index0 (current code, no MAS_MULTI_ACTION)",
           "replay_dir": REPLAY_DIR, "stats": st}
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2)
    print("[p0-1a] BASELINE %s" % json.dumps(st)[:600], flush=True)


if __name__ == "__main__":
    main()
