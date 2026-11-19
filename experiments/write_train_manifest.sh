#!/usr/bin/env bash
# 为一次训练写可复现清单：experiments/write_train_manifest.sh <tag> <epochs> <seed> [sstep]
# 由 train_bvb_m30.sh 在启动 python 前调用；也可单独运行补写历史 run。
set -u
TAG=${1:?tag}; EP=${2:?epochs}; SEED=${3:-20261108}; SSTEP=${4:-3}
M=/home/liuruoze/mini-AlphaStar
OUT="$M/model/sl/$TAG/train_manifest.json"
mkdir -p "$M/model/sl/$TAG"

DATA=${DATA:-$M/data/bvb_m30_sub}
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python

"$PY" - "$OUT" "$TAG" "$EP" "$SEED" "$SSTEP" "$DATA" << 'PY'
import hashlib
import json
import os
import subprocess
import sys
import time

out, tag, ep, seed, sstep, data = sys.argv[1:7]
M = "/home/liuruoze/mini-AlphaStar"

def md5(p):
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None

def env(k, d=None):
    return os.environ.get(k, d)

# 数据集指纹：manifest + 文件数 + 总帧数
ds = {"path": data}
try:
    n = len([x for x in os.listdir(data) if x.endswith(".pt")])
    ds["files"] = n
except Exception:
    ds["files"] = None
mpath = os.path.join(data, "manifest.json")
if os.path.exists(mpath):
    try:
        m = json.load(open(mpath))
        for k in ("dagger", "dtype", "frames", "frames_converted", "note", "source"):
            if k in m:
                ds[k] = m[k]
    except Exception:
        pass
    ds["manifest_md5"] = md5(mpath)

rec = {
    "tag": tag,
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "host": os.uname()[1],
    "epochs": int(ep),
    "seed": int(seed),
    "step_size": int(sstep),
    "dataset": ds,
    "optim": {
        "lr": env("MAS_SL_LR", "1e-4"),
        "gamma": env("MAS_SL_GAMMA", "0.2"),
        "val_frac": env("MAS_SL_VAL_FRAC", "0.2"),
        "split_seed": env("MAS_SL_SPLIT_SEED", "20261108"),
        "workers": env("MAS_SL_WORKERS", "0"),
        "batch_size_env": env("MAS_SL_BATCH", None),
    },
    "knobs": {
        # 2026-11-19：架构开关必须进 manifest，否则"这份权重是什么结构"无法追溯
        "lstm_ln": env("MAS_LSTM_LN", "off"),
        "lstm_hidden": env("MAS_LSTM_HIDDEN", "128"),
        "lstm_layers": env("MAS_LSTM_LAYERS", "1"),
        "ent_weight": env("MAS_SL_ENT_WEIGHT", "0"),
        "attack_weight": env("MAS_SL_ATTACK_WEIGHT", "1.0"),
        "class_weights": env("MAS_SL_CLASS_WEIGHTS", ""),
        "early_w": env("MAS_SL_EARLY_W", "1.0"),
        "early_frac": env("MAS_SL_EARLY_FRAC", "0.35"),
        "ent_w": env("MAS_SL_ENT_W", "0.0"),
        "label_smoothing": env("MAS_SL_LS", "0.0"),
        "balance": env("MAS_SL_BALANCE", "0"),
        "balance_maxw": env("MAS_SL_BALANCE_MAXW", None),
        "restore": env("MAS_SL_RESTORE_PATH", ""),
        "load_state": env("MAS_SL_LOAD_STATE", "0"),
        "dagger": env("MAS_DAGGER", "0"),
    },
    "code": {
        "trainer_md5": md5(os.path.join(M, "alphastarmini/core/sl/sl_train_by_tensor.py")),
        "loss_md5": md5(os.path.join(M, "alphastarmini/core/sl/sl_loss_multi_gpu.py")),
    },
}

# 不进 git，但能对齐"是不是同一份代码"
try:
    rec["code"]["git_head"] = subprocess.check_output(
        ["git", "-C", M, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    rec["code"]["git_head"] = None

json.dump(rec, open(out, "w"), ensure_ascii=False, indent=2)
print("[manifest] %s" % out)
PY
