#!/usr/bin/env bash
# m30 HPO 训练入口（所有新增开关显式列出，默认关闭项不写进历史脚本）
# 用法: GPU=0 bash experiments/sl/train_m30_hpo.sh <tag> <epochs> [seed] [data_dir]
set -u
cd /home/liuruoze/mini-AlphaStar || exit 2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
ulimit -n 65536

TAG=${1:?tag}
EPOCHS=${2:?epochs}
SEED=${3:-20261108}
DATA=${4:-/home/liuruoze/mini-AlphaStar/data/bvb_m30_sub}
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python

mkdir -p "model/sl/$TAG" log
echo "[$(date '+%F %T')] HPO TRAIN START tag=$TAG epochs=$EPOCHS seed=$SEED gpu=${GPU:-0} data=$DATA"

export MAS_ON_SERVER=1
export MAS_SL_SEED="$SEED"
export MAS_SL_EPOCHS="$EPOCHS"
export MAS_SL_LR=3e-4
export MAS_SL_WARMUP=300
export MAS_SL_SCHED=cosine
export MAS_SL_ETA_MIN=3e-5
export MAS_SL_BATCH=48
export MAS_SL_SEQ_LEN=8
export MAS_SL_CLASS_WEIGHTS='build=2.5,train=2.5,research=2.5,move=0.5'
export MAS_SL_CLASS_WEIGHTS_VERBOSE="${MAS_SL_CLASS_WEIGHTS_VERBOSE:-0}"
export MAS_SL_FRAME_SPLIT=0
export MAS_SL_VAL_NUM=6
export MAS_SL_WEIGHT_DECAY=1e-5
export MAS_SL_WORKERS=0
export MAS_DAGGER="${MAS_DAGGER:-0}"
export MAS_SL_MODEL_PATH="./model/sl/$TAG/"

"$PY" run_sl.py -p1 "$DATA/" > "log/train_${TAG}.log" 2>&1
rc=$?

echo "[$(date '+%F %T')] HPO TRAIN END tag=$TAG rc=$rc"
grep -E "SL-config|SL-manifest|SL-classw|SL-warmup|frame-split" "log/train_${TAG}.log" | head -n 6
grep -o "Batch/Epoch: \[[0-9]*/[0-9]*\]" "log/train_${TAG}.log" | tail -n 1
exit $rc
