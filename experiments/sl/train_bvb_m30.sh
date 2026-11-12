#!/usr/bin/env bash
# 训练矩阵(10x3)抽稀数据集：1 个 run
# 用法: bash experiments/sl/train_bvb_m30.sh <tag> <epochs> [seed] [step_size]
set -u
cd /home/liuruoze/mini-AlphaStar
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII
ulimit -n 65536
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TAG=${1:?tag}
EPOCHS=${2:?epochs}
SEED=${3:-20261108}
SSTEP=${4:-3}
DATA=${DATA:-/home/liuruoze/mini-AlphaStar/data/bvb_m30_sub}

mkdir -p "model/sl/$TAG"
echo "[$(date '+%F %T')] TRAIN START tag=$TAG epochs=$EPOCHS seed=$SEED ss=$SSTEP gpu=${GPU:-0} data=$DATA"

MAS_ON_SERVER=1 \
MAS_SL_SEED="$SEED" \
MAS_SL_EPOCHS="$EPOCHS" \
MAS_SL_STEP_SIZE="$SSTEP" \
MAS_SL_LR=1e-4 \
MAS_SL_GAMMA=0.2 \
MAS_SL_FRAME_SPLIT=1 \
MAS_SL_VAL_FRAC=0.2 \
MAS_SL_SPLIT_SEED=20261108 \
MAS_SL_WORKERS=0 \
MAS_SL_MODEL_PATH="./model/sl/$TAG/" \
/home/liuruoze/.conda/envs/th_1_7/bin/python run_sl.py -p1 "$DATA/" \
  > "log/train_$TAG.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] TRAIN END tag=$TAG rc=$rc"
ls -l "model/sl/$TAG/" | tail -3
