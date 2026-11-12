#!/usr/bin/env bash
# 单权重评测（固定 seed 3038，15 分钟局）
# 用法: bash experiments/sl/eval_m30.sh <ckpt.pth> <tag> [gpu] [steps]
set -u
CKPT=${1:?ckpt}
TAG=${2:?tag}
GPU=${3:-0}
STEPS=${4:-21600}          # 21,600 loops ≈ 16 min（15 min 目标留余量）；step_mul=8

D="/home/liuruoze/mini-AlphaStar/model/eval_$TAG"
rm -rf "$D"; mkdir -p "$D"
cp "$CKPT" "$D/sl_eval.pth"        # 目录里只留一个含 "sl" 的权重，避免 utils 取到别的
echo "[$(date '+%F %T')] EVAL START tag=$TAG gpu=$GPU steps=$STEPS ckpt=$(basename "$CKPT")"
cd "$HOME/mini-AlphaStar-off"
OFF_GAME_STEPS="$STEPS" OFF_GPUS="$GPU" \
  bash run_eval_batch.sh "$D" "$TAG" 3038 1 1
echo "[$(date '+%F %T')] EVAL END tag=$TAG"
tail -3 "outputs/${TAG}_3038.txt" 2>/dev/null
