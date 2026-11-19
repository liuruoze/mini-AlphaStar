#!/usr/bin/env bash
# 种子扫射：批量训练 N 个 seed，再逐个评测，最后汇总。
#
# 背景（2026-11-20）：同一套代码/数据，7 个 seed 的胜率是
#   12.5 / 21.9 / 25.0 / 27.5 / 32.5 / 40.0 / 63.8（%）
# 即约 1/7 的概率抽到"能打"的种子。既然单跑无法判断改动好坏，
# 更务实的做法是：改动 → 扫 6–10 个 seed → 看分布，而不是看单点。
#
# 用法：
#   bash experiments/seed_sweep.sh --seeds 20261201,20261202,20261203 [--epochs 4] [--games 40] [--prefix sw]
#   bash experiments/seed_sweep.sh --seeds 20261201-20261206 --dry-run
#
# 资源：训练最多 2 个并行（内存闸门），评测 8 并行；总时长 ≈ ceil(N/2)*45 + N*30 分钟。

set -u

M="$HOME/mini-AlphaStar"
DATA="${DATA:-$HOME/mini-AlphaStar/data/replay_data_a3nc_h5}"
OUT_ROOT="${OUT_ROOT:-$HOME/bvb/arch_ab}"
EPOCHS=4
GAMES=40
PREFIX="sw"
DRY=0
WEIGHTS="train_probe=0.15,build_pylon=8,build_gateway=8,build_nexus=0.1,train_zealot=4"
SEEDS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --games) GAMES="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --weights) WEIGHTS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "未知参数 $1"; exit 2;;
  esac
done

[ -z "$SEEDS" ] && { echo "必须给 --seeds，例如 20261201,20261202 或 20261201-20261206"; exit 2; }

# 展开 a-b 区间
if echo "$SEEDS" | grep -q -- "-"; then
  lo="${SEEDS%-*}"; hi="${SEEDS#*-}"
  SEED_LIST=$(seq "$lo" "$hi")
else
  SEED_LIST=$(echo "$SEEDS" | tr "," " ")
fi
N=$(echo "$SEED_LIST" | wc -w)

echo "== 计划 =="
echo "seeds   : $(echo $SEED_LIST)"
echo "runs    : $N（prefix=$PREFIX, epochs=$EPOCHS, games=$GAMES）"
echo "data    : $DATA"
echo "预计时长: $(( (N+1)/2 * 45 + N * 30 )) 分钟"
[ "$DRY" = "1" ] && { echo "[dry-run] 不执行"; exit 0; }

train_one() {
  local seed="$1" gpu="$2" tag="$3"
  env GPU="$gpu" MAS_DAGGER=1 DATA="$DATA" MAS_SL_LR=1e-3 MAS_SL_CLASS_WEIGHTS="$WEIGHTS" \
    bash "$M/experiments/train_bvb_m30.sh" "$tag" "$EPOCHS" "$seed" 3 > "$M/log_night/$tag.log" 2>&1
}

# 两两并行训练
slot=0
for s in $SEED_LIST; do
  tag="${PREFIX}${s}"
  if [ -f "$M/model/sl/$tag/train_manifest.json" ]; then
    echo "[skip] $tag 已存在"
    continue
  fi
  gpu=$(( slot % 2 == 0 ? 2 : 3 ))
  echo "[train] $tag (gpu=$gpu)"
  train_one "$s" "$gpu" "$tag" &
  slot=$((slot + 1))
  if [ $((slot % 2)) -eq 0 ]; then wait; fi
done
wait

# 逐个评测
for s in $SEED_LIST; do
  tag="${PREFIX}${s}"
  ckpt=$(ls "$M/model/sl/$tag"/*_best.pth 2>/dev/null | head -1)
  [ -z "$ckpt" ] && { echo "[warn] $tag 没有权重，跳过"; continue; }
  echo "[eval] $tag -> $OUT_ROOT/${tag}_out"
  ( cd "$M" && ~/.conda/envs/th_1_7/bin/python -u experiments/rank_checkpoints.py \
      --weights "$ckpt" --seeds "3038-$((3038 + GAMES - 1))" --jobs 8 \
      --out-dir "$OUT_ROOT/${tag}_out" > "$OUT_ROOT/${tag}_eval.log" 2>&1 )
done

echo "== 汇总 =="
cd "$M" && ~/.conda/envs/th_1_7/bin/python experiments/summarize_runs.py \
  $(for s in $SEED_LIST; do printf "%s=%s/${PREFIX}%s_out " "$s" "$OUT_ROOT" "$s"; done)
