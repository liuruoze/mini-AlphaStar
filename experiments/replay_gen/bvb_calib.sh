#!/usr/bin/env bash
# 难度标定：不同 P/T 难度组合的胜负与时长，用于挑「均势且 Protoss 能赢」的采集配置
# 用法: bash experiments/replay_gen/bvb_calib.sh [每配置局数] [并发]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
OUT=/home/liuruoze/bvb/calib
LOG=/home/liuruoze/bvb/calib_logs
PER=${1:-5}
JOBS=${2:-8}

mkdir -p "$OUT" "$LOG"
echo "[$(date '+%F %T')] CALIB START per=$PER jobs=$JOBS"

# 配置表：标签:protoss难度:terran难度
CONFIGS=(
  "mm:medium:medium"
  "vh_h:veryhard:hard"
  "m_h:medium:hard"
)

TASKS=$(mktemp)
for cfg in "${CONFIGS[@]}"; do
  tag="${cfg%%:*}"; rest="${cfg#*:}"; dp="${rest%%:*}"; dt="${rest##*:}"
  for i in $(seq 1 "$PER"); do
    echo "$tag $dp $dt $i" >> "$TASKS"
  done
done

cat "$TASKS" | xargs -P "$JOBS" -L 1 bash -c '
  tag=$0; dp=$1; dt=$2; i=$3
  cd '"$ROOT"'
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII timeout 300 \
    '"$PY"' experiments/replay_gen/_bot_vs_bot.py \
      --diff-p "$dp" --diff-t "$dt" \
      --max-loops 45000 --chunk 128 \
      --out-dir '"$OUT"' --tag "${tag}_${i}" \
      > '"$LOG"'/calib_${tag}_${i}.log 2>&1
  echo "$tag $i rc=$?"
' >> "$LOG/calib_driver.log"

rm -f "$TASKS"
echo "[$(date '+%F %T')] CALIB DONE replays=$(ls "$OUT"/*.SC2Replay 2>/dev/null | wc -l)" | tee -a "$LOG/calib_driver.log"
