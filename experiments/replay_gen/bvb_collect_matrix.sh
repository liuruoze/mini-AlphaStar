#!/usr/bin/env bash
# 10×3 难度矩阵采集：Protoss 难度 1..10 × Terran 难度 1..3，共 30 局。
# 每局要求我方出生在左上（LT），否则弃局换种子重试。
# 用法: bash experiments/replay_gen/bvb_collect_matrix.sh [并发] [每个组合最多尝试次数]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
OUT=/home/liuruoze/bvb/matrix
LOG=/home/liuruoze/bvb/matrix_logs
JOBS=${1:-6}
MAXTRY=${2:-10}

mkdir -p "$OUT" "$LOG"

P_DIFFS="veryeasy easy medium mediumhard hard harder veryhard cheatvision cheatmoney cheatinsane"
T_DIFFS="veryeasy easy medium"

TASKS=$(mktemp)
for pd in $P_DIFFS; do
  for td in $T_DIFFS; do
    echo "$pd $td" >> "$TASKS"
  done
done
echo "[$(date '+%F %T')] MATRIX START combos=$(wc -l < "$TASKS") jobs=$JOBS" | tee "$LOG/driver.log"

cat "$TASKS" | xargs -P "$JOBS" -L 1 bash -c '
  pd=$0; td=$1
  tag="p${pd}_t${td}"
  cd '"$ROOT"'
  ok=0
  for k in $(seq 1 '"$MAXTRY"'); do
    seed=$(( (RANDOM << 15 ^ RANDOM) % 2000000000 + 1 ))
    DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII timeout 400 \
      '"$PY"' experiments/replay_gen/_bot_vs_bot.py \
        --diff-p "$pd" --diff-t "$td" \
        --corner LT --seed "$seed" \
        --max-loops 45000 --chunk 128 \
        --out-dir '"$OUT"' --tag "${tag}_try${k}" \
        > '"$LOG"'/collect_${tag}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=1; break; fi
    if [ $rc -ne 42 ]; then
      echo "$tag try=$k rc=$rc (非出生角问题，停止重试)" >> '"$LOG"'/driver.log
      break
    fi
    rm -f '"$OUT"'/${tag}_try${k}_*.SC2Replay
  done
  if [ $ok -eq 1 ]; then
    echo "$tag OK (try=$k)" >> '"$LOG"'/driver.log
  else
    echo "$tag FAILED" >> '"$LOG"'/driver.log
  fi
'

rm -f "$TASKS"
echo "[$(date '+%F %T')] MATRIX DONE replays=$(ls "$OUT"/*.SC2Replay 2>/dev/null | wc -l)" | tee -a "$LOG/driver.log"
