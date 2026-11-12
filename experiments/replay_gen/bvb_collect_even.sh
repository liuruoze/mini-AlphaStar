#!/usr/bin/env bash
# 采集「均势」Bot vs Bot 对局：可配置难度表，用于补被压制/残局状态
# 用法: bash experiments/replay_gen/bvb_collect_even.sh
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
OUT=/home/liuruoze/bvb/even
LOG=/home/liuruoze/bvb/even_logs
JOBS=${JOBS:-8}

mkdir -p "$OUT" "$LOG"

# 配置表：标签:protoss难度:terran难度:局数
# vh_h = P veryhard vs T hard：标定 P 胜 3/5，均势且 P 能赢（观察者才会是 Protoss）
# mm   = P medium vs T medium：P 胜 1/5，但赢得那几局是「逆风翻盘」，逆风帧最密集
CONFIGS=(
  "vh:veryhard:hard:24"
  "mm:medium:medium:24"
)

TASKS=$(mktemp)
for cfg in "${CONFIGS[@]}"; do
  tag="${cfg%%:*}"; rest="${cfg#*:}"
  dp="${rest%%:*}"; rest="${rest#*:}"
  dt="${rest%%:*}"; n="${rest##*:}"
  for i in $(seq 1 "$n"); do
    echo "$tag $dp $dt $i" >> "$TASKS"
  done
done
echo "[$(date '+%F %T')] EVEN COLLECT START tasks=$(wc -l < "$TASKS") jobs=$JOBS" | tee "$LOG/collect_driver.log"

cat "$TASKS" | xargs -P "$JOBS" -L 1 bash -c '
  tag=$0; dp=$1; dt=$2; i=$3
  cd '"$ROOT"'
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII timeout 400 \
    '"$PY"' experiments/replay_gen/_bot_vs_bot.py \
      --diff-p "$dp" --diff-t "$dt" \
      --max-loops 45000 --chunk 128 \
      --out-dir '"$OUT"' --tag "${tag}${i}" \
      > '"$LOG"'/collect_${tag}${i}.log 2>&1
  echo "$tag$i rc=$?" >> '"$LOG"'/collect_driver.log
'

rm -f "$TASKS"
echo "[$(date '+%F %T')] EVEN COLLECT DONE replays=$(ls "$OUT"/*.SC2Replay 2>/dev/null | wc -l)" | tee -a "$LOG/collect_driver.log"
