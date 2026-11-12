#!/usr/bin/env bash
# 批量采集 Bot vs Bot 对局（P hard vs T medium，保证 Protoss 胜 → 观察者=Protoss）
# 用法: bash experiments/replay_gen/bvb_collect.sh [局数] [并发数]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
OUT=/home/liuruoze/bvb/replays
LOG=/home/liuruoze/bvb/logs
N=${1:-50}
JOBS=${2:-8}

mkdir -p "$OUT" "$LOG"
echo "[$(date '+%F %T')] START collect N=$N jobs=$JOBS"
: > "$LOG/collect_driver.log"

seq 1 "$N" | xargs -P "$JOBS" -I{} bash -c "
  cd $ROOT
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII timeout 200 \
    $PY experiments/replay_gen/_bot_vs_bot.py \
      --diff-p hard --diff-t medium \
      --max-loops 45000 --chunk 128 \
      --out-dir $OUT --tag bvb{} \
      > $LOG/collect_{}.log 2>&1
  rc=\$?
  echo \"game {} rc=\$rc\" >> $LOG/collect_driver.log
"

N_REP=$(ls "$OUT"/*.SC2Replay 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] DONE replays=$N_REP" | tee -a "$LOG/collect_driver.log"
