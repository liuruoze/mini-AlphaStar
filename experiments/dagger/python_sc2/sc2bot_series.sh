#!/bin/bash
# 一台机器上「一局一进程」地跑 N 局，绕开 python-sc2 0.8.4 在同进程内
# 连续开局的旧版本兼容问题（第二局起 SC2 常报 unable to parse frame）。
#
# 用法： bash sc2bot_series.sh <bot_file> <bot_class> <games> <minutes> <out_root> [difficulty] [enemy]
set -u

BOT_FILE=${1:?bot file}
BOT_CLASS=${2:?bot class}
GAMES=${3:-5}
MINUTES=${4:-12}
OUT=${5:-$HOME/bvb/sc2bot_replays/series}
DIFF=${6:-very_hard}
ENEMY=${7:-protoss}

BIN=$HOME/venv_sc2_084/bin/python
RUN=$HOME/mini-AlphaStar/experiments/dagger/python_sc2/sc2bot_run.py
LOG=$HOME/bvb/matrix_logs/sc2bot_series_$(basename "$OUT").log

mkdir -p "$OUT" "$(dirname "$LOG")"
: > "$LOG"

for i in $(seq 1 "$GAMES"); do
  echo "=== round $i ===" >> "$LOG"
  env DISPLAY=:1024 SC2PATH="$HOME/StarCraftII" \
    "$BIN" -u "$RUN" \
      --bot-file "$BOT_FILE" --bot-class "$BOT_CLASS" \
      --games 1 --minutes "$MINUTES" \
      --difficulty "$DIFF" --enemy "$ENEMY" \
      --out "$OUT/g$i" >> "$LOG" 2>&1
  sleep 4
done

echo "=== tally ===" >> "$LOG"
grep -c "^\[result\] Victory" "$LOG" | sed 's/^/wins=/' >> "$LOG"
grep -c "^\[result\] Defeat" "$LOG" | sed 's/^/losses=/' >> "$LOG"
grep -c "^\[result\] Tie" "$LOG" | sed 's/^/ties=/' >> "$LOG"
echo "SERIES DONE" >> "$LOG"
