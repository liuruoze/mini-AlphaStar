#!/usr/bin/env bash
# 单局探针：把指定 replay 各自转换 + 过滤到独立目录，便于逐局统计「被压制帧」
# 用法: bash experiments/replay_gen/bvb_probe.sh <replay1> [replay2 ...]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
BASE=/home/liuruoze/bvb/probe
LOG=/home/liuruoze/bvb/probe_logs
JOBS=${JOBS:-4}

mkdir -p "$BASE" "$LOG"

for f in "$@"; do
  b=$(basename "$f" .SC2Replay | sed 's/_[0-9-]\{10\}-[0-9-]\{6\}$//')
  echo "$f|$b"
done | xargs -P "$JOBS" -I{} bash -c '
  line="{}"
  f="${line%%|*}"; b="${line##*|}"
  work='"$BASE"'/work_$b; raw='"$BASE"'/raw_$b; dst='"$BASE"'/$b
  rm -rf "$work" "$raw" "$dst"; mkdir -p "$work" "$raw" "$dst"
  ln -sf "$f" "$work"/
  cd '"$ROOT"'
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII \
    MAS_MAX_STEPS=21000 MAS_ON_SERVER=0 timeout 1500 '"$PY"' run_transform.py \
      --no_server_replay_path="$work"/ --save_path_tensor="$raw"/ \
      > '"$LOG"'/tr_$b.log 2>&1
  echo "$b tr=$?"
  '"$PY"' experiments/replay_gen/_mk_nonop.py --src "$raw" --dst "$dst" > '"$LOG"'/nonop_$b.log 2>&1
  echo "$b nonop=$?"
  rm -rf "$work" "$raw"
' 2>&1 | tee "$LOG/probe_driver.log"

echo "[$(date '+%F %T')] PROBE DONE" | tee -a "$LOG/probe_driver.log"
