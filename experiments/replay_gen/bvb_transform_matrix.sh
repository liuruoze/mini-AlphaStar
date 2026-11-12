#!/usr/bin/env bash
# 矩阵(10x3)批量转换：transform -> nonop -> 动作抽稀(move/attack 20%)
# 观察者强制为 Protoss（我方），输赢都保留（方案 A）
# 用法: bash experiments/replay_gen/bvb_transform_matrix.sh <replay_dir> <dst_dir> [jobs]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
SRC=${1:?need replay dir}
DST=${2:?need dst dir}
JOBS=${3:-10}
BASE=/home/liuruoze/bvb/mtx_run
LOG=/home/liuruoze/bvb/matrix_logs

mkdir -p "$BASE" "$DST" "$LOG" "$LOG/tr"
echo "[$(date '+%F %T')] START matrix transform n=$(ls "$SRC"/*.SC2Replay | wc -l) jobs=$JOBS dst=$DST" \
  | tee -a "$LOG/transform_matrix_driver.log"

ls "$SRC"/*.SC2Replay | xargs -P "$JOBS" -I{} bash -c '
  f={}
  b=$(basename "$f" .SC2Replay)
  W='"$BASE"'/$b
  rm -rf "$W"; mkdir -p "$W/work" "$W/raw" "$W/nonop" "$W/sub"
  ln -sf "$f" "$W/work/"
  cd '"$ROOT"'
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII \
    MAS_MAX_STEPS=21000 MAS_ON_SERVER=0 MAS_REPLAY_OBSERVER=proto \
    timeout 3000 '"$PY"' run_transform.py \
      --no_server_replay_path="$W/work/" \
      --save_path_tensor="$W/raw/" \
      > '"$LOG"'/tr/tr_$b.log 2>&1
  trc=$?
  '"$PY"' experiments/replay_gen/_mk_nonop.py --src "$W/raw" --dst "$W/nonop" > "$W/nonop.log" 2>&1
  nrc=$?
  '"$PY"' experiments/_research/_subsample_actions.py --src "$W/nonop" --dst "$W/sub" \
      --frac 0.2 --actions 13,3 --seed $(( $(printf %s "$b" | cksum | cut -d" " -f1) % 100000000 + 1 )) > "$W/sub.log" 2>&1
  src=$?
  obs=$(grep -m1 "forced observer(proto)" '"$LOG"'/tr/tr_$b.log | sed "s/.*= //")
  fin=$(grep -m1 "^\[sub\] total" "$W/sub.log")
  cp -f "$W"/sub/*.pt '"$DST"'/ 2>/dev/null
  nout=$(ls "$W"/sub/*.pt 2>/dev/null | wc -l)
  rm -rf "$W"
  echo "$b tr=$trc nonop=$nrc sub=$src out=$nout obs=[$obs] $fin" >> '"$LOG"'/transform_matrix_driver.log
'

echo "[$(date '+%F %T')] DONE files=$(ls "$DST"/*.pt 2>/dev/null | wc -l)" \
  | tee -a "$LOG/transform_matrix_driver.log"
du -sh "$DST" | tee -a "$LOG/transform_matrix_driver.log"
