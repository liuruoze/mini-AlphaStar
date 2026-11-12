#!/usr/bin/env bash
# 批量转换 + nonop 过滤（每局独立进程，转完立即过滤并删中间文件）
# 用法: bash experiments/replay_gen/bvb_transform.sh [并发数]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
SRC=/home/liuruoze/bvb/replays
WORK=/home/liuruoze/bvb/work
RAW=/home/liuruoze/bvb/raw
DST=$ROOT/data/bvb_50_nonop
LOG=/home/liuruoze/bvb/logs
JOBS=${1:-8}

mkdir -p "$WORK" "$RAW" "$DST" "$LOG"
echo "[$(date '+%F %T')] START transform jobs=$JOBS" | tee -a "$LOG/transform_driver.log"

ls "$SRC"/*.SC2Replay 2>/dev/null | xargs -P "$JOBS" -I{} bash -c "
  f={}
  b=\$(basename \$f .SC2Replay)
  rm -rf $WORK/\$b $RAW/\$b
  mkdir -p $WORK/\$b $RAW/\$b
  ln -sf \$f $WORK/\$b/
  cd $ROOT
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII \
    MAS_MAX_STEPS=21000 MAS_ON_SERVER=0 \
    timeout 1200 $PY run_transform.py \
      --no_server_replay_path=$WORK/\$b/ \
      --save_path_tensor=$RAW/\$b/ \
      > $LOG/transform_\$b.log 2>&1
  trc=\$?
  $PY experiments/replay_gen/_mk_nonop.py --src $RAW/\$b --dst $DST > $LOG/nonop_\$b.log 2>&1
  nrc=\$?
  rm -rf $WORK/\$b $RAW/\$b
  echo \"\$b tr=\$trc nonop=\$nrc\" >> $LOG/transform_driver.log
"

N_PT=$(ls "$DST"/*.pt 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] DONE files=$N_PT" | tee -a "$LOG/transform_driver.log"
du -sh "$DST" | tee -a "$LOG/transform_driver.log"
