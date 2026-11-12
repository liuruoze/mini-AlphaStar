#!/usr/bin/env bash
# 按清单批量转换 + nonop 过滤到同一目录
# 用法: bash experiments/replay_gen/bvb_transform_sel.sh <清单文件> <输出目录> [并发]
set -u

ROOT=/home/liuruoze/mini-AlphaStar
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python
LIST=${1:?need list file}
DST=${2:?need dst dir}
JOBS=${3:-8}
WORK=/home/liuruoze/bvb/work
RAW=/home/liuruoze/bvb/raw
LOG=/home/liuruoze/bvb/logs

mkdir -p "$WORK" "$RAW" "$DST" "$LOG"
echo "[$(date '+%F %T')] START transform_sel n=$(grep -c . "$LIST") jobs=$JOBS dst=$DST" | tee -a "$LOG/transform_sel_driver.log"

cat "$LIST" | xargs -P "$JOBS" -I{} bash -c "
  f={}
  b=\$(basename \$f .SC2Replay)
  rm -rf $WORK/\$b $RAW/\$b
  mkdir -p $WORK/\$b $RAW/\$b
  ln -sf \$f $WORK/\$b/
  cd $ROOT
  DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII \
    MAS_MAX_STEPS=21000 MAS_ON_SERVER=0 \
    timeout 1500 $PY run_transform.py \
      --no_server_replay_path=$WORK/\$b/ \
      --save_path_tensor=$RAW/\$b/ \
      > $LOG/tr_\$b.log 2>&1
  trc=\$?
  $PY experiments/replay_gen/_mk_nonop.py --src $RAW/\$b --dst $DST > $LOG/nonop_\$b.log 2>&1
  nrc=\$?
  rm -rf $WORK/\$b $RAW/\$b
  echo \"\$b tr=\$trc nonop=\$nrc\" >> $LOG/transform_sel_driver.log
"

echo "[$(date '+%F %T')] DONE files=$(ls "$DST"/*.pt 2>/dev/null | wc -l)" | tee -a "$LOG/transform_sel_driver.log"
du -sh "$DST" | tee -a "$LOG/transform_sel_driver.log"
