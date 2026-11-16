#!/usr/bin/env bash
# 批量训练队列：一次跑 N 个独立 run（每次 2 个并行，GPU 0/1），全部完成后退出。
# 用法: [EXTRA_ENV="MAS_SL_EARLY_W=3.0 ..."] bash experiments/queue_train.sh \
#          <tag_prefix> <data_dir> <lr> <epochs> <n_runs> [seed_start]
set -u
TAGP=${1:?tag prefix}
DATA=${2:?data dir}
LR=${3:?lr}
EP=${4:?epochs}
NRUN=${5:?n runs}
SEED0=${6:-20261121}
EXTRA_ENV=${EXTRA_ENV:-}
BASE_W='train_probe=0.15,build_pylon=8,build_gateway=8,build_nexus=0.1,train_zealot=4'

cd /home/liuruoze/mini-AlphaStar
echo "[$(date '+%F %T')] QUEUE START tag=$TAGP data=$DATA lr=$LR ep=$EP n=$NRUN seed0=$SEED0 extra=[$EXTRA_ENV]"

launch () {   # $1=gpu  $2=seed
  local G=$1 S=$2 TAG="${TAGP}_s${S}"
  nohup env GPU=$G MAS_DAGGER=1 DATA="$DATA" MAS_SL_LR="$LR" \
        MAS_SL_CLASS_WEIGHTS="${MAS_SL_CLASS_WEIGHTS:-$BASE_W}" \
        $EXTRA_ENV \
        bash experiments/train_bvb_m30.sh "$TAG" "$EP" "$S" 3 \
        > "log_night/$TAG.log" 2>&1 &
}

k=0
while [ $k -lt $NRUN ]; do
  S1=$((SEED0 + k))
  launch 0 $S1
  if [ $((k + 1)) -lt $NRUN ]; then
    S2=$((SEED0 + k + 1))
    launch 1 $S2
  fi
  wait
  k=$((k + 2))
done

echo "[$(date '+%F %T')] QUEUE DONE tag=$TAGP"
for d in model/sl/${TAGP}_s*; do
  [ -d "$d" ] || continue
  printf '%-28s %s\n' "$(basename "$d")" "$(ls -1 "$d"/*_best.pth 2>/dev/null | wc -l)"
done
