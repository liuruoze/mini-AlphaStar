#!/usr/bin/env bash
# 训练矩阵(10x3)抽稀数据集：1 个 run
# 用法: [DATA=<数据集目录>] bash experiments/train_bvb_m30.sh <tag> <epochs> [seed] [step_size]
#
# 注意：manifest 里声明 "dagger": true 的数据集（如 replay_data_a3nc）必须补 MAS_DAGGER=1，
#       否则训练器按设计拒绝启动（防"DAgger 数据当普通数据训"）。懒人用法见 smoke_bench.sh。
#
# 2026-11-14 R1 加固：
#   ① 启动前做前置检查（磁盘/内存/已有训练数/GPU 显存/数据目录），不满足直接退出码 9；
#   ② 打印完整生效配置（含所有 MAS_SL_* 开关），便于复盘"这一版到底改了什么"；
#   ③ 退出码透传（原来返回的是最后一条 ls 的码，失败也会是 0）。
set -u

_THIS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
. "$_THIS/_preflight.sh"

cd /home/liuruoze/mini-AlphaStar
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII
ulimit -n 65536
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TAG=${1:?tag}
EPOCHS=${2:?epochs}
SEED=${3:-20261108}
SSTEP=${4:-3}
DATA=${DATA:-/home/liuruoze/mini-AlphaStar/data/bvb_m30_sub}

PF_TAG="train:$TAG"
if [ "${MAS_ALLOW_MORE:-0}" = "1" ]; then
  echo "[$PF_TAG][warn] MAS_ALLOW_MORE=1 -> 跳过训练数上限与内存检查（可能 OOM）"
  pf_need_dataset "$DATA"
  pf_disk_gb 20
else
  pf_need_dataset "$DATA"
  pf_disk_gb 25
  pf_ram_gb 55
  pf_train_slots 2
  pf_gpu_free_mb "${GPU:-0}" 4000
fi

mkdir -p "model/sl/$TAG"

# 可复现清单：把这次 run 的完整生效配置写进 model/sl/<tag>/train_manifest.json
# （之前只有零散日志，事后无法确定某个 ckpt 到底用了什么）
export MAS_SL_LR=${MAS_SL_LR:-1e-4}
export MAS_SL_GAMMA=${MAS_SL_GAMMA:-0.2}
bash "$_THIS/write_train_manifest.sh" "$TAG" "$EPOCHS" "$SEED" "$SSTEP" || true
echo "[$(date '+%F %T')] TRAIN START tag=$TAG epochs=$EPOCHS seed=$SEED ss=$SSTEP gpu=${GPU:-0} data=$DATA"
echo "[$PF_TAG] cfg lr=${MAS_SL_LR:-1e-4} gamma=${MAS_SL_GAMMA:-0.2} frame_split=1 val_frac=${MAS_SL_VAL_FRAC:-0.2}"
echo "[$PF_TAG] cfg class_w=[${MAS_SL_CLASS_WEIGHTS:-none}] balance=${MAS_SL_BALANCE:-0}"
echo "[$PF_TAG] cfg restore=${MAS_SL_RESTORE_PATH:-none} load_state=${MAS_SL_LOAD_STATE:-0}"
# 2026-11-20：把架构/目标开关也打出来。manifest 里有，但要等跑完才写；
# 起跑时就看见"这份权重是什么结构"，能省掉一次 40 分钟的错跑。
echo "[$PF_TAG] cfg lstm=${MAS_LSTM_HIDDEN:-128}x${MAS_LSTM_LAYERS:-1} ln=${MAS_LSTM_LN:-off} ent=${MAS_SL_ENT_WEIGHT:-0} attack_w=${MAS_SL_ATTACK_WEIGHT:-1.0} noop_w=${MAS_SL_NOOP_WEIGHT:-1.0} classw_json=${MAS_SL_CLASS_WEIGHTS_JSON:-none}"

if [ "${MAS_DRY_RUN:-0}" = "1" ]; then
  echo "[$PF_TAG] DRY-RUN：前置检查与配置已打印，不启动训练"
  exit 0
fi

MAS_ON_SERVER=1 \
MAS_SL_SEED="$SEED" \
MAS_SL_EPOCHS="$EPOCHS" \
MAS_SL_STEP_SIZE="$SSTEP" \
MAS_SL_LR=${MAS_SL_LR:-1e-4} \
MAS_SL_GAMMA=${MAS_SL_GAMMA:-0.2} \
MAS_SL_FRAME_SPLIT=1 \
MAS_SL_VAL_FRAC=${MAS_SL_VAL_FRAC:-0.2} \
MAS_SL_SPLIT_SEED=20261108 \
MAS_SL_WORKERS=${MAS_SL_WORKERS:-0} \
MAS_SL_MODEL_PATH="./model/sl/$TAG/" \
/home/liuruoze/.conda/envs/th_1_7/bin/python run_sl.py -p1 "$DATA/" \
  > "log/train_$TAG.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] TRAIN END tag=$TAG rc=$rc"
if [ "$rc" -eq 0 ]; then
  ls -l "model/sl/$TAG/" | tail -3
else
  echo "[$PF_TAG][error] 训练失败 rc=$rc，日志尾部：" >&2
  tail -5 "log/train_$TAG.log" | tr -d '\000' >&2
fi
exit "$rc"
