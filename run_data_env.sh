#!/bin/bash
# 90 replay 训练启动器（用户定规：训练一律 90 个 replay，与官方同量）
set -u
cd "$HOME/mini-AlphaStar" || exit 1
LOG="${1:-log/p2_s90.log}"
export MAS_ON_SERVER=1
export MAS_MAX_ENTITIES=${MAS_MAX_ENTITIES:-128}
export MAS_SL_EPOCHS=${MAS_SL_EPOCHS:-10}
export MAS_SL_LR=${MAS_SL_LR:-1e-4}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${P2_GPU:-1}"
mkdir -p "$(dirname "$LOG")"
echo "LAUNCH $(date +%F_%H:%M:%S) gpu=${CUDA_VISIBLE_DEVICES} files=90" > "$LOG"
${MAS_SL_PY:-/home/liuruoze/.conda/envs/th_1_7/bin/python} -u run_sl.py -p1 ${MAS_DATA_DIR:-./data/replay_data_v3_90_nonop/} >> "$LOG" 2>&1
RC=$?
echo "EXIT=$RC END $(date +%F_%H:%M:%S)" >> "$LOG"
exit $RC
