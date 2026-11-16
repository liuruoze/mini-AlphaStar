#!/usr/bin/env bash
# DAgger 采集：learner 用指定 ckpt 正常出动作，专家 append 覆盖并标注
# 用法: collect_dagger.sh <learner_ckpt> <prefix> <conc> <seed...>
set -u
CKPT=${1:?ckpt}; PREFIX=${2:?prefix}; CONC=${3:-10}; shift 3
export MAS_DUAL_MODE=append
export MAS_EXPERT_NOOP=0          # 专家没动作时沉默，避免在双命令里塞 no-op
export MAS_EXPERT_STRAT=fast_rush
i=0
for SEED in "$@"; do
  TAG="${PREFIX}_${SEED}"
  MAS_EXPERT_RAND_SEED="$SEED" \
    bash "$HOME/mini-AlphaStar/experiments/collect_v3.sh" "$CKPT" "$TAG" "$SEED" 1 30000 1 \
    > "/tmp/${TAG}.out" 2>&1 &
  i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then wait; fi
done
wait
echo "[dagger] all done: $i games"
