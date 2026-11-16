#!/usr/bin/env bash
# 专家自主采集：learner 全 no-op + 专家 replace（专家自己打）
# 用法: collect_self.sh <ckpt> <prefix> <conc> <seed...>
set -u
CKPT=${1:?ckpt}; PREFIX=${2:?prefix}; CONC=${3:-6}; shift 3
export MAS_LEARNER_NOOP=1
export MAS_DUAL_MODE=replace
export MAS_EXPERT_NOOP=1
STRATS=(fast_rush two_gate heavy_gate macro harass turtle late)
i=0
for SEED in "$@"; do
  TAG="${PREFIX}_${SEED}"
  ST="${MAS_EXPERT_STRAT:-${STRATS[$((i % 7))]}}"
  MAS_EXPERT_RAND_SEED="$SEED" MAS_EXPERT_STRAT="$ST" MAS_V4_DEBUG="${MAS_V4_DEBUG:-0}" \
    bash "$HOME/mini-AlphaStar/experiments/collect_v3.sh" "$CKPT" "$TAG" "$SEED" 1 30000 1 \
    > "/tmp/${TAG}.out" 2>&1 &
  i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then wait; fi
done
wait
echo "[self] all done: $i games"
