#!/usr/bin/env bash
# 一次训练：train_sweep_one.sh <data_dir> <tag> <lr> <epochs> <gpu> [seed] [step_size]
set -u
DATA=${1:?data}; TAG=${2:?tag}; LR=${3:?lr}; EP=${4:?epochs}; GPU=${5:?gpu}
SEED=${6:-20261108}; SS=${7:-3}
DATA="$DATA" GPU="$GPU" MAS_SL_LR="$LR" MAS_DAGGER="${MAS_DAGGER:-1}" \
  bash "$HOME/mini-AlphaStar/experiments/train_bvb_m30.sh" "$TAG" "$EP" "$SEED" "$SS"
