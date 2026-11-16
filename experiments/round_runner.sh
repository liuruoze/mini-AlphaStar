#!/usr/bin/env bash
# 一轮完整实验的编排器（采集 → 转换 → 建集 → 训练 → 评测）
# 目的：把今晚反复手写的命令固化，减少路径/参数拼错（已发生过两次失败）。
#
# 用法:
#   round_runner.sh collect <prefix> <learner_ckpt> <conc> <seed_lo> <seed_hi>
#   round_runner.sh convert <prefix>
#   round_runner.sh build   <out_tag> <src_prefix> [--with-base]
#   round_runner.sh train   <data_dir> <tag> <lr> <epochs> [gpu] [weights]
#   round_runner.sh eval    <tag> [seed...]
#   round_runner.sh runs   [--n 30] [--only-good]
#   round_runner.sh check  <data_dir>
#   round_runner.sh table  <tag...> [--seeds 3038-3077] [--merge] [--compare ...]
#   round_runner.sh kill   <tag子串> [--yes]
#   round_runner.sh status
set -u
H="$HOME"
M="$H/mini-AlphaStar"
PY="$H/.conda/envs/th_1_7/bin/python"
CMD=${1:?usage: round_runner.sh <collect|convert|build|train|eval|status> ...}
shift

_log() { echo "[$(date '+%F %T')] $*"; }

case "$CMD" in
  collect)
    PREFIX=${1:?prefix}; CKPT=${2:?ckpt}; CONC=${3:-10}; LO=${4:-3038}; HI=${5:-3067}
    SEEDS=$(seq -s ' ' "$LO" "$HI")
    _log "collect prefix=$PREFIX learner=$CKPT seeds=$LO-$HI conc=$CONC"
    nohup bash "$M/experiments/collect_dagger.sh" "$CKPT" "$PREFIX" "$CONC" $SEEDS \
      > "$M/log_night/collect_$PREFIX.log" 2>&1 &
    _log "launched -> log_night/collect_$PREFIX.log"
    ;;
  convert)
    PREFIX=${1:?prefix}
    _log "convert prefix=$PREFIX（转完自动删 raw）"
    nohup bash "$M/experiments/convert_self_clean.sh" "$PREFIX" 6 \
      > "$M/log_night/convert_$PREFIX.log" 2>&1 &
    _log "launched -> log_night/convert_$PREFIX.log"
    ;;
  build)
    OUT=${1:?out_tag}; SRC=${2:?src_prefix}; WITH_BASE=${3:-}
    DST="$M/data/replay_data_$OUT"
    rm -rf "$DST"; mkdir -p "$DST"
    for f in "$H"/bvb/onpolicy/${SRC}_*/nonop/*.pt; do
      [ -f "$f" ] || continue
      ln -sf "$(readlink -f "$f")" "$DST/${SRC}_$(basename "$f")"
    done
    if [ "$WITH_BASE" = "--with-base" ]; then
      for f in "$M"/data/bvb_m30_sub/*.pt; do
        ln -sf "$(readlink -f "$f")" "$DST/base_$(basename "$f")"
      done
    fi
    "$PY" - "$DST" "$OUT" << 'PY'
import glob, json, os, sys, torch
d, tag = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(d, '*.pt')))
n = sum(int(torch.load(p, map_location='cpu')[0].shape[0]) for p in fs)
json.dump({'dataset': tag, 'dagger': True, 'files': len(fs), 'frames': n},
          open(os.path.join(d, 'manifest.json'), 'w'), ensure_ascii=False, indent=2)
print('[build] %s files=%d frames=%d' % (tag, len(fs), n))
PY
    ;;
  train)
    DATA=${1:?data}; TAG=${2:?tag}; LR=${3:-1e-3}; EP=${4:-4}; GPU=${5:-0}; W=${6:-}
    _log "train tag=$TAG lr=$LR ep=$EP gpu=$GPU data=$DATA"
    bash "$M/experiments/preflight.sh" | sed 's/^/[preflight] /' 
    if [ -n "$W" ]; then
      nohup env GPU=$GPU MAS_DAGGER=1 DATA=$DATA MAS_SL_LR=$LR MAS_SL_CLASS_WEIGHTS="$W" \
        bash "$M/experiments/train_bvb_m30.sh" "$TAG" "$EP" 20261108 3 \
        > "$M/log_night/$TAG.log" 2>&1 &
    else
      nohup env GPU=$GPU MAS_DAGGER=1 DATA=$DATA MAS_SL_LR=$LR \
        bash "$M/experiments/train_bvb_m30.sh" "$TAG" "$EP" 20261108 3 \
        > "$M/log_night/$TAG.log" 2>&1 &
    fi
    _log "launched -> log_night/$TAG.log"
    ;;
  eval)
    TAG=${1:?tag}; shift
    SEEDS=${*:-3038 3039 3040 3041 3042}
    _log "eval tag=$TAG seeds=$SEEDS"
    nohup bash "$M/experiments/run_p8_eval.sh" "$TAG" $SEEDS \
      > "$M/log_night/eval_$TAG.log" 2>&1 &
    _log "launched -> log_night/eval_$TAG.log"
    ;;
  status)
    date '+%H:%M:%S'
    echo "sl_running=$(pgrep -fc run_sl.py) eval_running=$(pgrep -fc eval_sl_off_rl)"
    df -h /home | tail -1
    ;;
  check)
    DATA=${1:?data dir}; shift || true
    "$PY" "$M/experiments/check_dataset.py" "$DATA" "$@"
    ;;
  runs)
    "$PY" "$M/experiments/runs_overview.py" "$@"
    ;;
  table)
    "$PY" "$M/experiments/eval_table.py" "$@"
    ;;
  kill)
    TAG=${1:?tag 子串}; shift || true
    bash "$M/experiments/kill_tag.sh" "$TAG" "$@"
    ;;
  *)
    echo "unknown cmd: $CMD"; exit 2;;
esac
