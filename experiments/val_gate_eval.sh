#!/usr/bin/env bash
# 双检查点门禁评测（2026-11-15 更新）
#
# 依据（同配方内、共 15 个 h5 run + 4 个 a3nc run 的实测）：
#   1) ep4（最终）val >= 0.60 必崩（7/7 + 今夜 5/5 命中）
#   2) ep2 检查点能救"晚崩"run，但**只在 ep2 val <= 0.40 时有效**
#      （ep2 val 0.29/0.35/0.38 -> 45%/40%/67%；ep2 val 0.483 -> 1/20 = 5%）
#   3) 两个 checkpoint 都要评：好 run 的 ep4 更强，晚崩 run 的 ep2 才是可用的那个
#
# 用法:
#   bash experiments/val_gate_eval.sh "<seeds>" <tag1> [tag2 ...]
#   MAS_GATE_EP4=0.55 MAS_GATE_EP2=0.40 bash experiments/val_gate_eval.sh "3038-3057" tg1 tg2
#   bash experiments/val_gate_eval.sh --dry "<seeds>" tg1 ...   # 只打印决定，不启动
set -u
DRY=0
[ "${1:-}" = "--dry" ] && { DRY=1; shift; }
SEEDS=${1:?seeds（如 "3038-3057" 或 "3038 3039"）}
shift

THR4=${MAS_GATE_EP4:-0.55}
THR2=${MAS_GATE_EP2:-0.40}
cd /home/liuruoze/mini-AlphaStar

_le() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 <= b+0)}'; }

_ep_val() {   # $1=log  $2=epoch
  grep -a 'SL-best' "$1" 2>/dev/null \
    | sed -n "s/.*val_action_acc=\([0-9.]*\) at epoch $2 .*/\1/p" | head -1
}

_launch() {
  local tag="$1"
  if [ "$DRY" = "1" ]; then
    echo "        [dry] 会评测 $tag"
  else
    nohup bash experiments/run_p8_eval.sh "$tag" $SEEDS > "log_night/eval_$tag.log" 2>&1 &
  fi
}

N_EVAL=0
for TAG in "$@"; do
  LOG="log/train_$TAG.log"
  if [ ! -f "$LOG" ]; then printf '%-26s -> SKIP（无日志）\n' "$TAG"; continue; fi
  V4=$(_ep_val "$LOG" 4)
  V2=$(_ep_val "$LOG" 2)
  [ -z "$V4" ] && V4=$(grep -a '最终' "$LOG" 2>/dev/null | tail -1 | sed -n 's/.*val_action_acc=\([0-9.]*\).*/\1/p')

  HAS4=$(ls -1 "model/sl/$TAG"/*_best.pth 2>/dev/null | wc -l)
  HAS2=$(ls -1 "model/sl/$TAG"/*_ep02.pth 2>/dev/null | wc -l)
  DEC="SKIP"
  TARGET=""

  # 规则 1：ep4 过门禁 -> 评 ep4
  if [ -n "$V4" ] && [ "$HAS4" -gt 0 ] && _le "$V4" "$THR4"; then
    DEC="EVAL-ep4"; TARGET="$TAG"
  # 规则 2：ep4 崩 但 ep2 过门禁 -> 造一个 ep2 变体再评
  elif [ -n "$V2" ] && [ "$HAS2" -gt 0 ] && _le "$V2" "$THR2"; then
    DEC="EVAL-ep2"; TARGET="${TAG}ep2"
    if [ "$DRY" != "1" ]; then
      mkdir -p "model/sl/$TARGET"
      cp -f "$(ls -1 model/sl/$TAG/*_ep02.pth | head -1)" "model/sl/$TARGET/${TARGET}_best.pth"
    fi
  fi

  printf '%-26s ep4=%-7s ep2=%-7s -> %s\n' "$TAG" "${V4:-N/A}" "${V2:-N/A}" "$DEC"
  if [ -n "$TARGET" ]; then _launch "$TARGET"; N_EVAL=$((N_EVAL + 1)); fi
done

if [ "$DRY" != "1" ]; then sleep 5; echo "[gate] 启动 $N_EVAL 个评测，当前进程数: $(pgrep -fc eval_sl_off_rl)"; fi
