#!/usr/bin/env bash
# 早停看门狗：ep2 的 val acc 超过阈值就杀掉该 run。
#
# 依据（2026-11-20，12 个 run 的实测）：ep2 val > 0.45 的 run 全部塌缩（5/5），
# 6 个基线 run 的 ep2 最高只有 0.284，因此 0.45 处没有误杀。
#
# 用法：
#   bash experiments/early_kill.sh sw07 sw08           # 默认阈值 0.45，检查第 2 条 val
#   nohup bash experiments/early_kill.sh sw07 sw08 > /tmp/ek.log 2>&1 &

set -u
THRESH="${EK_THRESH:-0.45}"
EPOCH_N="${EK_EPOCH:-2}"
LOG_DIR="${LOG_DIR:-$HOME/mini-AlphaStar/log}"
M="$HOME/mini-AlphaStar"
TAGS="$*"
[ -z "$TAGS" ] && { echo "用法: early_kill.sh <tag> [tag...]"; exit 2; }

echo "[early_kill] watching $TAGS  thresh=$THRESH at val#$EPOCH_N"
declare -A done_flag

while true; do
  all_done=1
  for t in $TAGS; do
    [ -n "${done_flag[$t]:-}" ] && continue
    f="$LOG_DIR/train_$t.log"
    [ -f "$f" ] || { all_done=0; continue; }
    v=$(grep "Val action acc" "$f" 2>/dev/null | sed -n "${EPOCH_N}p" | awk '{print $4}' | tr -d '.')
    # 形如 0.501288 -> 去点后是 0501288，取前几位比较
    if [ -n "$v" ]; then
      val=$(grep "Val action acc" "$f" | sed -n "${EPOCH_N}p" | awk '{print $4}' | sed 's/\.$//')
      if awk -v a="$val" -v b="$THRESH" 'BEGIN{exit !(a>b)}'; then
        echo "[early_kill] $t ep${EPOCH_N} val=$val > $THRESH → 杀"
        pkill -f "train_bvb_m30.sh $t" 2>/dev/null
        pkill -f "$t 4 2026" 2>/dev/null
        pkill -f "run_sl.py.*$t" 2>/dev/null
        echo "$val" > "$M/model/sl/$t/KILLED_EP${EPOCH_N}_VAL"
        done_flag[$t]=1
      else
        done_flag[$t]=1
        echo "[early_kill] $t ep${EPOCH_N} val=$val ≤ $THRESH → 放行"
      fi
    else
      all_done=0
    fi
  done
  [ "$all_done" = "1" ] && { echo "[early_kill] 全部处理完毕"; exit 0; }
  sleep 60
done
