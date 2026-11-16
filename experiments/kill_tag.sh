#!/usr/bin/env bash
# 按 tag 杀进程（run_sl.py / eval_sl_off_rl.py）。
#
# 背景：tag 只存在于**环境变量**（MAS_SL_MODEL_PATH / OFF_TAG），
#      `ps` 的 cmdline 里只有 `python run_sl.py -p1 <data>`，
#      所以 `pkill -f <tag>` 永远匹配不到 —— 今晚因此误留过一个训练进程并触发 OOM。
#
# 用法：
#   bash experiments/kill_tag.sh <tag子串>          # 只列出（dry-run）
#   bash experiments/kill_tag.sh <tag子串> --yes    # 真杀
set -u
PAT=${1:?usage: kill_tag.sh <tag-substring> [--yes]}
DRY=1
[ "${2:-}" = "--yes" ] && DRY=0
found=0

for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ -r "$p/cmdline" ] || continue
  cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
  case "$cmd" in
    *run_sl.py*|*eval_sl_off_rl.py*) ;;
    *) continue ;;
  esac
  envs=$(tr '\0' '\n' < "$p/environ" 2>/dev/null) || continue
  if printf '%s\n' "$envs" | grep -qE "^(MAS_SL_MODEL_PATH|OFF_TAG|OFF_MODEL_PATH)=.*${PAT}"; then
    printf 'PID %-7s %s\n' "$pid" "$(printf '%s' "$cmd" | cut -c1-100)"
    printf '        %s\n' "$(printf '%s\n' "$envs" | grep -E '^(MAS_SL_MODEL_PATH|OFF_TAG)=' | tr '\n' ' ')"
    found=$((found + 1))
    if [ "$DRY" -eq 0 ]; then
      kill -9 "$pid" 2>/dev/null && echo "        -> killed"
    fi
  fi
done

echo "matched=$found dry_run=$DRY"
[ "$DRY" -eq 1 ] && [ "$found" -gt 0 ] && echo "（确认无误后加 --yes 真杀）"
exit 0
