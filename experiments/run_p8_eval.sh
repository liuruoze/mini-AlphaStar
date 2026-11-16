#!/usr/bin/env bash
# P8 评测：run_p8_eval.sh <tag> <seed> [seed...]
#
# 2026-11-14 R1 加固：
#   ① 并发上限（默认 8，OFF_MAX_JOBS 可调）—— 40 个种子一次性全起会打爆单卡显存；
#   ② 逐局校验输出文件，缺局以非零码退出并列出缺失项（原来只打印 MISSING，调用方看不出失败）；
#   ③ GPU 轮转按实际卡数（原来写死 4）。
set -u

_THIS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
. "$_THIS/_preflight.sh"

H=$HOME
TAG=${1:?tag}; shift
SEEDS="$*"
SRC=$H/mini-AlphaStar/model/sl/$TAG
W=$(ls -1 "$SRC"/*_best.pth 2>/dev/null | head -1)
[ -z "${W:-}" ] && { echo "[eval] no best weight in $SRC"; exit 3; }
MD=$H/mini-AlphaStar/model/eval_$TAG
mkdir -p "$MD"; cp -f "$W" "$MD/sl_eval.pth"
cd "$H/mini-AlphaStar-off" || exit 4
export DISPLAY=:1024 SC2PATH=$H/StarCraftII
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -n 65536

PF_TAG="p8:$TAG"
if [ -n "${OFF_GPUS:-}" ]; then IFS=',' read -r -a GPUS <<< "$OFF_GPUS"; else GPUS=(0 1 2 3); fi
NG=${#GPUS[@]}; [ "$NG" -gt 0 ] || NG=1
MAXJ=${OFF_MAX_JOBS:-$((NG * 6))}   # 默认每卡 6 个（实测同卡 8 仍安全，留一点余量）
[ "$MAXJ" -gt $((NG * 8)) ] && MAXJ=$((NG * 8))
# 内存按**实际并发**算：只评 1 个种子时不该要求 24 个进程的内存（R8 修）
_nseed=0; for _s in $SEEDS; do _nseed=$((_nseed + 1)); done
_effj=$MAXJ; [ "$_nseed" -lt "$_effj" ] && _effj=$_nseed
pf_ram_gb $((3 * _effj + 8))

# 全局并发闸门（2026-11-15 修）：MAXJ 只限制**本脚本**，同时起 N 个脚本就会 N 倍并发，
# 02:11 就是这样把 80 个评测进程压上去、触发 37 次 OOM 的。
# OFF_GLOBAL_MAX 是**所有评测脚本合计**的上限（默认 26 ≈ 65GB）。
GLOB=${OFF_GLOBAL_MAX:-26}
[ "$MAXJ" -gt "$GLOB" ] && MAXJ=$GLOB
# 注意：不能用 `pgrep -fc X || echo 0` —— 无匹配时 pgrep 既打印 "0" 又返回 1，
# 结果变成 "0\n0"，`[ "$_n" -lt ... ]` 报整数比较错误、闸门退化成无限等待（03:27 踩过）。
# 只匹配真实评测进程（cmdline 形如 `/path/python eval_sl_off_rl.py`）。
# 用裸 'eval_sl_off_rl' 会被任何命令行里含该字样的 shell/编辑器误匹配 -> 闸门假死锁（08:53 实测踩到）。
_glob_slots() { ps -eww -o comm=,args= 2>/dev/null | awk '$1 ~ /^python/ && /eval_sl_off_rl/ {n++} END{print n+0}'; }
_glob_gate() {
  local _n
  while :; do
    _n=$(_glob_slots)
    [ "$_n" -lt "$GLOB" ] && return 0
    sleep 5
  done
}

echo "[$PF_TAG] tag=$TAG seeds=[$SEEDS] jobs=$_effj global_max=$GLOB gpus=${GPUS[*]}"
i=0
_skip=0
batch=()
for s in $SEEDS; do
  # 断点续跑：已经有了且非空就跳过（OFF_SKIP_EXISTING=1）
  if [ "${OFF_SKIP_EXISTING:-0}" = "1" ] && [ -s "outputs/${TAG}s${s}_${s}.txt" ]; then
    _skip=$((_skip + 1)); continue
  fi
  _glob_gate
  g=${GPUS[$((i % NG))]}
  OFF_DIFFICULTY="${OFF_DIFFICULTY:-3}" OFF_GAME_STEPS="${OFF_GAME_STEPS:-40000}" \
    OFF_TAG="${TAG}s${s}" OFF_GPUS="$g" \
    bash run_eval_batch.sh "$MD" "${TAG}s${s}" "$s" 1 1 > "/tmp/${TAG}_${s}.out" 2>&1 &
  batch+=($!)
  i=$((i+1))
  if [ "${#batch[@]}" -ge "$MAXJ" ]; then
    wait "${batch[@]}" 2>/dev/null || true
    batch=()
  fi
done
[ "${#batch[@]}" -gt 0 ] && wait "${batch[@]}" 2>/dev/null || true

echo "[$(date '+%F %T')] EVAL DONE $TAG (skipped_existing=$_skip)"
_miss=0
for s in $SEEDS; do
  f="outputs/${TAG}s${s}_${s}.txt"
  if [ -s "$f" ]; then head -1 "$f"; else echo "MISSING $f"; _miss=$((_miss + 1)); fi
done
if [ "$_miss" -gt 0 ]; then
  echo "[$PF_TAG][error] $_miss 局缺失（见上）" >&2
  exit 7
fi
exit 0
