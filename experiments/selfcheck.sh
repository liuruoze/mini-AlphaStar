#!/usr/bin/env bash
# 一条命令自检仓库健康度（改完代码、交接、或怀疑环境有问题时跑）
# 用法: bash experiments/selfcheck.sh [--fast]
#   默认会跑离线单测（~10s）；--fast 跳过单测，只做语法/静态检查。
set -u
FAST=0
[ "${1:-}" = "--fast" ] && FAST=1
H=$HOME
M=$H/mini-AlphaStar
OFF=$H/mini-AlphaStar-off
PY=$H/.conda/envs/th_1_7/bin/python
FAIL=0

hr() { printf '%s\n' "------------------------------------------------------------"; }
ok() { printf '[ok]   %s\n' "$*"; }
bad() { printf '[FAIL] %s\n' "$*"; FAIL=$((FAIL + 1)); }

hr; echo "1) Shell 脚本语法"
for f in "$M"/experiments/*.sh "$OFF"/*.sh; do
  [ -f "$f" ] || continue
  if bash -n "$f" 2>/dev/null; then :; else bad "bash -n $(basename "$f")"; fi
done
ok "shell 语法检查完成（失败数见上）"

hr; echo "2) Python 关键模块编译"
for f in "$M/alphastarmini/core/sl/sl_train_by_tensor.py" \
         "$M/alphastarmini/core/sl/sl_loss_multi_gpu.py" \
         "$M/experiments/to_fp16.py" \
         "$M/experiments/eval_table.py" \
         "$M/experiments/env_index.py" \
         "$M/experiments/check_dataset.py" \
         "$M/experiments/test_patches.py" \
         "$M/experiments/runs_overview.py" \
         "$M/experiments/pool_report.py" \
         "$OFF/alphastarmini/core/arch/action_type_head.py"; do
  [ -f "$f" ] || { bad "缺文件 $f"; continue; }
  if ! "$PY" -m py_compile "$f" 2>/dev/null; then bad "py_compile $(basename "$f")"; fi
done
ok "py_compile 完成（失败数见上）"

hr; echo "3) 开关文档与代码是否一致"
if "$PY" "$M/experiments/env_index.py" --check >/tmp/_envcheck.log 2>&1; then
  ok "ENV_KNOBS.md 与代码一致"
else
  bad "ENV_KNOBS.md 过期，跑 python3 experiments/env_index.py --md experiments/ENV_KNOBS.md"
  tail -2 /tmp/_envcheck.log
fi

if [ "$FAST" -eq 0 ]; then
  hr; echo "4) 离线单测"
  if "$PY" "$M/experiments/test_patches.py" >/tmp/_unittest.log 2>&1; then
    ok "单测全过（$(grep -c PASS /tmp/_unittest.log) 项）"
  else
    bad "单测失败"; grep -E 'FAIL|失败项' /tmp/_unittest.log | head -5
  fi
fi

hr; echo "5) 最近是否有 OOM 杀进程"
_oom=$(dmesg 2>/dev/null | grep -c 'Out of memory: Killed process' || true)
if [ "${_oom:-0}" -gt 0 ]; then
  echo "[warn] dmesg 里有 ${_oom} 次 OOM kill（最近一次）："
  dmesg 2>/dev/null | grep 'Out of memory: Killed process' | tail -1 | cut -c1-140
else
  ok "没有 OOM 记录"
fi

hr; echo "6) 两仓同名脚本一致性"
# 契约：off 仓（官方口径评测仓）里凡与主仓 experiments/ 同名的脚本，
# 要么内容完全相同，要么是只做 exec 转发的桩；否则就是"两份各改一半"。
for _a in "$M"/experiments/*.sh; do
  _f=$(basename "$_a"); _b="$OFF/$_f"
  [ -f "$_b" ] || continue
  if cmp -s "$_a" "$_b"; then
    :
  elif grep -q "mini-AlphaStar/experiments/$_f" "$_b" 2>/dev/null; then
    :
  else
    bad "两仓同名脚本内容不同且不是转发桩：$_f"
  fi
done
ok "两仓同名脚本检查完成（失败数见上）"

hr; echo "7) 运行环境快照"
bash "$M/experiments/preflight.sh" 2>/dev/null | sed 's/^/       /'

hr
if [ "$FAIL" -eq 0 ]; then
  echo "SELFCHECK PASS"
  exit 0
else
  echo "SELFCHECK FAIL ($FAIL 项)"
  exit 1
fi
