#!/usr/bin/env bash
# 清理孤儿 SC2 进程。
# 背景：评测脚本被 kill 或 OOM 时，python 包一层 run_eval_batch，SC2_x64 是**子进程**，
# 杀掉 python 不会带走 SC2 —— 2026-11-15 04:09 发现 50 个孤儿占 19.5G，把双训练挤到只剩 3G。
# 用法：
#   bash clean_sc2_orphans.sh          # 只看（dry-run）
#   bash clean_sc2_orphans.sh --yes    # 真杀
#
# 2026-11-16 加固：改用**精确进程名**匹配（pgrep -x / comm 列）。
# 原来用 `ps -ef | grep '[S]C2_x64'`，凡是命令行里提到 SC2_x64 的进程（ssh 的 bash 包装、
# 编辑器、一次 grep）都会被算作孤儿，然后被 kill -9 —— 实测无对局时会误报 2 个。
set -u
YES=0
[ "${1:-}" = "--yes" ] && YES=1

_live=$(ps -eww -o comm=,args= 2>/dev/null | awk '$1 ~ /^python/ && /eval_sl_off_rl/ {n++} END{print n+0}')
[ -z "$_live" ] && _live=0
PIDS=$(pgrep -x SC2_x64 2>/dev/null | tr '\n' ' ')
_n=$(echo "$PIDS" | wc -w)
_tot=$(ps -eo rss=,comm= 2>/dev/null | awk '$2 == "SC2_x64" {s += $1} END {printf "%.1f", s / 1048576}')
echo "SC2 进程 ${_n} 个，合计 ${_tot:-0}G；当前活跃评测进程 ${_live} 个"

if [ "$_live" -gt 0 ]; then
  echo "[skip] 还有评测在跑，不能清理 SC2（会杀掉正在对局的进程）"
  exit 2
fi
if [ "$_n" -eq 0 ]; then
  echo "没有孤儿，退出"
  exit 0
fi
if [ "$YES" -ne 1 ]; then
  echo "[dry-run] 加 --yes 才会真杀"
  for p in $PIDS; do ps -o pid=,rss=,etime= -p "$p" 2>/dev/null | cut -c1-100; done
  exit 0
fi
for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
sleep 3
echo "已清理，剩余 $(pgrep -x SC2_x64 2>/dev/null | wc -l) 个"
free -g | sed -n 2p
