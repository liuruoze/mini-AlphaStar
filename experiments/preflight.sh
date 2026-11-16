#!/usr/bin/env bash
# 启动前体检（只报不改，任何一条不满足都提示风险，但不会退出非零 —— 与 _preflight.sh 的 pf_* 区分开）
# 用法: bash experiments/preflight.sh [--need-ram 55]
set -u
NEED_RAM=${2:-55}
[ "${1:-}" = "--need-ram" ] || NEED_RAM=55

echo "=== s224 体检 $(date '+%F %T') ==="
df -h /home | tail -1 | awk '{printf "磁盘: 可用 %s / %s (%s)\n", $4, $2, $5}'
awk '/^MemTotal:/{t=$2/1048576} /^MemAvailable:/{a=$2/1048576} END{printf "内存: 可用 %.0fG / 总 %.0fG\n", a, t}' /proc/meminfo
echo "训练进程: $(ps -eo cmd | grep -c '[r]un_sl.py')  评测进程: $(ps -eo cmd | grep -c '[e]val_sl_off_rl')"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/GPU /'
echo "--- 判据 ---"
RAM=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
AV=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
NTR=$(ps -eo cmd | grep -c '[r]un_sl.py')
[ "$RAM" -ge "$NEED_RAM" ] && echo "  内存: OK (${RAM}G >= ${NEED_RAM}G)" || echo "  内存: 不足（${RAM}G < ${NEED_RAM}G）—— 别开新训练，会 OOM"
[ "$AV" -ge 25 ] && echo "  磁盘: OK (${AV}G >= 25G)" || echo "  磁盘: 紧张（${AV}G）—— 先清中间数据"
[ "$NTR" -lt 2 ] && echo "  训练槽位: OK ($NTR/2)" || echo "  训练槽位: 已满（$NTR/2）—— 新 run 会 OOM"
echo "提示：2 训练 + 2 评测 = 越界（今天两次 OOM 都是这样来的）；评测并发请用 run_p8_eval 的默认上限。"
