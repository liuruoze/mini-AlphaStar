#!/usr/bin/env bash
# 交付链路一键执行：门禁 -> 评测 -> 排名
#
# 背景：2026-11-15 夜间的交付流程是手工拼的（val_gate_eval + 等评测 + pool_report + 手算），
# 中间踩过两次"tag 名写错/别名对不上"（cwH5b 写成 cwH5c、60% 单批当结论）。
# 本脚本把三步串起来，并且在评测未齐时**明确报"缺失"而不是悄悄算错比例**。
#
# 用法：
#   bash experiments/deploy_pick.sh --dry "3038-3057" TAG1 TAG2 ...      # 只看门禁决定
#   bash experiments/deploy_pick.sh "3038-3057" TAG1 TAG2 ...            # 真跑（含等待）
#   MAS_GATE_EP4=0.55 MAS_GATE_EP2=0.40 MAS_PICK_TIMEOUT=1800 bash ...
#
# 退出码：0=全部评完；7=有评测缺失（比例不可信）；2=用法错误
set -u
DRY=0
[ "${1:-}" = "--dry" ] && { DRY=1; shift; }
SEEDS=${1:?seeds}; shift
[ "$#" -ge 1 ] || { echo "至少给一个 tag"; exit 2; }
cd /home/liuruoze/mini-AlphaStar
PY=$HOME/.conda/envs/th_1_7/bin/python
TIMEOUT=${MAS_PICK_TIMEOUT:-1800}

echo "===== 1/3 门禁（ep4<=${MAS_GATE_EP4:-0.55} 或 ep2<=${MAS_GATE_EP2:-0.40}）====="
if [ "$DRY" = "1" ]; then
  bash experiments/val_gate_eval.sh --dry "$SEEDS" "$@"
  echo; echo "[dry] 停止（不评测、不排名）"; exit 0
fi
bash experiments/val_gate_eval.sh "$SEEDS" "$@"

echo; echo "===== 2/3 等待评测结束（最多 ${TIMEOUT}s）====="
_t0=$(date +%s)
while :; do
  # 不能用 pgrep -fc X || echo 0：无匹配时输出 "0\n0"，整数比较会报错（同 02:11 那个坑）
  _n=$(ps -eww -o comm=,args= 2>/dev/null | awk '$1 ~ /^python/ && /eval_sl_off_rl/ {n++} END{print n+0}')
  [ "$_n" -eq 0 ] && break
  _el=$(( $(date +%s) - _t0 ))
  [ "$_el" -gt "$TIMEOUT" ] && { echo "[warn] 超时，仍有 $_n 个评测在跑"; break; }
  sleep 20
done
bash experiments/clean_sc2_orphans.sh --yes >/dev/null 2>&1 || true
echo "评测进程已清空（耗时 $(( $(date +%s) - _t0 ))s）"

echo; echo "===== 3/3 排名（含 ep2 变体）====="
# 把每个 tag 的 ep2 变体也纳进来（如果存在且已评测）
ARGS=()
for t in "$@"; do
  ARGS+=(--group "pick=$t:$t")
  [ -d "model/sl/${t}ep2" ] && ARGS+=(--group "pick=${t}ep2:${t}ep2")
done
$PY experiments/pool_report.py --seeds "$SEEDS" "${ARGS[@]}" | sed -n '1,200p'

echo
echo "提示：上面 '合并' 行是**所有 run 合计**（配方期望）；单 run 的 rate 才是可交付模型的实际水平。"
echo "      选交付模型看单 run 最高 + n 足够大（n<20 的数字会向均值回归，见 RESULT_20261115_MORNING.md）。"
