#!/bin/bash
# 重新转换 replay：大幅降低 no_op 占比（第 79 轮，DS 备用方案）
#
# 动机（DS12 79.19-79.20）：
#   现有 replay_data_v3 里 no_op 占 **81%**，导致 SL 模型实战中 99.9% 输出 no_op、
#   killed=0。官方模型实战 no_op < 4%、每帧都在建造/造兵/攻击 ——
#   说明官方训练数据的 no_op 占比远低于我们。
#
#   根因在 transform_replay_data.py 的下采样参数：
#     no_op_threshold = 0.10              -> 只保留 10% 的 no_op 帧
#     smart_unit_threshold = 0.3          -> 只保留 70%
#     Harvest_Gather_unit_threshold = 0.3 -> 只保留 70%
#     Attack_pt_threshold = 0.5           -> 只保留 50%
#   看起来 no_op 已被压到 10%，但**有动作的帧也被大量丢弃**（动作解析失败 + 阈值），
#   最终 no_op 仍占 81%。
#
#   本脚本把"有动作的帧"几乎全保留（0.95）、no_op 只留 2%，
#   预期把 no_op 占比压到 20-30% 量级。
#
# ⚠️ 这会改变数据分布，必须与基线（现有 v3）做消融对照。
# 用法：bash experiments/_transform_lowop.sh [N]
#   N = 处理前 N 个 replay（默认 10，先小规模验证）
set -u
cd "$HOME/mini-AlphaStar" || exit 9

export SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
export DISPLAY="${DISPLAY:-:1024}"
ulimit -n 65536

N="${1:-10}"
OUT="$HOME/mini-AlphaStar/data/replay_data_v3_lowop"
mkdir -p "$OUT"

echo "=== 重新转换 $N 个 replay -> $OUT ==="
echo "阈值：no_op=0.02 smart_unit=0.95 harvest=0.95 attack=0.95"

MAS_ON_SERVER=1 MAS_DATA_FROM=0 MAS_DATA_NUM="$N" CUDA_VISIBLE_DEVICES=-1 \
python run_transform.py \
  --no_server_replay_path="$HOME/mini-AlphaStar/data/Replays/filtered_replays_1/" \
  --save_path_tensor="$OUT/" \
  --no_op_threshold=0.02 \
  --Smart_unit_threshold=0.95 \
  --Harvest_Gather_unit_threshold=0.95 \
  --Attack_pt_threshold=0.95
rc=$?

echo "=== transform rc=$rc ==="
echo "产出文件数: $(ls -1 "$OUT"/*.pt 2>/dev/null | wc -l) (期望 $N)"
echo "总大小: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
exit $rc
