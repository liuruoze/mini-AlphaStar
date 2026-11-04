#!/bin/bash
# 把 HierNet-SC2 的 30 个 replay 转成 replay_data_v3 格式（凑官方 90 全量）
#
# 背景（第 78 轮，DS）：
#   官方 data/Replays/readme.txt 明确要求 "use all the 60 + 30 = 90 replays for training"，
#   另外 30 个在 github.com/liuruoze/HierNet-SC2 → data/replay/，已下载到
#   data/Replays/filtered_replays_2/（30 个，429 KB）。
#
# 关键点（都是踩过的坑）：
#   1. MAS_ON_SERVER=1 必须设：transform 里 `SIMPLE_TEST = not P.on_server`，
#      忘了就只转 2 个（与 SL 训练那个 1/60 的坑同源）。
#   2. 输出目录必须显式给 --save_path_tensor，默认是旧格式目录
#      replay_data_tensor_new_small（1,056,596 维），和 v3（343,892 维）不是一套。
#   3. replay_path 要绝对路径且带尾斜杠：脚本里是字符串拼接，且 pysc2 会把相对路径
#      拼到 $SC2PATH/Replays/ 下面。
#
# 用法：bash experiments/_transform_hiernet30.sh
set -u
cd "$HOME/mini-AlphaStar" || exit 9

export SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
export DISPLAY="${DISPLAY:-:1024}"
ulimit -n 65536

# 转换是纯 CPU/IO（CUDA_VISIBLE_DEVICES=-1），不占 GPU
echo "=== transform 30 HierNet-SC2 replays -> replay_data_v3 ==="
MAS_ON_SERVER=1 \
MAS_DATA_FROM=0 \
MAS_DATA_NUM=30 \
CUDA_VISIBLE_DEVICES=-1 \
python run_transform.py \
  --no_server_replay_path="$HOME/mini-AlphaStar/data/Replays/filtered_replays_2/" \
  --save_path_tensor="$HOME/mini-AlphaStar/data/replay_data_v3/"
rc=$?
echo "=== transform rc=$rc ==="

# 校验：只数新产出的 30 个（日期 2021-11-21 ~ 12-09），不把原来 60 个算进来
n_new=$(ls -1 "$HOME/mini-AlphaStar/data/replay_data_v3"/Simple64_2021-11-*.pt \
             "$HOME/mini-AlphaStar/data/replay_data_v3"/Simple64_2021-12-09-*.pt 2>/dev/null | wc -l)
echo "新增（2021-11-21~12-09）tensor 文件数: $n_new  (期望 30)"
echo "replay_data_v3 总数: $(ls -1 "$HOME/mini-AlphaStar/data/replay_data_v3"/*.pt 2>/dev/null | wc -l)  (期望 90)"
exit $rc
