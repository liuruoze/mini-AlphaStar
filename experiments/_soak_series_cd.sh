#!/bin/bash
# 第 64 轮：公平对照（Kimi-06 / Kimi-11），arms 用**完全相同**的模型模式与窗口长度。
#
# A 臂 local、B 臂 batch，各 1800 s；唯一差别是 MAS_BATCH_ACTING。
# 顺序固定（不可 ABBA，只有一次机会），这点在日志里必须写明。
set -u
cd "$HOME/mini-AlphaStar" || exit 9
DUMP=$HOME/mini-AlphaStar/log/traj_eval30.pt
rm -f "$DUMP"

echo "########## ARM A: local 15x2 (eval) 1800s ##########"
./experiments/_soak_run.sh gpu_local30e 1800 0 15 2 1
echo "########## A done rc=$? ##########"
sleep 5

echo "########## ARM B: batch 15x2 (eval) 1800s + dump ##########"
MAS_DUMP_TRAJ="$DUMP" ./experiments/_soak_run.sh gpu_batch30e 1800 1 15 2 1
echo "########## B done rc=$? ##########"
ls -la "$DUMP" 2>&1
echo "########## SERIES CD END ##########"
