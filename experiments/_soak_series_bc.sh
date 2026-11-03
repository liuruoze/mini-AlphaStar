#!/bin/bash
# 连续长跑编排（第 61–62 轮，DeepSeek）。
#
# 为什么要串成一条：三组 30 min 长跑如果每次都由我手动起，中间会有 3–5 min 空档，
# 而且容易忘传环境变量（第 53 轮就漏传过一次 on_server）。
# 串成一个 tmux 会话后，B/C 两组之间只隔 2 s 清场时间，且环境在脚本里锁死。
#
# 用法：`tmux new -d -s series './_soak_series_bc.sh'`
#
# 注意：本脚本**不**做 `pkill -f run_rl.py`（内层 `_soak_run.sh` 已经做了），
# 因为外层脚本自己的命令行里含有 "run_rl" 之外的词，但内层 pkill 会匹配到
# **正在执行 `python run_rl.py` 的那个进程**——这是期望行为。
set -u
cd "$HOME/mini-AlphaStar" || exit 9

DUMP=$HOME/mini-AlphaStar/log/traj_batch30b.pt
rm -f "$DUMP"

echo "########## B: batch 15 Worker x 1 actor (15 actor) ##########"
./experiments/_soak_run.sh gpu_batch15 1800 1 15 1 1
echo "########## B done rc=$? ##########"
sleep 5

echo "########## C: batch 15 Worker x 2 actor (30 actor) + dump ##########"
MAS_DUMP_TRAJ="$DUMP" ./experiments/_soak_run.sh gpu_batch30b 1800 1 15 2 1
echo "########## C done rc=$? ##########"

ls -la "$HOME/mini-AlphaStar/log/" | grep -E "soak_gpu_batch15|soak_gpu_batch30b"
ls -la "$DUMP" 2>&1
echo "########## SERIES END ##########"
