#!/bin/bash
# 长跑编排器（第 50 轮，DeepSeek）。
#
# 职责：一次调用完成「清场 -> 起监控 -> 跑训练 -> 收尾统计」，
# 让 4 种配置（local/batch × 小规模/默认规模）可以用**完全相同**的口径复现。
#
# 用法：
#   experiments/_soak_run.sh <tag> <秒> <BATCH:0|1> <PARALLEL> <ACTOR_NUMS> [ON_SERVER:0|1]
# 例：
#   experiments/_soak_run.sh batch30 1800 1 15 2 1
#
# 第 50 轮新增第 6 个参数 `ON_SERVER`：**这一条是审计发现的关键开关**。
# `rl_vs_inner_bot_mp.py` 里 `SIMPLE_TEST = not P.on_server`，而 `test()` 中
# `if SIMPLE_TEST: use_cuda_device = False` —— 也就是说 `MAS_ON_SERVER=0`（默认）
# 时整个训练**在 CPU 上跑**。之前所有 `run_rl.py` 的 smoke 数字都是 CPU 数字，
# 由此推导的「显存 2 份 vs N+1 份」也没有跑过对应路径。
# 复现 GPU 路径必须 `MAS_ON_SERVER=1`。
#
# 为什么必须用脚本而不是一行命令：
#   1) `ulimit -n 65536` 必须在**同一 shell** 里、且在 python 启动之前生效
#      （750 个参数张量 + Adam state 撞 fd=1024 会报 unable to open shared memory object）；
#   2) 监控进程必须与训练进程同生命周期，否则长跑结束还留个采样器；
#   3) 墙钟要单独记（训练自己只报 actor 的活跃时间，不含启动和收尾）。
set -u

TAG=${1:?tag}
SECS=${2:?seconds}
BATCH=${3:?batch 0/1}
PAR=${4:?parallel}
ACT=${5:?actor_nums}
ONSRV=${6:-0}

ROOT=$HOME/mini-AlphaStar
OUT=$ROOT/log/soak_$TAG
mkdir -p "$OUT"

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate th_1_7
cd "$ROOT" || exit 9

export DISPLAY=:1024
export SC2PATH=$HOME/StarCraftII
ulimit -n 65536

# nproc=128 时 BLAS 默认每个进程开 128 线程，50 个进程抢 128 核 -> preprocess 200 ms
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

export MAS_ON_SERVER=$ONSRV
export MAS_BATCH_ACTING=$BATCH
export MAS_PARALLEL=$PAR
export MAS_ACTOR_NUMS=$ACT
export MAS_MAX_TIME=$SECS
export MAS_MAX_EPISODES=10000
export MAS_MAX_FRAMES=0
# 队列超时保持默认 1800 s —— **故意不调小**。
# 第 50 轮第一次跑时我设成 60 s，结果 Parameter_Server 在 60 s 就因 queue.Empty 退出，
# 哨兵（reaper 投递）根本没被验证到 —— 探针把要测的东西短路掉了。
# 现在 1800 s 远大于一个 episode 的耗时，Parameter_Server 必然活着等到哨兵。
export MAS_QUEUE_TIMEOUT=1800
# 第 65 轮（Kimi-03）：KL 语义开关。legacy=截断部分和（历史口径，默认）；renorm=共同支撑集重归一化 KL
export MAS_KL_MODE=${MAS_KL_MODE:-legacy}

# 避开 cuda:0（实测偏慢 20~45%）；4 张卡都可见，白名单从 1 开始
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAS_GPU_LIST=1,2,3

echo "=== soak tag=$TAG secs=$SECS batch=$BATCH par=$PAR act=$ACT on_server=$ONSRV kl_mode=$MAS_KL_MODE ==="
echo "=== start $(date '+%F %T') ==="

# 清场：上一轮如果异常退出，SC2 会留着占内存和端口
pkill -f SC2_x64 2>/dev/null
pkill -f run_rl.py 2>/dev/null
sleep 2

python experiments/_soak_monitor.py --out "$OUT/monitor.csv" --interval 10 \
    --duration $((SECS + 300)) > "$OUT/monitor.log" 2>&1 &
MON=$!

START=$(date +%s)
python run_rl.py > "$OUT/run.log" 2>&1
RC=$?
END=$(date +%s)

kill "$MON" 2>/dev/null
wait "$MON" 2>/dev/null

echo "rc=$RC wall=$((END - START))" > "$OUT/summary.txt"
echo "=== end $(date '+%F %T') rc=$RC wall=$((END - START))s ==="

# 收尾：不管训练怎么退的，SC2 都不能留
pkill -f SC2_x64 2>/dev/null
exit $RC
