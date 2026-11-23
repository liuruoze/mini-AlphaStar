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
# 第 72 轮：通知机制 event|sem（默认 event，保持历史行为）
# 2026-11-23：默认改 sem。实测（同配置 A/B）通知 14.39 ms → 0.08 ms、
# 每轮 98.97 → 64.00 ms、单 actor 吞吐 2.90 → 3.51（+21%）。
# 需要与历史数据可比时显式传 MAS_NOTIFY=event。
export MAS_NOTIFY=${MAS_NOTIFY:-sem}

# ---- 第 84 轮（DS，#71）：把 §84 长跑里"手工 export"的那批开关固化进脚本 ----
# 起因（Hunyuan H7）：§84 的长跑是**在命令行里显式 export** 的，不是走这个脚本的默认值。
# 后来者照脚本复现就会丢掉三条观测通道 + 显式初始化，soak#6 那种
# "6 个 actor 同停、5 进程全活、零崩溃栈"的情况没法再取证。
# 这里改成脚本负责，且 RL 初始化权重**必须显式给**：
#   #47 修的就是"不设 MAS_RL_INIT_MODEL 时静默加载字典序最大的 checkpoint"
#   （可能是个坏 epoch），所以这里不给默认值，直接报错退出，避免又一次静默。
export MAS_RL_INIT_MODEL=${MAS_RL_INIT_MODEL:?必须显式指定 SL 初始化权重，例如 MAS_RL_INIT_MODEL=model/BEST_SL_v82_soup_prior3.pth（不设会静默加载字典序最大的 checkpoint）}
export MAS_STATIC_NUM=${MAS_STATIC_NUM:-4}
export MAS_HEARTBEAT_SEC=${MAS_HEARTBEAT_SEC:-60}
export MAS_HB_FILE=${MAS_HB_FILE:-$OUT/heartbeat.txt}
export MAS_FAULT_FILE=${MAS_FAULT_FILE:-$OUT/fault.txt}
export MAS_ACTOR_TRACE=${MAS_ACTOR_TRACE:-$OUT/trace.txt}

# 避开 cuda:0（实测偏慢 20~45%）；4 张卡都可见，白名单从 1 开始
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAS_GPU_LIST=1,2,3

echo "=== soak tag=$TAG secs=$SECS batch=$BATCH par=$PAR act=$ACT on_server=$ONSRV kl_mode=$MAS_KL_MODE notify=$MAS_NOTIFY ==="
echo "=== init=$MAS_RL_INIT_MODEL static_num=$MAS_STATIC_NUM hb=${MAS_HEARTBEAT_SEC}s ==="
echo "=== hb_file=$MAS_HB_FILE fault_file=$MAS_FAULT_FILE trace=$MAS_ACTOR_TRACE ==="
echo "=== start $(date '+%F %T') ==="

# 清场：上一轮如果异常退出，SC2 会留着占内存和端口
pkill -f SC2_x64 2>/dev/null
pkill -f run_rl.py 2>/dev/null
sleep 2

python experiments/_soak_monitor.py --out "$OUT/monitor.csv" --interval 10 \
    --duration $((SECS + 300)) > "$OUT/monitor.log" 2>&1 &
MON=$!

# 2026-11-23（HY 建议）：记录本轮前后的 GPU 层错误计数。
# `Xid 43 = GPU stopped processing`，触发后任何 CUDA 调用会永久挂起且不抛 Python 异常——
# 这正是「六号僵死」的物理候选。若某轮复现僵死且计数增加，假设即被直接证实。
XID_BEFORE=$(dmesg 2>/dev/null | grep -ci "xid" || true)
START=$(date +%s)
# 第 84 轮（DS，#71）：`>` 改 `>>`。README #53 说过要改但一直没改：
# python 父进程与 SC2 子进程共享同一个 stdout fd、各自独立偏移，
# `>` 会让两边的写入互相覆盖，事后 grep 不到关键行。
python run_rl.py >> "$OUT/run.log" 2>&1
RC=$?
END=$(date +%s)
XID_AFTER=$(dmesg 2>/dev/null | grep -ci "xid" || true)
if [ "${XID_AFTER:-0}" != "${XID_BEFORE:-0}" ]; then
    {
        echo "[xid] 本轮期间出现新的 GPU 错误：$XID_BEFORE -> $XID_AFTER"
        dmesg 2>/dev/null | grep -i xid | tail -3
    } >> "$OUT/xid.log" 2>&1
    echo "[xid] 见 $OUT/xid.log（$XID_BEFORE -> $XID_AFTER）"
fi

kill "$MON" 2>/dev/null
wait "$MON" 2>/dev/null

echo "rc=$RC wall=$((END - START))" > "$OUT/summary.txt"
echo "=== end $(date '+%F %T') rc=$RC wall=$((END - START))s ==="

# 收尾：不管训练怎么退的，SC2 都不能留
pkill -f SC2_x64 2>/dev/null
exit $RC
