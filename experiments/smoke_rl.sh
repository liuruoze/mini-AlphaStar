#!/usr/bin/env bash
# RL 冒烟：4 分钟跑通一次训练，确认「能更新、无异常、无 OOM」。
#
#   用法: bash experiments/smoke_rl.sh [输出目录]
#   返回: 0 = PASS，非 0 = FAIL（并打印原因）
#
# 判据（全部满足才算 PASS）：
#   1) `[PV] learner_update` 至少 1 行 —— 说明 optimizer.step() 真的走到了；
#   2) 日志里没有 Traceback / out of memory；
#   3) 进程正常退出（rc=0 或 timeout 的 124，都算正常收尾）。
#
# 为什么需要它：RL 的失败经常是"跑完了、但一次更新都没做"（数据不足、actor 全崩），
# 只看 rc 会误判成功。
set -u

R=${R:-$HOME/mini-AlphaStar}
W=${1:-$HOME/bvb/onpolicy/smoke_rl}
mkdir -p "$W"
cd "$R" || exit 9

INIT=$(ls model/sl/*/*_best.pth 2>/dev/null | head -1)
[ -z "${INIT:-}" ] && { echo "[smoke] 找不到 SL 初始化权重"; exit 9; }

export MAS_ON_SERVER=1 MAS_PARALLEL=1 MAS_ACTOR_NUMS=2 MAS_MAX_TIME=240 \
       MAS_STATIC_NUM=4 MAS_RL_DIFFICULTY=1 MAS_RL_SEED=7 MAS_PV_LOG=1 \
       MAS_RL_INIT_MODEL="$INIT" DISPLAY=:1024 SC2PATH=$HOME/StarCraftII \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

timeout 420 ~/.conda/envs/th_1_7/bin/python run_rl.py > "$W/smoke.log" 2>&1
rc=$?

pv=$(grep -c "\[PV\] learner_update" "$W/smoke.log" || true)
# tensorboardX 的写线程在进程被 terminate() 时会打一条 threading Traceback，
# 那是收尾噪声、不是训练故障（已用 writer.close() 缓解，但子进程被强杀时仍会出现）。
# 判据改为：Traceback 数量不得多于这类收尾噪声的数量。
tb=$(grep -ci "traceback" "$W/smoke.log" || true)
benign=$(grep -ci "event_file_writer.py" "$W/smoke.log" || true)
bad_tb=$((tb - benign)); [ "$bad_tb" -lt 0 ] && bad_tb=0
oom=$(grep -ci "out of memory" "$W/smoke.log" || true)
skips=$(grep -c "skip update" "$W/smoke.log" || true)

echo "[smoke] rc=$rc pv=$pv traceback=$tb(benign=$benign) oom=$oom skip_update=$skips log=$W/smoke.log"

fail=0
[ "$pv" -ge 1 ] || { echo "[smoke] FAIL: 没有发生任何 optimizer step"; fail=1; }
[ "$bad_tb" -eq 0 ] || { echo "[smoke] FAIL: 日志里有 $bad_tb 条非收尾 Traceback"; fail=1; }
[ "$oom" -eq 0 ] || { echo "[smoke] FAIL: 显存/内存不足"; fail=1; }
case "$rc" in 0|124) ;; *) echo "[smoke] FAIL: 退出码 $rc"; fail=1 ;; esac

if [ "$fail" -eq 0 ]; then
  echo "[smoke] PASS"
  exit 0
fi
exit 1
