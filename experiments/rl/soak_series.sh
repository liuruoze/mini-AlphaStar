#!/bin/bash
# 连续 soak 监督器（2026-11-22 夜）
#
# 目的：把"跑一次看看"变成"连续暴露"。每轮 50 分钟，异常自动进入下一轮，
# 全程带 batch server 看门狗（300 s）与自愈开关，卡死时留下栈并自动退出。
#
# 轮次配置见 ROUNDS：前 5 轮生产默认，第 6/7 轮降并发做对照（文档建议 #3），
# 第 8 轮回到默认收尾。
set -u
ROOT=/tmp/series
mkdir -p "$ROOT"
cd /home/liuruoze/mini-AlphaStar || exit 9

export MAS_RL_INIT_MODEL=model/sl/cwH5h_s20261207/sl_26-11-15_09-37-33_best.pth
export MAS_BS_STALL_WATCH=1
export MAS_BS_STALL_EXIT=1
export MAS_BS_STALL_SEC=300

ROUNDS=(
  "r1 15 2"
  "r2 15 2"
  "r3 15 2"
  "r4 15 2"
  "r5 15 2"
  "r6 8 2"
  "r7 15 1"
  "r8 15 2"
)

cleanup_gpu() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill "$p" 2>/dev/null
  done
  sleep 6
}

: > "$ROOT/summary.csv"
echo "tag,parallel,actors,wall,rc,stalls,vsteps_last" >> "$ROOT/summary.csv"

for spec in "${ROUNDS[@]}"; do
  set -- $spec
  TAG=$1; PAR=$2; ACT=$3
  OUT=$ROOT/$TAG
  mkdir -p "$OUT"
  echo "[series] $(date '+%F %T') 开始 $TAG (par=$PAR act=$ACT)" >> "$ROOT/series.log"

  cleanup_gpu

  # 70 分钟上限：正常 50 分钟结束，卡死由看门狗 300 秒后自愈退出
  timeout 4200 experiments/_soak_run.sh "$TAG" 3000 1 "$PAR" "$ACT" 1 > "$OUT/driver.log" 2>&1
  RC=$?

  SLOG=$HOME/mini-AlphaStar/log/soak_$TAG
  # summary.txt 形如 "rc=0 wall=3073"；按空格切再取 = 后面的值，
  # 不能按 = 切（字段会错位成 rc/0/wall/3073，$3 是字符串 "wall"）。
  WALL=$(awk '/^rc=/{print $3}' "$SLOG/summary.txt" 2>/dev/null | cut -d= -f2)
  STALLS=$(grep -c "\[stall\]" "$SLOG/run.log" 2>/dev/null || echo 0)
  VSTEP=$(grep -h "v_steps" "$SLOG/heartbeat.txt" 2>/dev/null | tail -1 | tr -d '\n' | sed -E 's/.*v_steps=([0-9]+).*/\1/')
  [ -z "${WALL:-}" ] && WALL="na"
  echo "$TAG,$PAR,$ACT,${WALL},$RC,$STALLS,${VSTEP:-na}" >> "$ROOT/summary.csv"
  echo "[series] $(date '+%F %T') 结束 $TAG rc=$RC wall=${WALL} stalls=$STALLS vsteps=${VSTEP:-na}" >> "$ROOT/series.log"

  # 若出现过卡死，把现场单独留档
  if [ "${STALLS:-0}" -gt 0 ]; then
    mkdir -p "$ROOT/$TAG/evidence"
    cp "$SLOG"/fault.txt "$ROOT/$TAG/evidence/" 2>/dev/null
    cp -r "$SLOG"/stacks "$ROOT/$TAG/evidence/" 2>/dev/null
    tail -200 "$SLOG/run.log" > "$ROOT/$TAG/evidence/run_tail.log" 2>/dev/null
  fi

  cleanup_gpu
done

echo "[series] $(date '+%F %T') 全部完成" >> "$ROOT/series.log"
