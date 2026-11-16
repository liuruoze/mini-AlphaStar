#!/usr/bin/env bash
# 推送前基准体检：真实跑一遍训练 + 评测（默认约 6 分钟），确认代码能跑通。
#
# 与 selfcheck.sh 分工：selfcheck 只做静态检查（shell 语法 / py_compile / 开关文档一致性 /
# 离线单测 / 两仓脚本一致性），抓不到"单测全过但真跑就崩"的问题。这里跑真实张量与真实 SC2。
#
# 用法: bash experiments/smoke_bench.sh [all|train|eval]
#
# 可调（环境变量）:
#   SMOKE_DATA    训练数据目录        默认 data/replay_data_a3nc
#   SMOKE_FILES   参与训练的文件数    默认 20（按 VAL_FRAC=0.2 切 16 训练 / 4 验证）
#   SMOKE_EPOCHS  训练 epoch 数       默认 2
#   SMOKE_CKPT    评测权重            默认 model/sl/cwH5h_s20261207/*_best.pth（已知最强）
#   SMOKE_SEEDS   评测种子（空格分隔）默认 "3038 3039 3040 3045"
#   SMOKE_DIFF    评测难度            默认 3
#   SMOKE_GPUS_T  训练用 GPU          默认 0
#   SMOKE_GPUS_E  评测用 GPU 列表     默认 1,2,3
#
# 判据（全部满足才 exit 0）:
#   训练: rc=0 / 生成 *_best.pth / 日志含 val_action_acc / train_manifest.json 存在
#   评测: rc=0 / 每个种子都产出结果文件且首行含 outcome 字段
#   收尾: 无 SC2 孤儿进程残留
#
# 说明: 数据集 manifest 声明 "dagger": true 时自动补 MAS_DAGGER=1。
#       训练器对 dagger 数据集有硬闸门（数据集 dagger=1 而 MAS_DAGGER=0 直接拒绝启动），
#       这个脚本替调用方从 manifest 推断，避免"照用法敲一遍就报错"。
#       训练与评测分别落在 model/sl/_SMOKE_TRAIN 与 model/sl/_SMOKE_EVAL，
#       并断言评测目录里恰好 1 个 *_best.pth、软链解析到指定权重
#       （同目录会让 run_p8_eval.sh 按字母序挑到刚训出的模型，已实测踩过）。
set -u

ROOT=/home/liuruoze/mini-AlphaStar
OFFROOT=/home/liuruoze/mini-AlphaStar-off
cd "$ROOT" || exit 1
export DISPLAY=:1024 SC2PATH=/home/liuruoze/StarCraftII
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MODE=${1:-all}
DATA=${SMOKE_DATA:-$ROOT/data/replay_data_a3nc}
FILES=${SMOKE_FILES:-20}
EPOCHS=${SMOKE_EPOCHS:-2}
SEEDS=${SMOKE_SEEDS:-"3038 3039 3040 3045"}
DIFF=${SMOKE_DIFF:-3}
GPUT=${SMOKE_GPUS_T:-0}
GPUE=${SMOKE_GPUS_E:-1,2,3}
TAG_T=_SMOKE_TRAIN
TAG_E=_SMOKE_EVAL
CAND=${SMOKE_CKPT:-}
if [ -z "$CAND" ]; then
  CAND=$(ls -1 model/sl/cwH5h_s20261207/*_best.pth 2>/dev/null | head -1)
fi

FAIL=0
say() { echo "[smoke] $*"; }
bad() { echo "[smoke][FAIL] $*"; FAIL=1; }

DAGGER=0
if [ -f "$DATA/manifest.json" ] && grep -q '"dagger"[[:space:]]*:[[:space:]]*true' "$DATA/manifest.json" 2>/dev/null; then
  DAGGER=1
fi

t0=$(date +%s)

if [ "$MODE" = "all" ] || [ "$MODE" = "train" ]; then
  say "训练基准: files=$FILES epochs=$EPOCHS dagger=$DAGGER data=$(basename "$DATA") gpu=$GPUT"
  rm -rf "model/sl/$TAG_T"
  mkdir -p "model/sl/$TAG_T"
  DATA="$DATA" MAS_SL_MAX_FILES="$FILES" MAS_SL_EPOCHS="$EPOCHS" \
  MAS_ON_SERVER=1 MAS_DAGGER="$DAGGER" GPU="$GPUT" \
    bash experiments/train_bvb_m30.sh "$TAG_T" "$EPOCHS" 20261116 3 > /tmp/smoke_train.out 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    bad "训练 rc=$rc；尾部：$(tr -d '\000' < /tmp/smoke_train.out | tail -1)"
  fi
  n=$(ls -1 "model/sl/$TAG_T"/*_best.pth 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] || bad "没有生成 *_best.pth"
  v=$(tr -d '\000' < "log/train_$TAG_T.log" 2>/dev/null | grep -oE "best val_action_acc=[0-9.]+" | tail -1)
  [ -n "$v" ] || bad "日志里没有 val_action_acc"
  [ -f "model/sl/$TAG_T/train_manifest.json" ] || bad "缺 train_manifest.json"
  say "训练结果: rc=$rc ckpt=$n ${v:-val=NA}"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "eval" ]; then
  if [ -z "$CAND" ] || [ ! -f "$CAND" ]; then
    bad "找不到评测权重（用 SMOKE_CKPT 指定一个 .pth）"
  else
    CAND_ABS=$(cd "$(dirname "$CAND")" && pwd)/$(basename "$CAND")
    # 评测必须用**独立目录**：run_p8_eval.sh 取 `ls *_best.pth | head -1`，
    # 若与训练烟测共用目录，会按字母序挑到刚训出来的 2-epoch 废模型（已实测踩过）。
    rm -rf "model/sl/$TAG_E"
    mkdir -p "model/sl/$TAG_E"
    ln -sf "$CAND_ABS" "model/sl/$TAG_E/smoke_20261116_best.pth"
    nb=$(ls -1 "model/sl/$TAG_E"/*_best.pth 2>/dev/null | wc -l)
    [ "$nb" -eq 1 ] || bad "评测目录里有 $nb 个 *_best.pth，必须恰好 1 个"
    got=$(readlink -f "model/sl/$TAG_E"/*_best.pth 2>/dev/null)
    [ "$got" = "$CAND_ABS" ] || bad "评测权重解析到 $got，期望 $CAND_ABS"
    rm -f "$OFFROOT"/outputs/${TAG_E}s*.txt 2>/dev/null
    say "评测基准: seeds=[$SEEDS] diff=D$DIFF ckpt=$(basename "$got") gpus=$GPUE"
    OFF_GPUS="$GPUE" OFF_MAX_JOBS=$(echo "$SEEDS" | wc -w) OFF_GLOBAL_MAX=22 \
    OFF_DIFFICULTY="$DIFF" OFF_GAME_STEPS=40000 \
      bash experiments/run_p8_eval.sh "$TAG_E" $SEEDS > /tmp/smoke_eval.out 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || bad "评测 rc=$rc；尾部：$(tail -1 /tmp/smoke_eval.out)"
    miss=0
    for s in $SEEDS; do
      f="$OFFROOT/outputs/${TAG_E}s${s}_${s}.txt"
      if [ -s "$f" ] && head -1 "$f" | grep -q "outcome"; then :; else miss=$((miss + 1)); fi
    done
    [ "$miss" -eq 0 ] || bad "有 $miss 个种子没产出结果文件"
    games=$(ls "$OFFROOT"/outputs/${TAG_E}s*_*.txt 2>/dev/null | wc -l)
    wins=$(grep -l "outcome: +1" "$OFFROOT"/outputs/${TAG_E}s*_*.txt 2>/dev/null | wc -l)
    say "评测结果: rc=$rc 局数=$games 官方判胜=$wins"
  fi
fi

# 收尾：SC2 在对局结束后会延迟几秒才真正退出，先等，再清孤儿。
# 注意 clean_sc2_orphans.sh 默认 dry-run，必须带 --yes 才真杀；它自带"有评测在跑就不杀"的保护。
o=$(pgrep -x SC2_x64 2>/dev/null | wc -l)
for _ in 1 2 3 4 5 6; do
  [ "$o" -eq 0 ] && break
  sleep 2
  o=$(pgrep -x SC2_x64 2>/dev/null | wc -l)
done
if [ "$o" -ne 0 ] && [ -x experiments/clean_sc2_orphans.sh ]; then
  bash experiments/clean_sc2_orphans.sh --yes >/dev/null 2>&1 || true
  sleep 2
  o=$(pgrep -x SC2_x64 2>/dev/null | wc -l)
fi
[ "$o" -eq 0 ] || bad "仍有 $o 个 SC2 孤儿进程"

t=$(( $(date +%s) - t0 ))
if [ "$FAIL" -eq 0 ]; then
  say "PASS（耗时 ${t}s）"
else
  say "FAIL（耗时 ${t}s）"
fi
exit "$FAIL"
