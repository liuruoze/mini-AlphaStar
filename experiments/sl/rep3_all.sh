#!/bin/bash
# 用户 2026-11-07 17:18 要求：每个训练 run 跑 3 局，验证「同一权重 + 固定 3038」是否逐局一致。
# 15 个训练 run（5 配置 × 3 run）× 3 局 = 45 局。
# 每局独立进程：run_eval_batch.sh 内部并发 3 个 eval_sl_off_rl.py。
# 分 5 批（每批 1 个配置的 3 个 run 并发 = 9 个 SC2 进程），避免一次 45 进程压垮 CPU。
# 与 BENCH v1 的 1 局评测保持完全相同的调用方式：不额外设置 OFF_MAX_ENTITIES 等环境变量。
set -u
cd "$HOME/mini-AlphaStar-off"
mkdir -p outputs log
DRV=log/rep3_driver.log
: > "$DRV"

NAMES=(B0_17 dropout01 wd1e4 B0_21 d130_eq)
GPULIST=("0,1,2" "1,2,3" "2,3,0")

echo "REP3 START total=15 runs x 3 games = 45 at $(date -Is)" >> "$DRV"
for b in 0 1 2 3 4; do
  n=${NAMES[$b]}
  pids=()
  for j in 0 1 2; do
    r=$((j + 1))
    dir="/home/liuruoze/mini-AlphaStar/model_bench_${n}_${r}/"
    tag="rep_${n}_${r}"
    g=${GPULIST[$j]}
    OFF_GPUS=$g bash run_eval_batch.sh "$dir" "$tag" 3038 3 3 > "log/${tag}.log" 2>&1 &
    pids+=($!)
    echo "  START batch=$b $tag gpus=$g pid=$! at $(date -Is)" >> "$DRV"
  done
  wait "${pids[@]}" 2>/dev/null || true
  echo "BATCH $b DONE $n (9 games) at $(date -Is)" >> "$DRV"
done
echo "ALL DONE 45 games at $(date -Is)" >> "$DRV"
