#!/bin/bash
# 全量回归（第 59 轮，DeepSeek）。
#
# 为什么单独写成脚本：
#   这 14 个脚本各自需要 `ulimit -n 65536` / OMP 单线程 / SC2PATH / DISPLAY，
#   一个个手敲极易漏环境（漏了 ulimit 会报 `unable to open shared memory object`，
#   看起来像显存/内存问题，其实是 fd 用尽 —— 第 10 段的教训）。
#
# 判据：每个脚本的 **rc == 0**。`rc != 0` 一律视为失败，
#       **不看日志里有没有 "PASS"**（第 2 类错误：脚本自己 except 吞掉异常也能打印 PASS）。
set -u
cd "$HOME/mini-AlphaStar" || exit 9
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate th_1_7

export DISPLAY=:1024
export SC2PATH=$HOME/StarCraftII
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -n 65536
mkdir -p log

# 顺序有意为之：先纯函数/数值（快、无 env），再 env/多进程（慢、依赖 SC2）
# 第 70 轮加入 _test_r70_mapping：批内 actor 映射 + hidden 生命周期契约（纯 CPU，秒级）
TESTS="_test_t0_t1 _test_r70_mapping _test_r71_pack _test_r72_notify _test_r73_snapshot _test_r73_modes _test_r74_mask _audit_batched_hidden _audit_behavior_logits _audit_r14_fixes _test_r15_client _test_t9_trajectory_to_loss _test_t5_restart _test_r19_worker_obs"

FAIL=0
for t in $TESTS; do
    START=$(date +%s)
    python "experiments/$t.py" > "log/reg_$t.log" 2>&1
    RC=$?
    END=$(date +%s)
    if [ "$RC" -eq 0 ]; then
        printf "%-32s rc=0   %3ss  OK\n" "$t" "$((END - START))"
    else
        printf "%-32s rc=%-3s %3ss  **FAIL**（见 log/reg_%s.log）\n" "$t" "$RC" "$((END - START))" "$t"
        FAIL=$((FAIL + 1))
    fi
done

echo "----"
echo "失败 $FAIL / $(echo $TESTS | wc -w)"
pkill -f "[S]C2_x64" 2>/dev/null
exit $FAIL
