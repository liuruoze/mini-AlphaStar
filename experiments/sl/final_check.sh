#!/bin/bash
# HY25 / O7：09:00 收尾一键核对（只读，不改任何配置）。
# 用法：bash experiments/sl/final_check.sh > log/final_check_0900.log 2>&1
set -u
cd "$HOME/mini-AlphaStar"
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python

echo "==================== 09:00 收尾核对 ===================="
echo "date: $(date)"
echo
echo "--- 现场 ---"
echo -n "训练进程: "; ps -eo args | grep -c "[r]un_sl.py" || true
echo -n "评测进程: "; ps -eo args | grep -c "[e]val_sl_off_rl.py" || true
echo "GPU:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo -n "tmux: "; tmux ls 2>/dev/null | tr "\n" ";" || true
echo
echo "磁盘:"; df -h /home | tail -1
echo
echo "--- 结果一致性（按 name+run 精确复核） ---"
"$PY" experiments/_research/_verify_results.py | tail -4
echo
echo "--- 全配置总表（数字来自 JSON） ---"
"$PY" experiments/_research/overnight_summary.py | sed -n '1,20p'
echo
echo "--- 脚本语法 ---"
ok=0; bad=0
for f in experiments/*.sh; do
  bash -n "$f" 2>/dev/null && ok=$((ok + 1)) || { echo "  bash FAIL $f"; bad=$((bad + 1)); }
done
echo "  bash -n OK=$ok FAIL=$bad"
"$PY" -m py_compile experiments/*.py 2>/dev/null && echo "  py_compile 主仓 OK" || echo "  py_compile 主仓有失败"
(cd "$HOME/mini-AlphaStar-off" && "$PY" -m py_compile experiments/*.py 2>/dev/null) \
  && echo "  py_compile off 仓 OK" || echo "  py_compile off 仓有失败"
echo
echo "--- 两仓关键文件 md5（远端；本地需另行比对） ---"
md5sum alphastarmini/core/sl/sl_train_by_tensor.py run_data_env.sh \
       alphastarmini/core/arch/entity_encoder.py
echo
echo "--- 数据资产 ---"
echo -n "bench_results JSON: "; ls bench_results/*.json | wc -l
echo -n "doc/batch_acting/data CSV: "; ls doc/batch_acting/data/*.csv | wc -l
echo -n "flipx 数据: "; du -sh data/replay_data_v3_90_nonop_flipx 2>/dev/null | cut -f1
echo -n "up512 数据: "; du -sh data/replay_data_v3_90_nonop_512 2>/dev/null | cut -f1
echo
echo "==================== 收尾核对结束 ===================="
