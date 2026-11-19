#!/usr/bin/env bash
# 一条命令输出所有需要看的东西：时间/训练曲线/权重/评测/进程/GPU/内存
# 用法: bash experiments/status_snapshot.sh <tag1> [tag2 ...]
cd ~/mini-AlphaStar || exit 1
echo "TIME $(date +%H:%M)"
echo "TRAIN $(pgrep -fc "run_sl.py" || echo 0)  EVAL $(pgrep -fc "run_rl_eval.py" || echo 0)"
for t in "$@"; do
  acc=$(grep "Val action acc" log/train_$t.log 2>/dev/null | tail -1 | awk "{print \$4}")
  ck=$(ls model/sl/$t/*.pth 2>/dev/null | wc -l)
  echo "RUN $t val_acc=${acc:-none} ckpt=$ck"
done
echo "GPU $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr "\n" " ")"
free -g | awk "/Mem:/{print \"RAM used=\" \$3 \"G free=\" \$7 \"G\"}"
