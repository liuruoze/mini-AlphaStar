#!/usr/bin/env bash
# 转换 <prefix> 的 replay -> raw -> nonop，转完立即删该 tag 的 raw（省一半空间）
# 用法: convert_self_clean.sh <prefix> [max_parallel_tags]
set -u
PREFIX=${1:?prefix}
N=${2:-4}
H="$HOME"
M="$H/mini-AlphaStar"
PY="$H/.conda/envs/th_1_7/bin/python"
R="$H/mini-AlphaStar-off/revised_simple64_replays"
i=0
for F in "$R"/${PREFIX}_*.SC2Replay; do
  [ -f "$F" ] || continue
  B="$(basename "$F" .SC2Replay)"
  T="$(echo "$B" | cut -d_ -f1-2)"
  D="$H/bvb/onpolicy/$T"
  [ -s "$D/nonop/$(echo "$B").pt" ] && continue
  mkdir -p "$D/raw" "$D/nonop"
  (
    bash "$M/experiments/convert_all.sh" "$T" 2
    "$PY" "$M/experiments/_mk_nonop.py" --src "$D/raw" --dst "$D/nonop"
    rm -rf "$D/raw"                       # 只留 nonop
    echo "[conv+] $T nonop=$(ls -1 "$D"/nonop/*.pt 2>/dev/null | wc -l) raw_deleted"
  ) &
  i=$((i+1))
  if [ $((i % N)) -eq 0 ]; then wait; fi
done
wait
echo "[conv+] ALL DONE prefix=$PREFIX new=$i nonop_total=$(ls -1 "$H"/bvb/onpolicy/${PREFIX}_*/nonop/*.pt 2>/dev/null | wc -l)"
