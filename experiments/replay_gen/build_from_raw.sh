#!/usr/bin/env bash
# 从「已转换好的 raw/*.pt」继续：nonop -> 与 30 局基座混合 -> manifest
# 与 build_r1.sh 的差别：跳过 replay->pt 转换（转换慢时用于并行转换后的收尾）
# 用法: build_from_raw.sh <tag>
set -u
TAG=${1:?tag}
H=$HOME
M=$H/mini-AlphaStar
W=$H/bvb/onpolicy/$TAG
BASE=${BASE:-$M/data/bvb_m30_sub}
MIX=$M/data/replay_data_${TAG}
PY=/home/liuruoze/.conda/envs/th_1_7/bin/python

raw_n=$(ls -1 "$W"/raw/*.pt 2>/dev/null | wc -l)
echo "[build] raw_pt=$raw_n"
[ "$raw_n" -lt 1 ] && { echo "[build] FAIL no raw pt"; exit 2; }

"$PY" "$M/experiments/replay_gen/_mk_nonop.py" --src "$W/raw" --dst "$W/nonop" \
  > "$W/nonop.log" 2>&1
echo "[build] nonop_pt=$(ls -1 "$W"/nonop/*.pt 2>/dev/null | wc -l)"

rm -rf "$MIX"; mkdir -p "$MIX"
cp "$BASE"/*.pt "$MIX/"
for f in "$W"/nonop/*.pt; do
  [ -f "$f" ] && cp "$f" "$MIX/expert_$(basename "$f")"
done

"$PY" - "$MIX" "$TAG" <<'PY'
import glob, json, os, sys
import torch
d, tag = sys.argv[1], sys.argv[2]
fs = [p for p in glob.glob(d + "/*.pt") if not os.path.basename(p).startswith("expert_")]
efs = [p for p in glob.glob(d + "/*.pt") if os.path.basename(p).startswith("expert_")]

def nf(ps):
    return sum(torch.load(p, map_location="cpu")[0].shape[0] for p in ps)

a, e = nf(fs), nf(efs)
man = {"dataset": tag, "dagger": True, "round": 1,
       "expert": "v4 rule expert, concurrent dual-action on learner states (medium)",
       "source_nodagger": "bvb_m30_sub",
       "orig_files": len(fs), "orig_frames": int(a),
       "expert_files": len(efs), "expert_frames": int(e),
       "layout": "128", "transform_seed": 0,
       "note": "learner 与专家同一步双命令注入；全存 replay（不只胜局）"}
json.dump(man, open(d + "/manifest.json", "w"), ensure_ascii=False, indent=2)
print("[build] orig_frames=%d expert_frames=%d total=%d expert_pct=%.2f%%"
      % (a, e, a + e, 100.0 * e / max(a + e, 1)))
print("[build] GATE expert_pct>=5%%: %s"
      % ("PASS" if 100.0 * e / max(a + e, 1) >= 5 else "WARN(小样本)"))
PY

echo "[build] DONE mix=$MIX files=$(ls -1 "$MIX"/*.pt 2>/dev/null | wc -l)"
