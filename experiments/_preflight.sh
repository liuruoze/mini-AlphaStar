#!/usr/bin/env bash
# 共用前置检查库（被 train_bvb_m30.sh / run_eval_batch.sh / run_p8_eval.sh source）
# 目的：把"跑到一半才炸"变成"启动前就说清楚"，返回码统一 9 = 前置条件不满足。
#
# 提供：
#   pf_disk_gb <min_gb>             /home 可用空间
#   pf_ram_gb  <min_gb>             MemAvailable
#   pf_train_slots <max>            正在跑的 run_sl.py 数量（默认上限 2）
#   pf_gpu_free_mb <idx> <min_mb>   指定 GPU 的空闲显存
#   pf_need_file <path> <desc>
#   pf_die <msg>
#
# 可用 MAS_PF_SKIP=1 全部跳过（仅用于明确知道后果的场合）。

PF_TAG=${PF_TAG:-pf}

pf_die() { echo "[$PF_TAG][FATAL] $*" >&2; exit 9; }

_pf_skip() { [ "${MAS_PF_SKIP:-0}" = "1" ]; }

pf_disk_gb() {
  _pf_skip && return 0
  local need=${1:-30}
  local avail
  avail=$(df -BG --output=avail /home 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -z "${avail:-}" ] && { echo "[$PF_TAG][warn] 无法读取磁盘可用空间，跳过检查" >&2; return 0; }
  if [ "$avail" -lt "$need" ]; then
    pf_die "磁盘可用 ${avail}G < 要求 ${need}G（/home）。先清中间数据再跑。"
  fi
  echo "[$PF_TAG] disk=${avail}G ok (>=${need}G)"
}

pf_ram_gb() {
  _pf_skip && return 0
  local need=${1:-55}
  local avail
  avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "${avail:-0}" -lt "$need" ]; then
    pf_die "可用内存 ${avail}G < 要求 ${need}G（单 run ≈45G）。等其它任务结束或减少并行。"
  fi
  echo "[$PF_TAG] ram=${avail}G ok (>=${need}G)"
}

pf_train_slots() {
  _pf_skip && return 0
  local max=${1:-2}
  local n
  n=$(ps -eo cmd | grep -c '[r]un_sl.py')
  if [ "$n" -ge "$max" ]; then
    pf_die "已有 ${n} 个训练在跑（上限 ${max}，45G/个）。要强跑请设 MAS_ALLOW_MORE=1。"
  fi
  echo "[$PF_TAG] train_slots=${n}/${max} ok"
}

pf_gpu_free_mb() {
  _pf_skip && return 0
  local idx=${1:-0} need=${2:-4000}
  local free
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', *' -v i="$idx" '$1==i{print $2}')
  [ -z "${free:-}" ] && { echo "[$PF_TAG][warn] 读不到 GPU${idx} 显存，跳过" >&2; return 0; }
  if [ "$free" -lt "$need" ]; then
    pf_die "GPU${idx} 空闲显存 ${free}MiB < 要求 ${need}MiB。"
  fi
}

pf_need_file() {
  _pf_skip && return 0
  local p=${1:?path} d=${2:-文件}
  [ -e "$p" ] || pf_die "$d 不存在: $p"
}

# 数据集检查：目录存在且至少有一个 .pt
pf_need_dataset() {
  _pf_skip && return 0
  local p=${1:?data dir}
  [ -d "$p" ] || pf_die "数据目录不存在: $p"
  local n
  n=$(ls -1 "$p"/*.pt 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || pf_die "数据目录里没有 *.pt: $p"
  echo "[$PF_TAG] dataset=$p files=$n ok"
}
