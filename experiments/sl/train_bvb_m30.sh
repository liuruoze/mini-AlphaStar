#!/usr/bin/env bash
# ⚠ 已废弃（2026-11-15）：本文件曾是训练入口的旧副本（1107 字节，缺前置检查/manifest/dry-run/退出码透传）。
# 真源：~/mini-AlphaStar/experiments/train_bvb_m30.sh
# 保留此转发是为了不让historically 写死这个路径的脚本静默跑到旧逻辑。
echo "[deprecated] experiments/sl/train_bvb_m30.sh 已废弃，转发到 experiments/train_bvb_m30.sh" >&2
exec bash "$HOME/mini-AlphaStar/experiments/train_bvb_m30.sh" "$@"
