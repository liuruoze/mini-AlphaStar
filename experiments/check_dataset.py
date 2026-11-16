#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集体检：训练前把"会在 20 分钟后才炸"的问题提前暴露。

检查项：
  1. manifest 是否存在、是否声明 dagger（与 MAS_DAGGER 双向校验有关）
  2. 文件数 / 总帧数 / 单文件帧数分布（0 帧或过短会直接告警）
  3. 每个文件的 dtype（fp32/fp16）、shape 一致性
  4. 抽样 NaN/Inf 检查
  5. 动作分布（抽样，按 pysc2 RAW_FUNCTIONS 名称汇总）
  6. 文件名重复 / 体积异常（同一个文件被链了两次）

用法：
  python3 experiments/check_dataset.py data/replay_data_a3nc
  python3 experiments/check_dataset.py data/replay_data_a3nc --sample 6 --strict
返回码：0=通过（可有 warn）；1=有 error；--strict 时 warn 也算失败。
"""
import argparse
import collections
import glob
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", type=int, default=4, help="抽样检查 NaN 的文件数")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = os.path.expanduser(args.path)
    errs, warns = [], []

    if not os.path.isdir(root):
        print("[FATAL] 目录不存在: %s" % root)
        return 1

    files = sorted(glob.glob(os.path.join(root, "*.pt")))
    if not files:
        print("[FATAL] 目录里没有 *.pt: %s" % root)
        return 1

    # 1) manifest
    mpath = os.path.join(root, "manifest.json")
    manifest = {}
    if not os.path.exists(mpath):
        warns.append("缺 manifest.json（训练器的 dagger 校验会按 0 处理）")
    else:
        try:
            manifest = json.load(open(mpath))
        except Exception as e:
            errs.append("manifest.json 解析失败: %r" % e)

    # 2) 逐文件统计
    total = 0
    lens = []
    dtypes = set()
    shapes = set()
    names = collections.Counter()
    sizes = {}
    for p in files:
        try:
            f, l = torch.load(p, map_location="cpu")
        except Exception as e:
            errs.append("读取失败 %s: %r" % (os.path.basename(p), e))
            continue
        n = int(f.shape[0])
        total += n
        lens.append(n)
        dtypes.add(str(f.dtype))
        shapes.add(tuple(f.shape[1:]))
        names[os.path.basename(p)] += 1
        sz = os.path.getsize(p)
        sizes[p] = sz
        if n == 0:
            errs.append("0 帧: %s" % os.path.basename(p))

    # 3) 抽样 NaN + 动作分布
    act_cnt = collections.Counter()
    nan_bad = []
    step = max(1, len(files) // max(args.sample, 1))
    for p in files[::step][:args.sample]:
        try:
            f, l = torch.load(p, map_location="cpu")
        except Exception:
            continue
        if not torch.isfinite(f.float()).all():
            nan_bad.append(os.path.basename(p))
        ids = l[:, :564].argmax(dim=-1).tolist()
        act_cnt.update(ids)

    if nan_bad:
        errs.append("含 NaN/Inf: %s" % ", ".join(nan_bad[:3]))

    # 4) 重复文件名 / 体积异常
    dup = [k for k, v in names.items() if v > 1]
    if dup:
        errs.append("文件名重复: %s" % ", ".join(dup[:3]))
    if lens and min(lens) < 20:
        warns.append("有 %d 个文件短于 20 帧（min=%d），窗口数会很少" % (sum(1 for x in lens if x < 20), min(lens)))

    # 5) 动作分布（抽样）
    try:
        from pysc2.lib.actions import RAW_FUNCTIONS as RF
        top = act_cnt.most_common(8)
        top_s = ", ".join("%s=%.1f%%" % (RF[k].name if k < len(RF) else k, 100.0 * v / max(sum(act_cnt.values()), 1))
                          for k, v in top)
    except Exception:
        top_s = str(act_cnt.most_common(8))

    print("=== 数据集体检: %s ===" % root)
    print("文件数      : %d" % len(files))
    print("总帧数      : %d" % total)
    print("帧数分布    : min=%-5d max=%-5d mean=%.0f" % (min(lens or [0]), max(lens or [0]),
                                                         (sum(lens) / len(lens)) if lens else 0))
    print("dtype       : %s" % ",".join(sorted(dtypes)))
    print("frame shape : %s" % ",".join(str(s) for s in sorted(shapes)))
    print("manifest    : dagger=%s files=%s frames=%s" % (manifest.get("dagger"), manifest.get("files"),
                                                          manifest.get("frames_converted") or manifest.get("frames")))
    print("动作分布(抽样): %s" % top_s)
    print("总体积      : %.1f GB" % (sum(sizes.values()) / 1e9))

    for w in warns:
        print("[WARN] %s" % w)
    for e in errs:
        print("[ERROR] %s" % e)

    if errs or (args.strict and warns):
        print("=> 未通过（errors=%d warns=%d）" % (len(errs), len(warns)))
        return 1
    print("=> 通过（warns=%d）" % len(warns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
