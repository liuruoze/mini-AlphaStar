#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把张量数据集转成 fp16 存储（默认不改任何训练行为，需要配合 MAS_SL_FP16 使用）。

动机（2026-11-14 R3）：
  实测单帧 1.376MB(fp32)，a3nc 60 文件 ≈ 44GB 常驻内存，直接把训练并行度锁死在 2
  （单 run RSS 44.4G，125G 内存最多 2 个）。存储改 fp16 后 ≈22GB，可并行 4–5 个。
  标签是 0/1 one-hot，fp16 无损；特征只损失约 1e-3 相对精度（已在 1 epoch 对照里验证）。

用法：
  python3 experiments/to_fp16.py --src data/replay_data_a3nc --dst data/replay_data_a3nc_fp16
  python3 experiments/to_fp16.py --src ... --dst ... --limit 6   # 只转前 6 个（冒烟）
"""
import argparse
import glob
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 个文件（0=全部）")
    ap.add_argument("--force", action="store_true", help="目标已存在也重转")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)
    files = sorted(glob.glob(os.path.join(src, "*.pt")))
    if not files:
        raise SystemExit("[fp16] 源目录没有 *.pt: %s" % src)
    if args.limit:
        files = files[:args.limit]
    os.makedirs(dst, exist_ok=True)

    t0 = time.time()
    n_frames = 0
    n_done = 0
    n_skip = 0
    src_bytes = 0
    dst_bytes = 0
    for p in files:
        out = os.path.join(dst, os.path.basename(p))
        if os.path.exists(out) and not args.force:
            n_skip += 1
            continue
        f, l = torch.load(p, map_location="cpu")
        src_bytes += f.numel() * f.element_size()
        f16 = f.half()
        l16 = l.half()
        torch.save((f16, l16), out)
        dst_bytes += f16.numel() * f16.element_size()
        n_frames += int(f.shape[0])
        n_done += 1
        if n_done % 10 == 0 or n_done == len(files):
            print("[fp16] %d/%d frames=%d elapsed=%.0fs" % (n_done, len(files), n_frames, time.time() - t0),
                  flush=True)

    # 继承源 manifest 的关键标记（dagger 决定训练器是否允许启动，必须带过来）
    src_manifest = {}
    smpath = os.path.join(src, "manifest.json")
    if os.path.exists(smpath):
        try:
            src_manifest = json.load(open(smpath))
        except Exception:
            src_manifest = {}

    manifest = {
        "dataset": os.path.basename(dst),
        "source": src,
        "dtype": "float16",
        "files": len(files),
        "converted": n_done,
        "skipped": n_skip,
        "frames_converted": n_frames,
        "note": "fp16 存储；训练需 upcast 到 fp32（_WindowSubset 自动做）",
    }
    for k in ("dagger", "round", "expert", "expert_files", "expert_frames",
              "base_frames", "base", "old", "new", "a3nc", "human"):
        if k in src_manifest:
            manifest[k] = src_manifest[k]
    manifest["source_note"] = src_manifest.get("note", "")
    mpath = os.path.join(dst, "manifest.json")
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath))
            manifest["files"] = max(int(old.get("files", 0)), len(files))
            manifest["frames_converted"] = int(old.get("frames_converted", 0)) + n_frames
        except Exception:
            pass
    json.dump(manifest, open(mpath, "w"), ensure_ascii=False, indent=2)

    print("[fp16] done files=%d skip=%d frames=%d  src=%.1fGB dst=%.1fGB ratio=%.2f time=%.0fs"
          % (n_done, n_skip, n_frames, src_bytes / 1e9, dst_bytes / 1e9,
             (dst_bytes / src_bytes) if src_bytes else 0, time.time() - t0))


if __name__ == "__main__":
    main()
