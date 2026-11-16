#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 e3_*/nonop/*.pt（只去了 no_op，保留其余动作）汇总成训练数据集。

用法: python build_nonoop_ds.py --src-glob '/home/liuruoze/bvb/onpolicy/e3_*/nonop/*.pt' \
                                --dst  /home/liuruoze/mini-AlphaStar/data/replay_data_e3n90 \
                                --tag  e3n90
软链不复制；manifest 带 dagger=True（训练守卫要求）。
"""
import argparse
import glob
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-glob", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tag", default="e3n90")
    ap.add_argument("--skip-frames-check", action="store_true")
    a = ap.parse_args()

    fs = sorted(glob.glob(a.src_glob))
    if not fs:
        raise SystemExit("[ds] FAIL no src")
    os.makedirs(a.dst, exist_ok=True)
    for old in glob.glob(os.path.join(a.dst, "*.pt")):
        os.unlink(old)

    tot = 0
    n = 0
    for p in fs:
        if a.skip_frames_check:
            k = 0
        else:
            k = int(torch.load(p, map_location="cpu")[0].shape[0])
        tot += k
        n += 1
        dst = os.path.join(a.dst, os.path.basename(p))
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(os.path.abspath(p), dst)

    json.dump({"dataset": a.tag, "dagger": True, "round": 1,
               "expert": "V5 randomized fast_rush, expert self-play",
               "filter": "只去掉 no_op；Cancel/Attack/Harvest 全保留",
               "expert_files": n, "expert_frames": tot,
               "source": "bvb/onpolicy/e3_*/nonop"},
              open(os.path.join(a.dst, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    print("[ds] files=%d frames=%d mean=%.1f" % (n, tot, tot / max(n, 1)))


if __name__ == "__main__":
    main()
