#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 128 实体布局的 nonop .pt 纯张量升维到 512 布局（路径 A，不重转 replay）。

特征： [scalar 8020 | entity 128*1856 | map 98304] -> entity 段补零到 512*1856
标签： action 564 | delay 128 | queue 2 | select_units [12,128] | target_unit [128] | loc 4096
      -> select_units [12,512]、target_unit [512]，并把 NONE/padding 索引 127 重映射到 511
      （label.py:139-141 证明 select_units 的排列是 [max_selected, max_entities]）

用法：
    python experiments/sl/upscale_512.py --src data/replay_data_v3_90_nonop \
        --dst data/replay_data_v3_90_nonop_512 [--limit 2] [--verify]
"""
import argparse
import glob
import os

import torch

SC, ENT, MP = 8020, 128 * 1856, 98304
ACT, DLY, QUE, SU, TU = 564, 128, 2, 128 * 12, 128
MAXSEL, OLD_E, NEW_E = 12, 128, 512
FEAT_NEW, LABEL_NEW = 8020 + 512 * 1856 + 98304, 564 + 128 + 2 + 512 * 12 + 512 + 4096


def convert(x, y):
    n = x.shape[0]
    assert x.shape[1] == SC + ENT + MP, x.shape
    assert y.shape[1] == ACT + DLY + QUE + SU + TU + 4096, y.shape

    # --- feature: entity 段尾部补零 ---
    sc = x[:, :SC]
    ent = x[:, SC:SC + ENT].reshape(n, OLD_E, 1856)
    mp = x[:, SC + ENT:]
    ent_new = torch.zeros(n, NEW_E, 1856, dtype=x.dtype)
    ent_new[:, :OLD_E, :] = ent
    x_new = torch.cat([sc, ent_new.reshape(n, -1), mp], dim=1)

    # --- label: padding 索引 127 -> 511 ---
    a = y[:, :ACT]
    d = y[:, ACT:ACT + DLY]
    q = y[:, ACT + DLY:ACT + DLY + QUE]
    o = ACT + DLY + QUE
    su = y[:, o:o + SU].reshape(n, MAXSEL, OLD_E)
    tu = y[:, o + SU:o + SU + TU]
    loc = y[:, o + SU + TU:]

    su_new = torch.zeros(n, MAXSEL, NEW_E, dtype=y.dtype)
    su_new[:, :, :OLD_E - 1] = su[:, :, :OLD_E - 1]
    su_new[:, :, NEW_E - 1] = su[:, :, OLD_E - 1]

    tu_new = torch.zeros(n, NEW_E, dtype=y.dtype)
    tu_new[:, :OLD_E - 1] = tu[:, :OLD_E - 1]
    tu_new[:, NEW_E - 1] = tu[:, OLD_E - 1]

    y_new = torch.cat([a, d, q, su_new.reshape(n, -1), tu_new, loc], dim=1)
    assert x_new.shape[1] == FEAT_NEW and y_new.shape[1] == LABEL_NEW
    return x_new, y_new


def check(x, y, xn, yn):
    """逐项验证变换没有丢信息 / 没有错位。"""
    n = x.shape[0]
    assert torch.equal(xn[:, :SC], x[:, :SC]), "scalar 段变了"
    assert torch.equal(xn[:, SC + 512 * 1856:], x[:, SC + ENT:]), "map 段变了"
    assert torch.equal(xn[:, SC:SC + ENT].reshape(n, OLD_E, 1856),
                       x[:, SC:SC + ENT].reshape(n, OLD_E, 1856)), "entity 前128槽变了"
    assert xn[:, SC + ENT:SC + 512 * 1856].abs().sum() == 0, "新增槽非零"
    o = ACT + DLY + QUE
    su_old = y[:, o:o + SU].reshape(n, MAXSEL, OLD_E)
    su_new = yn[:, o:o + 512 * MAXSEL].reshape(n, MAXSEL, NEW_E)
    assert torch.equal(su_new[:, :, :127], su_old[:, :, :127]), "select 前127维变了"
    assert torch.equal(su_new[:, :, 511], su_old[:, :, 127]), "select padding 未重映射"
    assert float(su_new.sum(2).min()) == 1.0 and float(su_new.sum(2).max()) == 1.0, "select 非 one-hot"
    tu_old = y[:, o + SU:o + SU + TU]
    tu_new = yn[:, o + 512 * MAXSEL:o + 512 * MAXSEL + NEW_E]
    assert torch.equal(tu_new[:, :127], tu_old[:, :127]), "target_unit 前127维变了"
    assert torch.equal(tu_new[:, 511], tu_old[:, 127]), "target_unit padding 未重映射"
    assert float(tu_new.sum(1).min()) == 1.0 and float(tu_new.sum(1).max()) == 1.0, "tu 非 one-hot"
    assert torch.equal(yn[:, o + 512 * MAXSEL + NEW_E:], y[:, o + SU + TU:]), "location 段变了"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（冒烟用）")
    ap.add_argument("--verify", action="store_true", help="逐文件做完整性校验")
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.src, "*.pt")))
    if args.limit:
        fs = fs[:args.limit]
    os.makedirs(args.dst, exist_ok=True)
    print("src=%s  files=%d  dst=%s" % (args.src, len(fs), args.dst), flush=True)
    total = 0
    for i, f in enumerate(fs, 1):
        x, y = torch.load(f, map_location="cpu")
        xn, yn = convert(x, y)
        if args.verify:
            check(x, y, xn, yn)
        out = os.path.join(args.dst, os.path.basename(f))
        torch.save((xn, yn), out)
        total += os.path.getsize(out)
        if i % 10 == 0 or i == len(fs):
            print("  [%3d/%3d] %-42s %6d frames  %.2f GB written"
                  % (i, len(fs), os.path.basename(f), x.shape[0], total / 1024 ** 3),
                  flush=True)
        del x, y, xn, yn
    print("DONE files=%d total=%.2f GB" % (len(fs), total / 1024 ** 3), flush=True)


if __name__ == "__main__":
    main()
