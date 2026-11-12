#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bot vs Bot 数据门禁：训练前确认数据正确。

检查项（任一项 FAIL 就不该训练）：
  1. 文件数 / 总帧数 / 每局帧数（判据：每局中位 ≥300）
  2. 动作种类（判据：≥20）
  3. label 六段 one-hot 合法性
  4. 我方 / 敌方 / 中立实体数，被压制帧比例（判据：≥5%）
  5. 我方单位 ≤2 的帧比例（判据：≥3%）
  6. target_unit / selected_units 索引是否越界（≥max_entities 不可能，但要看 padding 占比）
  7. delay 分布（是否饱和在 127）

用法：
    python experiments/replay_gen/bvb_gate.py --dir data/bvb_10_nonop
"""

import argparse
import collections
import glob
import os
import statistics as st
import sys

import torch

SCALAR, N, EMB = 8020, 128, 1856
ALLI = 272                      # alliance one-hot 起点（272..276）
ACT_DIM = 564
S_DLY, E_DLY = 564, 692
S_QUE, E_QUE = 692, 694
S_SEL, E_SEL = 694, 2230
S_TGT, E_TGT = 2230, 2358
S_LOC, E_LOC = 2358, 6454

PAD = 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    if not files:
        print("NO FILES in", args.dir)
        return 2

    tot_frames = 0
    per_file = []
    act_counter = collections.Counter()
    legal = collections.Counter()
    suppressed = 0
    weak_self = 0
    self_counts, enemy_counts, neutral_counts = [], [], []
    tgt_pad = 0
    sel_pad = 0
    sel_slots = 0
    dly_vals = []
    dly_sat = 0
    loc_pad = 0

    for p in files:
        x, y = torch.load(p, map_location="cpu")
        T = x.shape[0]
        tot_frames += T
        per_file.append((os.path.basename(p), T))

        at = y[:, :ACT_DIM].argmax(1)
        act_counter.update(at.tolist())

        e = x[:, SCALAR:SCALAR + N * EMB].reshape(T, N, EMB)
        real = (e.abs().sum(2) > 1e-6)
        aidx = e[:, :, ALLI:ALLI + 5].argmax(2)
        self_n = ((aidx == 1) & real).sum(1).float()
        enemy_n = ((aidx == 4) & real).sum(1).float()
        neutral_n = ((aidx == 3) & real).sum(1).float()
        self_counts.append(float(self_n.mean()))
        enemy_counts.append(float(enemy_n.mean()))
        neutral_counts.append(float(neutral_n.mean()))
        suppressed += int((enemy_n > self_n).sum())
        weak_self += int((self_n <= 2).sum())
        del e, real, aidx

        for name, seg, dim, kind in [
            ("action_type", y[:, :ACT_DIM], 1, "onehot"),
            ("delay", y[:, S_DLY:E_DLY], 1, "onehot"),
            ("queued", y[:, S_QUE:E_QUE], 1, "onehot"),
        ]:
            ok = bool(((seg.sum(1) > 0.99) & (seg.sum(1) < 1.01)).all())
            legal[name] += int(ok)
        for name, seg, dim in [("selected_units", y[:, S_SEL:E_SEL], 12),
                               ("target_unit", y[:, S_TGT:E_TGT], 1),
                               ("target_location", y[:, S_LOC:E_LOC], 1)]:
            if name == "selected_units":
                s = seg.reshape(T, 12, 128)
                ok = bool(((s.sum(2) > 0.99) & (s.sum(2) < 1.01)).all())
            else:
                ok = bool(((seg.sum(1) > 0.99) & (seg.sum(1) < 1.01)).all())
            legal[name] += int(ok)

        tgt = y[:, S_TGT:E_TGT].argmax(1)
        tgt_pad += int((tgt == PAD).sum())
        sel = y[:, S_SEL:E_SEL].reshape(T, 12, 128).argmax(2)
        sel_pad += int((sel == PAD).sum())
        sel_slots += T * 12
        d = y[:, S_DLY:E_DLY].argmax(1)
        dly_vals.extend(d.tolist())
        dly_sat += int((d >= 127).sum())
        loc = y[:, S_LOC:E_LOC].reshape(T, 4096).argmax(1)
        loc_pad += int((loc == 4095).sum())
        del x, y

    n_files = len(files)
    frames = [t for _, t in per_file]

    print("=" * 78)
    print("Bot vs Bot 数据门禁   目录 = %s" % args.dir)
    print("=" * 78)
    print("文件数 = %d   总帧数 = %d   每局帧数 min/中位/max = %d/%d/%d"
          % (n_files, tot_frames, min(frames), int(st.median(frames)), max(frames)))
    print()

    nz = {k: v for k, v in act_counter.items() if k != 0}
    print("动作种类（非 no_op）= %d" % len(nz))
    print("  top8：%s" % [(k, v) for k, v in act_counter.most_common(8) if k != 0][:8])
    print()

    print("label 合法性（通过文件数 / 总数）：")
    for k in ["action_type", "delay", "queued", "selected_units", "target_unit", "target_location"]:
        print("  %-16s %d/%d" % (k, legal[k], n_files))
    print()

    print("实体构成（每局均值）：")
    print("  我方 %.1f   敌方 %.1f   中立 %.1f"
          % (st.mean(self_counts), st.mean(enemy_counts), st.mean(neutral_counts)))
    print("  被压制帧（敌方>我方）= %d / %d = %.2f%%"
          % (suppressed, tot_frames, 100.0 * suppressed / tot_frames))
    print("  我方 ≤2 单位的帧   = %d / %d = %.2f%%"
          % (weak_self, tot_frames, 100.0 * weak_self / tot_frames))
    print()

    print("索引 / 空值占比：")
    print("  target_unit padding   = %.1f%%" % (100.0 * tgt_pad / tot_frames))
    print("  selected_units padding = %.1f%%" % (100.0 * sel_pad / max(sel_slots, 1)))
    print("  target_location 占位   = %.1f%%" % (100.0 * loc_pad / tot_frames))
    dly = sorted(dly_vals)
    print("  delay 中位=%d p95=%d max=%d 饱和(=127)=%d (%.2f%%)"
          % (dly[len(dly) // 2], dly[int(0.95 * len(dly))], dly[-1],
             dly_sat, 100.0 * dly_sat / max(len(dly), 1)))
    print()

    checks = [
        ("每局帧数中位 ≥300", int(st.median(frames)) >= 300),
        ("动作种类 ≥20", len(nz) >= 20),
        ("被压制帧 ≥5%", 100.0 * suppressed / tot_frames >= 5.0),
        ("我方≤2 单位帧 ≥3%", 100.0 * weak_self / tot_frames >= 3.0),
        ("label 六段全合法", all(legal[k] == n_files for k in legal)),
        ("delay 无饱和", dly_sat == 0),
    ]
    print("=" * 78)
    print("判据")
    print("=" * 78)
    ok_all = True
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        ok_all &= ok
    print()
    print("总体：%s" % ("PASS —— 可以训练" if ok_all else "FAIL —— 先修数据，不要训练"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
