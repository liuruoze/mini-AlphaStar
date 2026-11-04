"""第 81 轮（DS，#34）：按"交战密度"给 replay 打分，用于数据筛选。

动机（DS12 81.6 / ERRATA A33）：
  11 个败局的 killed 均值只有 23 —— 是"零交战败局"。
  假设：训练集里"被进攻/交战"的状态太少，模型没见过。
  做法：用 label 的 action_type one-hot（第 2 列 = Attack_pt）统计每个文件的攻击帧密度，
  把密度高的文件排到前面，训练时用 MAS_SL_MAX_FILES 取前 N 个即可。

输出：data/replay_data_v3_cbt/ 目录，内含按"攻击密度降序"命名的软链接（00_xxx.pt ...）。
"""
from __future__ import print_function

import glob
import os
import sys

import torch

from alphastarmini.lib.hyper_parameters import Label_Size as LS

SRC = "data/replay_data_v3"
DST = "data/replay_data_v3_cbt"


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.pt")))
    if not files:
        print("找不到数据文件：%s" % SRC)
        return 1

    rows = []
    for i, fp in enumerate(files):
        try:
            feat, lab = torch.load(fp)
        except Exception as e:
            print("[skip] %s: %r" % (fp, e))
            continue
        n = lab.shape[0]
        at = lab[:, :LS.action_type_encoding]
        noop = float(at[:, 0].sum())
        atk = float(at[:, 2].sum())          # Attack_pt 的 one-hot 列
        smart = float(at[:, 12].sum())       # Smart_unit
        train = float(at[:, 49].sum())       # Train_Zealot_quick
        rows.append({
            "file": os.path.basename(fp),
            "path": os.path.abspath(fp),
            "n": n,
            "atk": atk,
            "density": atk / max(n, 1),
            "noop": noop / max(n, 1),
            "smart": smart / max(n, 1),
            "train": train / max(n, 1),
        })
        print("[%2d/%2d] %-42s n=%-6d atk=%-6.0f 密度=%.4f  no_op=%.3f  zealot=%.4f"
              % (i + 1, len(files), rows[-1]["file"], n, atk, rows[-1]["density"],
                 rows[-1]["noop"], rows[-1]["train"]), flush=True)

    rows.sort(key=lambda r: -r["density"])

    if not os.path.exists(DST):
        os.makedirs(DST)
    old = glob.glob(os.path.join(DST, "*.pt"))
    for p in old:
        os.remove(p)

    for i, r in enumerate(rows):
        link = os.path.join(DST, "%02d_%s" % (i, r["file"]))
        os.symlink(r["path"], link)

    print("\n=== 按攻击密度降序（前 20）===")
    for i, r in enumerate(rows[:20]):
        print("%2d  %-42s 密度=%.4f  n=%-6d zealot=%.4f"
              % (i, r["file"], r["density"], r["n"], r["train"]))
    print("\n软链接已写入 %s（共 %d 个）" % (DST, len(rows)))

    # 顺便给出"前 20 文件"与"全部 60 文件"的密度均值，便于对照
    d_all = sum(r["density"] for r in rows) / len(rows)
    d_top = sum(r["density"] for r in rows[:20]) / 20.0
    print("攻击密度均值：全部 %.4f / 前 20 %.4f（提升 %.2fx）"
          % (d_all, d_top, d_top / max(d_all, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
