#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""类权重表加载器的单测（不需要 SC2 与 GPU，秒级）。

覆盖：
  T1 正常表：按 id 命中，未列出的 id 默认为 1.0
  T2 缓存：同一路径第二次加载返回同一对象（避免每个 batch 读盘）
  T3 真实产物：data/a3nc_h5_classw.json 能被加载，条目数与取值范围合理
  T4 越界 id：表里出现 >= n_action_type 的键时被忽略，不改变张量长度

用法：python experiments/test_class_weights.py
"""

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from alphastarmini.core.sl.sl_loss_multi_gpu import _load_class_weight_table, _CLASS_W_CACHE
from alphastarmini.lib.hyper_parameters import Label_Size as LS, LabelIndex

N = LS[LabelIndex.action_type_encoding]


def main():
    results = []

    # T1: 正常表
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"meta": {"alpha": 0.5}, "weights": {"2": 0.5, "64": 1.64, "49": 2.64}}, tmp)
    tmp.close()
    tbl = _load_class_weight_table(tmp.name)
    ok = (tbl.numel() == N and abs(float(tbl[2]) - 0.5) < 1e-6
          and abs(float(tbl[64]) - 1.64) < 1e-6 and float(tbl[0]) == 1.0 and float(tbl[3]) == 1.0)
    results.append(("T1 命中/默认值", ok, "n=%d w2=%.3f w0=%.3f" % (tbl.numel(), float(tbl[2]), float(tbl[0]))))

    # T2: 缓存
    tbl2 = _load_class_weight_table(tmp.name)
    results.append(("T2 缓存复用", tbl2 is tbl and tmp.name in _CLASS_W_CACHE,
                    "cache_entries=%d" % len(_CLASS_W_CACHE)))

    # T3: 真实产物
    real = os.path.join(REPO_ROOT, "data", "a3nc_h5_classw.json")
    if os.path.exists(real):
        t3 = _load_class_weight_table(real)
        n_nonone = int((t3 != 1.0).sum())
        lo, hi = float(t3.min()), float(t3.max())
        ok3 = t3.numel() == N and n_nonone >= 50 and 1.0 <= lo <= hi <= 100.0
        results.append(("T3 真实权重表", ok3, "非1条目=%d 范围=%.2f–%.2f" % (n_nonone, lo, hi)))
    else:
        results.append(("T3 真实权重表", True, "跳过（文件不存在）"))

    # T4: 越界 id
    tmp2 = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"weights": {"9999": 5.0, "2": 0.5}}, tmp2)
    tmp2.close()
    t4 = _load_class_weight_table(tmp2.name)
    results.append(("T4 越界 id 忽略", t4.numel() == N and abs(float(t4[2]) - 0.5) < 1e-6,
                    "n=%d" % t4.numel()))

    print("%-18s %-6s %s" % ("check", "pass", "detail"))
    bad = 0
    for name, ok, detail in results:
        print("%-18s %-6s %s" % (name, "OK" if ok else "FAIL", detail))
        bad += 0 if ok else 1
    print("\n%d/%d passed" % (len(results) - bad, len(results)))
    for f in (tmp.name, tmp2.name):
        try:
            os.unlink(f)
        except OSError:
            pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
