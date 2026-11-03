#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 65 轮：KL 语义的独立手算参照（响应 Kimi-03 P0）。

## 为什么必须"手算"

`rl_algo.kl` 的注释原本写着「这不改变 KL 的数学语义」。Kimi 指出这句话是错的：
把被屏蔽位置**丢掉**又不重归一化，得到的是"共同支撑集上的部分和"，
既可能为负、概率质量和也 < 1。

所以本脚本不用被测公式自己验自己，而是**先用 math.log 手算**，
再要求 `kl_renorm` 命中手算值。这就是 Kimi 说的
「用独立手算的小概率分布做参照，不复用待测公式」。

## 覆盖的五个边界（Kimi-03 验收要求）

| 场景 | 目的 |
|---|---|
| A 两侧相同 | KL 必须 = 0 |
| B 支撑集相同但分布不同 | 两种模式都应命中手算值（对照组） |
| C 只有 teacher 有质量（student 被 mask） | **legacy 会算出负数**，renorm 必须 ≥ 0 |
| D 只有 student 有质量 | legacy 静默丢掉这部分，renorm 在交集上重归一化 |
| E 全部被 mask（空支撑集） | 必须给出 0，不能 nan/inf |

用法：
    python _test_r65_kl.py
"""
import math
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ["MAS_KL_MODE"] = "legacy"          # 显式关闭分派，直接比较两个函数

import torch

from alphastarmini.core.rl import rl_algo as RA

MASK = -1e9
TOL = 1e-5


def L(probs):
    """把概率向量转成 logits；0 表示"被屏蔽"，写成 -1e9 哨兵。"""
    vals = [(math.log(x) if x > 0 else MASK) for x in probs]
    return torch.tensor([vals], dtype=torch.float32)


def kl_manual(t_probs, s_probs, support):
    """在 support（索引集合）上重归一化后手算 KL。"""
    ts = [t_probs[i] for i in support]
    ss = [s_probs[i] for i in support]
    zt = sum(ts)
    zs = sum(ss)
    total = 0.0
    for tv, sv in zip(ts, ss):
        p = tv / zt
        q = sv / zs
        total += p * math.log(p / q)
    return total


def run_case(name, t_probs, s_probs, support, structural=None):
    print("\n--- %s ---" % name, flush=True)
    tl, sl = L(t_probs), L(s_probs)
    t_lp = torch.log_softmax(tl, dim=-1)
    s_lp = torch.log_softmax(sl, dim=-1)
    keep = (t_lp > -1e8) & (s_lp > -1e8)
    print("  共同支撑集 = %s" % torch.nonzero(keep[0]).reshape(-1).tolist(), flush=True)

    ref = kl_manual(t_probs, s_probs, support) if support else 0.0

    legacy = float(RA.kl([sl, tl]).sum())
    renorm = float(RA.kl_renorm([sl, tl]).sum())

    print("  手算（support=%s）: KL = %+.8f" % (support, ref), flush=True)
    print("  legacy 部分和      : %+.8f   %s"
          % (legacy, "（**负值**，不是 KL）" if legacy < -1e-9 else ""), flush=True)
    print("  renorm 重归一化 KL : %+.8f   %s"
          % (renorm, "PASS" if abs(renorm - ref) < TOL else "FAIL（与手算差 %.2e）" % abs(renorm - ref)),
          flush=True)

    ok = abs(renorm - ref) < TOL and renorm >= -1e-7
    if structural is not None:
        sm = structural
        r2 = float(RA.kl_renorm([sl, tl], selected_mask=sm).sum())
        print("  再加结构化掩码    : %+.8f   %s"
              % (r2, "PASS（=0）" if abs(r2) < 1e-9 else "FAIL"), flush=True)
        ok = ok and abs(r2) < 1e-9
    return ok


def main():
    all_ok = True

    # A 两侧相同：KL 必须 0
    t = [0.5, 0.5, 0.0, 0.0, 0.0]
    all_ok &= run_case("A 两侧完全相同", t, t, [0, 1])

    # B 支撑集相同、分布不同（对照组：两种模式都该命中）
    t = [0.6, 0.4, 0.0, 0.0, 0.0]
    s = [0.3, 0.7, 0.0, 0.0, 0.0]
    all_ok &= run_case("B 支撑集相同、分布不同", t, s, [0, 1])

    # C 只有 teacher 在 2 号位有质量，student 在那里被 mask
    t = [0.5, 0.3, 0.2, 0.0, 0.0]
    s = [0.7, 0.3, 0.0, 0.0, 0.0]
    all_ok &= run_case("C teacher-only（student 被 mask）", t, s, [0, 1])

    # D 只有 student 在 2 号位有质量，teacher 在那里被 mask
    t = [0.7, 0.3, 0.0, 0.0, 0.0]
    s = [0.2, 0.2, 0.6, 0.0, 0.0]
    all_ok &= run_case("D student-only（teacher 被 mask）", t, s, [0, 1])

    # E 全 mask：必须 0，不能 nan
    z = [0.0, 0.0, 0.0, 0.0, 0.0]
    print("\n--- E 空支撑集 ---", flush=True)
    tl, sl = L(z), L(z)
    v1 = float(RA.kl([sl, tl]).sum())
    v2 = float(RA.kl_renorm([sl, tl]).sum())
    print("  legacy = %+.8f   renorm = %+.8f   %s"
          % (v1, v2, "PASS" if (abs(v2) < 1e-9 and not math.isnan(v2)) else "FAIL"), flush=True)
    all_ok &= abs(v2) < 1e-9

    # F 结构化掩码把支撑集清空 -> 必须 0
    t = [0.6, 0.4, 0.0, 0.0, 0.0]
    s = [0.3, 0.7, 0.0, 0.0, 0.0]
    all_ok &= run_case("F 结构化掩码全 0", t, s, [0, 1],
                       structural=torch.zeros(1, 1))

    print("\n=== 结论 ===", flush=True)
    print("renorm 全部命中手算且非负：%s" % all_ok, flush=True)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
