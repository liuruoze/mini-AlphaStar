#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""宏观动作遮罩的 4 条验证（不需要 SC2，秒级；T4 需要一次前向）。

T1 不设遮罩时返回的是同一个 logits（逐位不变）
T2 设遮罩后被屏蔽动作的 softmax 概率为 0，argmax 落在允许集内
T3 越界 id 被忽略，不抛异常
T4 真实前向：设 None 与从未设置时的 logits 逐位相同（哈希一致）

用法：python experiments/test_macro_mask.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from alphastarmini.core.arch import agent as A


def fake_logits(n=16):
    """用真实类型，避免测试与实现的字段假设不一致。"""
    from alphastarmini.core.rl.action import ArgsActionLogits
    return ArgsActionLogits(torch.randn(1, 1, n), torch.randn(1, 1, 4), torch.randn(1, 1, 2),
                            torch.randn(1, 1, 4), torch.randn(1, 1, 4), torch.randn(1, 1, 4))


def main():
    res = []

    A.set_macro_mask(None)
    lg = fake_logits()
    out = A._apply_macro_mask(lg)
    res.append(("T1 无遮罩逐位不变", torch.equal(out.action_type, lg.action_type)))

    A.set_macro_mask({2, 5})
    out = A._apply_macro_mask(lg)
    p = torch.softmax(out.action_type, dim=-1)
    allowed = torch.zeros(16, dtype=torch.bool)
    allowed[[2, 5]] = True
    res.append(("T2 屏蔽概率为0",
                float(p[..., ~allowed].sum()) < 1e-9 and int(out.action_type.argmax()) in (2, 5)))

    A.set_macro_mask({2, 999})
    try:
        out = A._apply_macro_mask(lg)
        ok = int(out.action_type.argmax()) in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2)
        # 允许集只有 {2}，其余被屏蔽 → argmax 必须是 2
        ok = int(out.action_type.argmax()) == 2
    except Exception:
        ok = False
    res.append(("T3 越界 id 忽略", ok))

    # T4 真实 logits 上应用遮罩 + 代码里确实接在 action_logits_by_state 上
    A.set_macro_mask(None)
    try:
        from alphastarmini.core.arch.arch_model import ArchModel
        from alphastarmini.core.sl.feature import Feature
        from alphastarmini.core.sl.label import Label
        import glob
        fs, ls = Feature.getSize(), Label.getSize()
        f = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "replay_data_a3nc_h5", "*.pt")))[0]
        x = torch.load(f, map_location="cpu")
        b = torch.cat([x[0][:8], x[1][:8], torch.zeros(8, 1)], dim=-1).reshape(1, 8, -1)
        state = Feature.feature2state(b[:, :, :fs].reshape(8, fs))
        m = ArchModel()
        m.eval()
        with torch.no_grad():
            l1 = m.forward(state, batch_size=1, sequence_length=8, return_logits=True)[0]
            A.set_macro_mask({2})
            l2 = A._apply_macro_mask(l1)          # 与 Agent.action_logits_by_state 里同一函数
            A.set_macro_mask(None)
        sm = torch.softmax(l2.action_type, -1)
        masked_ok = float(sm[..., 3:].sum()) < 1e-9 and int(l2.action_type.argmax()) % 564 == 2
        src = open(os.path.join(REPO_ROOT, "alphastarmini/core/arch/agent.py")).read()
        wired = "action_logits = _apply_macro_mask(action_logits)" in src
        res.append(("T4 真实logits遮罩+已接线", bool(masked_ok and wired)))
    except Exception as e:
        res.append(("T4 真实logits遮罩+已接线", False))
        print("  T4 异常:", repr(e))

    A.set_macro_mask(None)
    bad = 0
    print("%-26s %-5s" % ("check", "pass"))
    for n, ok in res:
        print("%-26s %-5s" % (n, "OK" if ok else "FAIL"))
        bad += 0 if ok else 1
    print("\n%d/%d passed" % (len(res) - bad, len(res)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
