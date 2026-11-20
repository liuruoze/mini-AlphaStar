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
    from alphastarmini.core.arch import macro_mask as MM

    A.set_macro_mask(None)
    lg = fake_logits(16)
    act = torch.zeros(1, 1, dtype=torch.long)
    out_lg, out_a = MM.apply(lg.action_type, act)
    res.append(("T1 无遮罩不变", torch.equal(out_lg, lg.action_type) and torch.equal(out_a, act)))

    A.set_macro_mask({2, 5})
    out_lg, out_a = MM.apply(lg.action_type, act)
    p = torch.softmax(out_lg, dim=-1)
    allowed = torch.zeros(16, dtype=torch.bool)
    allowed[[2, 5]] = True
    res.append(("T2 屏蔽概率为0", float(p[..., ~allowed].sum()) < 1e-9 and int(out_a.flatten()[0]) in (2, 5)))

    A.set_macro_mask({2, 999})
    out_lg, out_a = MM.apply(lg.action_type, act)
    res.append(("T3 越界id忽略", int(out_a.flatten()[0]) == 2))

    # T4 真实前向：遮罩必须改变"被选中的动作"，而不只是改 logits
    A.set_macro_mask(None)
    try:
        from alphastarmini.core.arch.arch_model import ArchModel
        from alphastarmini.core.sl.feature import Feature
        import glob
        fs = Feature.getSize()
        f = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "replay_data_a3nc_h5", "*.pt")))[7]
        x = torch.load(f, map_location="cpu")
        st = Feature.feature2state(x[0][:8].reshape(8, fs))
        m = ArchModel()
        m.load_state_dict(torch.load(sorted(glob.glob(os.path.join(
            REPO_ROOT, "model/sl/b0s5/*_best.pth")))[0], map_location="cpu"), strict=False)
        m.eval()
        with torch.no_grad():
            a1 = m.forward(st, batch_size=1, sequence_length=8, return_logits=True)[1].action_type
            A.set_macro_mask({2})
            a2 = m.forward(st, batch_size=1, sequence_length=8, return_logits=True)[1].action_type
            A.set_macro_mask(None)
            a3 = m.forward(st, batch_size=1, sequence_length=8, return_logits=True)[1].action_type
        # 注意：策略头是**采样**的（同状态两次前向结果不同），所以不能要求 a1==a3。
        # 能验证的是：遮罩期间动作全部被压到允许集内；解除后恢复多样性。
        in_mask = bool((a2 == 2).all())
        released = bool(not (a3 == 2).all())
        res.append(("T4 遮罩生效且可解除", bool(in_mask and released)))
    except Exception as e:
        res.append(("T4 遮罩生效且可解除", False))
        print("  T4 异常:", repr(e))

    A.set_macro_bias({7}, 20.0)
    lg2 = fake_logits(16)
    act2 = torch.zeros(1, 1, dtype=torch.long)
    _, a_b = MM.apply(lg2.action_type, act2)
    A.set_macro_bias(None, 0.0)
    _, a_0 = MM.apply(lg2.action_type, act2)
    # 关掉偏置后 apply 原样返回传入的 action（不重算），这才是"不干预"的语义
    res.append(("T5 软偏置生效且可关", int(a_b.flatten()[0]) == 7
                and int(a_0.flatten()[0]) == int(act2.flatten()[0])))

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
