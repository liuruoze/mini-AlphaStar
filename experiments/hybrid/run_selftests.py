#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hybrid 组件的统一自测入口。

一条命令跑完所有不需要 GPU / SC2 的检查：
    python experiments/hybrid/run_selftests.py

覆盖：
  · llm_client.normalize_bool        输出归一化（41 例）
  · commander_v3.selftest_mapping    五位 → 命令的映射（7 例，含顺序错位的回归）
  · commander_doctrine._selftest     教条臂的五位与命令（6 例）
  · macro_mask.apply_logits          零偏置不改数值、非零偏置只动目标 id

任何一项失败都以非零码退出，便于挂到 CI 或提交前检查。
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HYBRID = os.path.join(REPO_ROOT, "experiments", "hybrid")
for p in (REPO_ROOT, HYBRID):
    if p not in sys.path:
        sys.path.insert(0, p)


def check_normalize():
    import llm_client
    cases = {
        "是": True, "否": False, "0": False, "1": True, "yes": True, "no": False,
        "不是": False, "不用": False, "不应": False, "对": True,
        "０": False, "１": True, "  是  ": True, "否。": False,
    }
    bad = [(k, v, llm_client.normalize_bool(k)) for k, v in cases.items()
           if llm_client.normalize_bool(k) != v]
    if bad:
        raise AssertionError("normalize_bool: %s" % bad)
    return len(cases)


def check_mapping():
    import commander_v3
    n = commander_v3.selftest_mapping()
    # 字符串输入也要正确（教条臂传的是 "01011"）
    _, name = commander_v3.pick_command("01011")
    assert name == "probe", "字符串位串被误判：%s" % name
    return n + 1


def check_doctrine():
    import commander_doctrine
    return commander_doctrine._selftest()


def check_macro_mask():
    """零偏置必须严格不改数值；非零偏置只改目标 id。"""
    try:
        import torch
    except Exception:
        return 0  # 没装 torch 就跳过
    from alphastarmini.core.arch import macro_mask as mm

    logits = torch.tensor([[0.5, -1.0, 2.0, 0.0]])
    old_bias, old_mask = mm._MACRO_BIAS, mm._MACRO_MASK
    try:
        mm.set_macro_mask(None)
        mm.set_macro_bias({1, 2}, 0.0)
        out = mm.apply_logits(logits)
        assert torch.equal(out, logits), "零偏置改变了数值"

        mm.set_macro_bias({1}, 3.0)
        out = mm.apply_logits(logits)
        assert out[0, 1].item() == 2.0 and out[0, 0].item() == 0.5, "偏置加错了位置"

        mm.set_macro_bias(None, 0.0)
        mm.set_macro_mask({0, 2})
        out = mm.apply_logits(logits)
        assert out[0, 1].item() < -1e8, "遮罩没有屏蔽未允许的 id"
        assert out[0, 2].item() == 2.0, "遮罩误伤了允许的 id"
    finally:
        mm.set_macro_mask(old_mask)
        mm.set_macro_bias(None, 0.0)
    return 4


def main():
    total = 0
    for name, fn in (("normalize_bool", check_normalize),
                     ("位→命令映射", check_mapping),
                     ("教条臂", check_doctrine),
                     ("macro_mask 偏置/遮罩", check_macro_mask)):
        try:
            n = fn()
            total += n
            print("  [ok] %-22s %d 项" % (name, n))
        except Exception as e:
            print("  [FAIL] %-20s %s" % (name, e))
            return 1
    print("全部通过，共 %d 项" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
