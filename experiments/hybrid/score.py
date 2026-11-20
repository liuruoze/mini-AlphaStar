# -*- coding: utf-8 -*-
"""评分的唯一实现（方案 B）。

任何脚本要算分，都必须 import 这里——不允许再写第二套。

规则（2026-11-20 20:27 用户定）：
  5 位全对 → 1.0；只错 1 位 → 0.5；错 2 位及以上 → 0
"""


def score_item(pred, truth):
    """单题得分。pred/truth 都是 5 位字符串。"""
    if pred is None or len(pred) != 5:
        return 0.0                     # 格式失败不计分（单独统计）
    wrong = sum(1 for a, b in zip(pred, truth) if a != b)
    return 1.0 if wrong == 0 else (0.5 if wrong == 1 else 0.0)


def score_all(preds, truths):
    """preds/truths 等长序列；返回 (总分, 全对率, 格式失败数)。"""
    assert len(preds) == len(truths)
    total = 0.0
    exact = 0
    bad = 0
    for p, t in zip(preds, truths):
        if p is None or len(p) != 5:
            bad += 1
            continue
        total += score_item(p, t)
        exact += (p == t)
    n = len(truths)
    return total / float(n), exact / float(n), bad


def per_bit(preds, truths):
    """逐位准确率，用于诊断"错在哪一条"。"""
    ok = [0] * 5
    tot = [0] * 5
    for p, t in zip(preds, truths):
        if p is None or len(p) != 5:
            continue
        for i in range(5):
            tot[i] += 1
            ok[i] += (p[i] == t[i])
    return [ok[i] / float(tot[i]) if tot[i] else 0.0 for i in range(5)]


if __name__ == "__main__":
    # 自测：文档里的例子
    cases = [("11100", "11100", 1.0), ("11000", "11100", 0.5),
             ("01100", "11100", 0.5), ("00000", "11100", 0.0),
             ("11110", "11100", 0.5), ("10000", "11100", 0.0), ("00001", "11100", 0.0)]
    for p, t, want in cases:
        got = score_item(p, t)
        print("%s vs %s → %.1f  %s" % (p, t, got, "OK" if got == want else "FAIL"))
    print(score_all(["11100", "11000", "00000"], ["11100", "11100", "11100"]))
