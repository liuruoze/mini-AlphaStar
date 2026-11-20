# -*- coding: utf-8 -*-
"""基准的平凡解基线：一个"不会思考"的策略能拿多少分。

如果平凡策略的分数接近真模型，说明题集没有区分度；
如果远低于，说明题集确实需要判断。

同时统计"正常对局"占比（有水兵营、有水晶的局面），
因为大量灾难局面会让基准偏离真实部署分布。
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
ITEMS = D["items"]
N = len(ITEMS)


def per_bit_acc(predict):
    """predict(item) -> 5 位串"""
    hit = sum(sum(1 for a, b in zip(predict(it), it["label"]) if a == b) for it in ITEMS)
    return hit / (5.0 * N)


def score_b(predict):
    """方案 B（用户 20:27 定）：5 位全对 1 分；只错 1 位 0.5 分；错 2 位及以上 0 分。"""
    tot = 0.0
    for it in ITEMS:
        wrong = sum(1 for a, b in zip(predict(it), it["label"]) if a != b)
        tot += 1.0 if wrong == 0 else (0.5 if wrong == 1 else 0.0)
    return tot / float(N)


def balanced_acc(predict):
    """每位分别算 (TPR+TNR)/2，再对五位平均 —— 只答一边恒为 50%。"""
    per = []
    for i in range(5):
        tp = fp = tn = fn = 0
        for it in ITEMS:
            p = predict(it)[i] == "1"
            t = it["label"][i] == "1"
            if t and p:
                tp += 1
            elif t and not p:
                fn += 1
            elif not t and p:
                fp += 1
            else:
                tn += 1
        tpr = tp / float(tp + fn) if tp + fn else 0.0
        tnr = tn / float(tn + fp) if tn + fp else 0.0
        per.append((tpr + tnr) / 2.0)
    return sum(per) / 5.0, per


def exact_acc(predict):
    return sum(1 for it in ITEMS if predict(it) == it["label"]) / float(N)


def main():
    print("题数 %d，逐位总数 %d\n" % (N, 5 * N))

    # 每一位的多数类
    maj = "".join("1" if sum(1 for it in ITEMS if it["label"][i] == "1") > N / 2 else "0"
                  for i in range(5))
    print("逐位多数 = %s（%s）" % (maj, "/".join(
        ["补水晶", "补农民", "补兵营", "量产", "进攻"])))

    strategies = [
        ("全 0（什么都不做）", lambda it: "00000"),
        ("全 1（全都做）", lambda it: "11111"),
        ("逐位多数（%s）" % maj, lambda it: maj),
        ("最常见组合 %s" % Counter(it["label"] for it in ITEMS).most_common(1)[0][0],
         lambda it: Counter(x["label"] for x in ITEMS).most_common(1)[0][0]),
    ]
    print("\n%-28s %8s %10s %10s %8s" % ("平凡策略", "得分B", "逐位准确", "平衡准确", "全对率"))
    for name, fn in strategies:
        if fn is None:
            continue
        print("%-28s %8.3f %9.1f%% %9.1f%% %7.1f%%"
              % (name, score_b(fn), 100 * per_bit_acc(fn),
                 100 * balanced_acc(fn)[0], 100 * exact_acc(fn)))

    # 用规则本身当"作弊模型"，验证上限
    from doctrine import decide_bits
    oracle = lambda it: decide_bits(**it["state"])
    print("%-28s %8.3f %9.1f%% %9.1f%% %7.1f%%"
          % ("规则本身（理论上限）", score_b(oracle), 100 * per_bit_acc(oracle),
             100 * balanced_acc(oracle)[0], 100 * exact_acc(oracle)))

    # 只看"兵≥8"这一条，其余全 0 —— 单一启发式
    def only_attack(it):
        return "0000" + ("1" if it["state"]["army"] >= 8 else "0")
    ba, per = balanced_acc(only_attack)
    print("%-28s %8.3f %9.1f%% %9.1f%% %7.1f%%   逐位平衡=%s"
          % ("只看兵≥8", score_b(only_attack), 100 * per_bit_acc(only_attack), 100 * ba,
             100 * exact_acc(only_attack), ["%.0f" % (100 * x) for x in per]))

    # 正常对局占比
    normal = [it for it in ITEMS if it["state"]["pylons"] >= 1 and it["state"]["gw"] >= 1]
    print("\n正常对局（有水晶且至少 1 兵营）: %d/%d = %.0f%%"
          % (len(normal), N, 100.0 * len(normal) / N))
    print("灾难局面（无水晶或兵营 0）:     %d 条" % (N - len(normal)))

    # 每条规则的 1 占比（看是否有位接近常数）
    print("\n各位为 1 的占比：")
    for i, name in enumerate(["补水晶", "补农民", "补兵营", "量产", "进攻"]):
        one = sum(1 for it in ITEMS if it["label"][i] == "1")
        print("  %-6s %3d/%d = %3.0f%%%s" % (name, one, N, 100.0 * one / N,
                                             "   ← 偏斜" if not (0.3 <= one / N <= 0.7) else ""))

    # 完全靠"位组合"就能猜中的比例
    top = Counter(it["label"] for it in ITEMS).most_common(3)
    print("\n最常见的三种组合：%s（合计 %.0f%%）"
          % ([(k, v) for k, v in top], 100.0 * sum(v for _, v in top) / N))


if __name__ == "__main__":
    main()
