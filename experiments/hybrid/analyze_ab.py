# -*- coding: utf-8 -*-
"""按难度汇总 A/B：胜率、击杀率、配对差。用法：python analyze_ab.py /tmp/ab3/summary.csv"""
import csv, statistics as st, sys
from math import comb
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("killed", "step", "minutes", "food", "army"):
        pass
    r["killed"] = int(r["killed"]); r["minutes"] = float(r["minutes"])
    r["rate"] = r["killed"] / r["minutes"] if r["minutes"] else 0
    r["native"] = float(r["native"]); r["custom"] = float(r["custom"])

by = defaultdict(dict)
for r in rows:
    by[(int(r["d"]), r["arm"])][int(r["seed"])] = r

def mcnemar(pairs):
    """配对二值检验：只统计方向不一致的对子。返回 (仅左赢, 仅右赢, p)。"""
    b = sum(1 for l, r in pairs if l and not r)
    c = sum(1 for l, r in pairs if r and not l)
    n = b + c
    if not n:
        return b, c, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)
    return b, c, p


print("%-4s %-4s %5s %10s %10s %8s %8s" % ("难度", "臂", "局数", "击杀均值", "击杀/分", "native", "custom"))
for d in sorted({k[0] for k in by}):
    for arm in ("sl", "hyb", "doc"):
        rs = by.get((d, arm), {})
        if not rs:
            continue
        print("%-4d %-4s %5d %10.0f %10.1f %7.0f%% %7.0f%%" % (
            d, arm, len(rs), st.mean(r["killed"] for r in rs.values()),
            st.mean(r["rate"] for r in rs.values()),
            100 * st.mean(r["native"] for r in rs.values()),
            100 * st.mean(r["custom"] for r in rs.values())))
    for left, right in (("sl", "hyb"), ("sl", "doc"), ("hyb", "doc")):
        a, b = by.get((d, left), {}), by.get((d, right), {})
        common = sorted(set(a) & set(b))
        if len(common) >= 3:
            dif = [b[s]["killed"] - a[s]["killed"] for s in common]
            m, sd = st.mean(dif), (st.stdev(dif) if len(dif) > 1 else 0)
            se = sd / (len(dif) ** 0.5) if sd else 0
            t = m / se if se else 0
            print("     配对差(%s-%s) 均值 %.0f ｜ sd %.0f ｜ se %.0f ｜ t=%.2f ｜ n=%d"
                  % (right, left, m, sd, se, t, len(common)))
            for col, label in (("native", "胜负"), ("custom", "custom")):
                pr = [(a[s][col] == 1.0, b[s][col] == 1.0) for s in common]
                bb, cc, pp = mcnemar(pr)
                print("     McNemar(%s)：仅左赢 %d ｜ 仅右赢 %d ｜ p=%.4f"
                      % (label, bb, cc, pp))
