# -*- coding: utf-8 -*-
"""按难度汇总 A/B：胜率、击杀率、配对差。用法：python analyze_ab.py /tmp/ab3/summary.csv"""
import csv, statistics as st, sys
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

print("%-4s %-4s %5s %10s %10s %8s %8s" % ("难度", "臂", "局数", "击杀均值", "击杀/分", "native", "custom"))
for d in sorted({k[0] for k in by}):
    for arm in ("sl", "hyb"):
        rs = by.get((d, arm), {})
        if not rs:
            continue
        print("%-4d %-4s %5d %10.0f %10.1f %7.0f%% %7.0f%%" % (
            d, arm, len(rs), st.mean(r["killed"] for r in rs.values()),
            st.mean(r["rate"] for r in rs.values()),
            100 * st.mean(r["native"] for r in rs.values()),
            100 * st.mean(r["custom"] for r in rs.values())))
    a, b = by.get((d, "sl"), {}), by.get((d, "hyb"), {})
    common = sorted(set(a) & set(b))
    if len(common) >= 3:
        dif = [b[s]["killed"] - a[s]["killed"] for s in common]
        m, sd = st.mean(dif), (st.stdev(dif) if len(dif) > 1 else 0)
        se = sd / (len(dif) ** 0.5) if sd else 0
        t = m / se if se else 0
        print("     配对差(hyb-sl) 均值 %.0f ｜ sd %.0f ｜ se %.0f ｜ t=%.2f ｜ n=%d"
              % (m, sd, se, t, len(common)))
