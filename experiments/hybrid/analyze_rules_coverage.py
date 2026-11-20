# -*- coding: utf-8 -*-
"""规则覆盖与边界分析：每条规则的临界值各有多少题命中，以及有哪些边界情况没被覆盖。

只做诊断，不改数据。
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

D = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
ITEMS = D["items"]


def s(it):
    return it["state"]


def main():
    print("总条数 %d\n" % len(ITEMS))

    print("== 各规则的临界值覆盖 ==")
    checks = [
        ("第1条 余量=9（应补水晶）", lambda x: x["cap"] - x["used"] == 9 and x["cap"] < 200),
        ("第1条 余量=10（不补）", lambda x: x["cap"] - x["used"] == 10),
        ("第1条 水晶=0", lambda x: x["pylons"] == 0),
        ("第1条 上限=200（补了也没用）", lambda x: x["cap"] >= 200),
        ("第2条 农民=14（补）", lambda x: x["workers"] == 14),
        ("第2条 农民=15（不补）", lambda x: x["workers"] == 15),
        ("第3条 兵营=2（补）", lambda x: x["gw"] == 2),
        ("第3条 兵营=3（不补）", lambda x: x["gw"] == 3),
        ("第3条 兵营<3 但无水晶（强制 0）", lambda x: x["gw"] < 3 and x["pylons"] == 0),
        ("第4条 兵=23（可量产）", lambda x: x["army"] == 23),
        ("第4条 兵=24（不可）", lambda x: x["army"] == 24),
        ("第4条 矿=99（不可）", lambda x: x["minerals"] == 99),
        ("第4条 矿=100（可）", lambda x: x["minerals"] == 100),
        ("第4条 农民=10（可）", lambda x: x["workers"] == 10),
        ("第4条 农民=9（不可）", lambda x: x["workers"] == 9),
        ("第4条 余量=1（不可）", lambda x: x["cap"] - x["used"] == 1),
        ("第4条 余量=2（可）", lambda x: x["cap"] - x["used"] == 2),
        ("第5条 兵=7（不打）", lambda x: x["army"] == 7),
        ("第5条 兵=8（打）", lambda x: x["army"] == 8),
    ]
    for name, fn in checks:
        n = sum(1 for it in ITEMS if fn(s(it)))
        print("  %-34s %3d 条 %s" % (name, n, "" if n else "  ← 未覆盖"))

    print("\n== 组合分布 ==")
    for k, v in Counter(it["label"] for it in ITEMS).most_common():
        print("  %s  %3d" % (k, v))

    print("\n== 潜在问题扫描 ==")
    # a) 空闲农民从未被规则使用
    idle_nonzero = sum(1 for it in ITEMS if s(it)["idle"] > 0)
    print("  空闲农民>0 的条目: %d/%d（规则完全不使用这个字段）" % (idle_nonzero, len(ITEMS)))

    # b) 时间为 0 位，但局面数随时间变化
    print("  时间不参与任何判定：同数字不同时间 → 标签必然相同")
    same = {}
    for it in ITEMS:
        key = tuple((k, v) for k, v in sorted(s(it).items()) if k != "t")
        same.setdefault(key, []).append(it)
    dup = {k: v for k, v in same.items() if len(v) > 1}
    print("  去掉时间后重复的局面组: %d" % len(dup))
    for k, v in list(dup.items())[:3]:
        print("    标签 %s × %d 条，时间 %s"
              % (v[0]["label"], len(v), [x["state"]["t"] for x in v]))

    # c) 全 0 标签（无事可做）
    zeros = [it for it in ITEMS if it["label"] == "00000"]
    print("  全 0（无事可做）: %d 条" % len(zeros))

    # d) 规则互斥性：第1条与第3条同时为 1 的题
    both = [it for it in ITEMS if it["label"][0] == "1" and it["label"][2] == "1"]
    print("  第1条与第3条同时为 1: %d 条（需两个农民同时建造，合理）" % len(both))

    # e) 只有单一位置 1 的题（其余全 0）
    single = Counter(sum(int(c) for c in it["label"]) for it in ITEMS)
    print("  每题的 1 的个数分布: %s" % dict(sorted(single.items())))


if __name__ == "__main__":
    main()
