# -*- coding: utf-8 -*-
"""把两轮标注（20 + 13）合起来重新拟合，并列出"任何单一规则都解释不了"的矛盾点。"""

import itertools
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

import bench_commander as B  # noqa: E402

# 第二轮 13 条的原始参数（导出脚本里的 ROWS）
BOUND = [
    ("A1", 420, 50, 70, 10, 4, 250, 1, 13, "补农民"),
    ("A2", 420, 50, 70, 10, 4, 250, 1, 14, "补农民"),
    ("A3", 420, 50, 70, 10, 4, 250, 1, 16, "进攻"),
    ("B1", 420, 50, 70, 6, 2, 300, 1, 25, "兵营"),
    ("B2", 420, 50, 70, 12, 2, 300, 1, 25, "兵营"),
    ("B3", 420, 50, 70, 6, 3, 300, 1, 25, "兵营"),
    ("C1", 480, 59, 70, 10, 4, 200, 2, 26, "进攻"),
    ("C2", 480, 59, 70, 11, 4, 200, 2, 26, "进攻"),
    ("C3", 450, 59, 70, 14, 4, 200, 2, 26, "进攻"),
    ("C4", 480, 59, 70, 14, 4, 200, 2, 26, "进攻"),
    ("D1", 420, 65, 70, 8, 4, 300, 1, 25, "补水晶"),
    ("D2", 420, 63, 70, 8, 4, 300, 1, 25, "补水晶"),
    ("D3", 420, 61, 70, 8, 4, 300, 1, 25, "补水晶"),
]

MAP = {"补水晶": "S2", "补农民": "S1", "兵营": "S9", "补兵营": "S9",
       "量产狂热者": "S3", "量产": "S3", "进攻": "S6", "全面进攻": "S6"}


def main():
    B.REGIMES = B.REGIMES_V5
    states = B.make_states(n_per=4, seed=20261120)
    first = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "labels_user_20261120.json"), encoding="utf-8"))["labels"]
    data = [(st, MAP[t]) for (_, st), t in zip(states, first)]
    for rid, t, used, cap, army, gw, mins, idle, wk, lab in BOUND:
        data.append((dict(t=t, used=used, cap=cap, army=army, gw=gw, mins=mins,
                          idle=idle, workers=wk, gas=0), MAP[lab]))

    def policy6(st):
        """六条规则版：在五条规则上补"存款高且产能不足就补兵营"（B3 那条矛盾点要求）。"""
        if st["cap"] - st["used"] <= 9:
            return "S2"
        if st["workers"] <= 14:
            return "S1"
        if st["gw"] < 3:
            return "S9"
        if st["gw"] < 4 and st.get("mins", 0) >= 250 and st["army"] < 10:
            return "S9"
        if st["t"] >= 420 and st["army"] >= 10:
            return "S6"
        return "S3"

    def policy(st, A, Bw, C, D, E):
        if st["cap"] - st["used"] <= A:
            return "S2"
        if st["workers"] <= Bw:
            return "S1"
        if st["gw"] < C:
            return "S9"
        if st["t"] >= D and st["army"] >= E:
            return "S6"
        return "S3"

    best = []
    for A, Bw, C, D, E in itertools.product(range(2, 12), range(10, 22), range(2, 6),
                                            [390, 420, 450, 480], range(8, 17)):
        hit = sum(policy(st, A, Bw, C, D, E) == lab for st, lab in data)
        best.append((hit, A, Bw, C, D, E))
    best.sort(reverse=True)
    print("合并 33 条标注，最优规则：%d/%d" % (best[0][0], len(data)))
    for hit, A, Bw, C, D, E in best[:3]:
        print("  gap≤%-2d workers≤%-2d gw<%-2d t≥%-3d army≥%-2d → %d"
              % (A, Bw, C, D, E, hit))

    _, A, Bw, C, D, E = best[0]
    print("\n按最优规则的错例（任何单一规则都解释不了的点）：")
    bad = 0
    for st, lab in data:
        p = policy(st, A, Bw, C, D, E)
        if p != lab:
            bad += 1
            print("  gap=%-3d workers=%-3d gw=%d t=%-3d army=%-3d 你=%s 规则=%s"
                  % (st["cap"] - st["used"], st["workers"], st.get("gw", 0), st["t"],
                     st["army"], lab, p))
    print("错 %d/%d" % (bad, len(data)))

    hit6 = sum(policy6(st) == lab for st, lab in data)
    print("\n六条规则版（补一条：存款高且兵营<4 就补兵营）：%d/%d" % (hit6, len(data)))
    for st, lab in data:
        p6 = policy6(st)
        if p6 != lab:
            print("  ✗ gap=%-3d w=%-3d gw=%d mins=%-3d t=%-3d army=%-3d 你=%s 规则=%s"
                  % (st["cap"] - st["used"], st["workers"], st.get("gw", 0), st.get("mins", 0),
                     st["t"], st["army"], lab, p6))


if __name__ == "__main__":
    main()
