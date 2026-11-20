# -*- coding: utf-8 -*-
"""用一套阈值拟合用户的人工标注，找出"他的策略"到底是多少。

候选规则（按优先级）：
  1) 人口余量 ≤ A      → S2 补水晶
  2) 农民 ≤ B          → S1 补农民
  3) 兵营 < C          → S9 补兵营
  4) 时间 ≥ D 且 army ≥ E → S6 进攻
  5) 否则              → S3 量产

我在标注表上手工反推得到 A=4、B=13、C=3、D=480s、E=12；这里用暴力搜索确认它是唯一解/最优解。
"""

import itertools
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

import json                              # noqa: E402

from compare_labels import HUMAN         # noqa: E402
import bench_commander as B              # noqa: E402

# 标注存在仓库内（避免脚本跨沙箱目录读文件）；内容与用户的 LABEL_SHEET 一一对应
LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_user_20261120.json")


def policy(st, A, Bw, C, D, E):
    if st["cap"] - st["used"] <= A:
        return "S2"
    if st["workers"] <= Bw:
        return "S1"
    if st.get("gw", 9) < C:
        return "S9"
    if st["t"] >= D and st["army"] >= E:
        return "S6"
    return "S3"


def main():
    B.REGIMES = B.REGIMES_V5
    states = B.make_states(n_per=4, seed=20261120)
    raw = json.load(open(LABELS, encoding="utf-8"))["labels"]
    human = []
    for txt in raw:
        hs = None
        for k, v in HUMAN.items():
            if k in txt:
                hs = v
                break
        human.append(hs)
    rows = [{"idx": i + 1} for i in range(len(human))]

    best = []
    for A, Bw, C, D, E in itertools.product(range(2, 8), range(10, 20), range(1, 5),
                                            [420, 450, 480, 510, 540], range(8, 17)):
        hit = sum(policy(st, A, Bw, C, D, E) == h for (_, st), h in zip(states, human))
        best.append((hit, A, Bw, C, D, E))
    best.sort(reverse=True)

    print("拟合最好的参数组合（满分 %d）：" % len(human))
    seen = set()
    for hit, A, Bw, C, D, E in best:
        key = (hit, A, Bw, C, D, E)
        if key in seen:
            continue
        seen.add(key)
        print("  %d/%d  gap≤%-2d workers≤%-2d gw<%-2d t≥%-3ds army≥%-2d"
              % (hit, len(human), A, Bw, C, D, E))
        if len(seen) >= 6:
            break

    hit, A, Bw, C, D, E = best[0]
    print("\n逐条核对（最优参数）：")
    for r, (name, st), h in zip(rows, states, human):
        p = policy(st, A, Bw, C, D, E)
        print("  #%-2d gap=%-3d w=%-3d gw=%d t=%d army=%-3d 你=%s 规则=%s %s"
              % (r["idx"], st["cap"] - st["used"], st["workers"], st.get("gw", 0),
                 st["t"], st["army"], h, p, "✓" if h == p else "✗"))


if __name__ == "__main__":
    main()
