# -*- coding: utf-8 -*-
"""把用户的人工标注与 V5 阶梯、LLM 的答案三方对比。

用法：python experiments/hybrid/compare_labels.py <label_sheet.md>
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from bench_commander import make_states, gt_v5  # noqa: E402

HUMAN = {
    "补水晶": "S2", "补农民": "S1", "补兵营": "S9", "量产狂热者": "S3",
    "全面进攻": "S6", "攀科技": "S5", "开二矿": "S4", "回防": "S7", "补气矿": "S8",
}


def parse(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.startswith("|") or "---" in line or "我会怎么做" in line:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 9 or not c[0].isdigit():
            continue
        rows.append({"idx": int(c[0]), "label": c[-1]})
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(REPO_ROOT), "workbuddy", "LABEL_SHEET_20261120.md")
    rows = parse(path)
    import bench_commander as B
    B.REGIMES = B.REGIMES_V5
    states = make_states(n_per=4, seed=20261120)

    print("%-3s %-14s %-8s %-6s %s" % ("#", "你的标注", "映射", "V5", "是否一致"))
    ok = 0
    for r, (name, st) in zip(rows, states):
        human = r["label"]
        hs = None
        for k, v in HUMAN.items():
            if k in human:
                hs = v
                break
        v5, why = gt_v5(st)
        same = (hs == v5)
        ok += same
        print("%-3d %-14s %-8s %-6s %s" % (r["idx"], human, hs or "?", v5, "✓" if same else "✗"))
    print("\nV5 与你的标注一致：%d/%d = %.0f%%" % (ok, len(rows), 100.0 * ok / len(rows)))

    # 反推你的阈值
    print("\n反推你的阈值（看分界）：")
    for r, (name, st) in zip(rows, states):
        print("  #%-2d gap=%-3d workers=%-3d gw=%d army=%-3d -> %s"
              % (r["idx"], st["cap"] - st["used"], st["workers"], st.get("gw", 0),
                 st["army"], r["label"]))


if __name__ == "__main__":
    main()
