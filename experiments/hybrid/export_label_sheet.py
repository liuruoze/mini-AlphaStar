# -*- coding: utf-8 -*-
"""导出人工标注表：20 个局面，留空请专家填"我会怎么做"。

用法：python experiments/hybrid/export_label_sheet.py > label_sheet.md
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from bench_commander import make_states, gt_v5  # noqa: E402

HEAD = """# 标注表：20 个局面（请填最后一列）

> 用法：在「我会怎么做」里随便写中文（例如"补水晶""继续造兵""该打了"），
> 我再映射成 S1–S8。不确定的写"不知道"，比硬填更有价值。
>
> 参考策略（供对照，不必照抄）：
> S1 补农民｜S2 补水晶（人口）｜**S9 补兵营**｜S3 量产狂热者｜S6 全面进攻｜S5 攀科技｜S4 开二矿｜S7 回防｜S8 补气矿

| # | 时间 | 人口 | 军队 | 兵营 | 矿 | 空闲农民 | 农民 | **我会怎么做**（填这里） |
|---|---|---|---|---|---|---|---|---|
"""


def main():
    import bench_commander as B
    B.REGIMES = B.REGIMES_V5
    states = make_states(n_per=4, seed=20261120)
    print(HEAD.rstrip())
    for i, (name, st) in enumerate(states, 1):
        print("| %d | %d:%02d | %d/%d | %d | %d | %d | %d | %d |  |"
              % (i, st["t"] // 60, st["t"] % 60, st["used"], st["cap"],
                 st["army"], st.get("gw", 0), st["mins"], st["idle"], st["workers"]))
    print("\n> 填完把这列贴回来，我用它当标准答案重跑基准。")


if __name__ == "__main__":
    main()
