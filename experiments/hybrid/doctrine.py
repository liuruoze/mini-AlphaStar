# -*- coding: utf-8 -*-
"""用户（项目作者）钦定的狂热者快攻策略，2026-11-20 由 33 条人工标注拟合而来。

来源：
  · LABEL_SHEET_20261120.md（20 条，盲标）
  · LABEL_BOUNDARY_20261120.md（13 条，每行只改一个变量）
拟合结果：**六条规则 33/33 全中**，没有例外。

与 V5 专家的差异（V5 是当前训练数据的老师）：
  | 阈值 | 用户 | V5 |
  | 人口余量 | ≤9 就补水晶 | ≤6 |
  | 农民 | ≤14 就补 | <18 就补 |
  | 兵营 | <3 必补；<4 且矿≥250 且兵<10 也补 | <4 一律补 |
  | 进攻 | 7:00 之后且兵≥10 | 兵≥10 随时 |
也就是说：**用户要的是"更早补人口、更少农民、更早开打"的快攻**，
而 V5（为了稳赢 very_easy 调的）会多补 3–4 个农民、多造 1 个兵营、提前开打。

用法：
    from doctrine import decide
    key = decide(t=420, used=50, cap=70, workers=25, army=6, gw=3, minerals=300)
"""

# 阈值（全部由标注确定；改动前先重新标定）
GAP_PYLON = 9        # 人口余量 ≤9 → 补水晶
WORKERS_MIN = 14     # 农民 ≤14 → 补农民
GATEWAY_MIN = 3      # 兵营 <3 → 补兵营
GATEWAY_MAX = 4      # 兵营 <4 且存款高且兵少 → 再补一个
BANK_FOR_GATEWAY = 250
ATTACK_TIME = 420    # 7:00 之后
ATTACK_ARMY = 10     # 且兵 ≥10 → 进攻


def decide(t, used, cap, workers, army, gw, minerals, **_):
    """返回 S 编号（与 commander.STRATEGIES 对应）。"""
    if cap - used <= GAP_PYLON:
        return "S2"          # 补水晶
    if workers <= WORKERS_MIN:
        return "S1"          # 补农民
    if gw < GATEWAY_MIN:
        return "S9"          # 补兵营（产能不够）
    if gw < GATEWAY_MAX and minerals >= BANK_FOR_GATEWAY and army < ATTACK_ARMY:
        return "S9"          # 存款高、兵少、还能再加一个兵营 → 把钱变成产能
    if t >= ATTACK_TIME and army >= ATTACK_ARMY:
        return "S6"          # 进攻
    return "S3"              # 量产狂热者


def explain(t, used, cap, workers, army, gw, minerals, **_):
    if cap - used <= GAP_PYLON:
        return "人口余量 %d ≤ %d，先补水晶" % (cap - used, GAP_PYLON)
    if workers <= WORKERS_MIN:
        return "农民 %d ≤ %d，先补农民" % (workers, WORKERS_MIN)
    if gw < GATEWAY_MIN:
        return "兵营 %d < %d，先补产能" % (gw, GATEWAY_MIN)
    if gw < GATEWAY_MAX and minerals >= BANK_FOR_GATEWAY and army < ATTACK_ARMY:
        return "存款 %d ≥ %d 且兵 %d < %d，再加兵营" % (
            minerals, BANK_FOR_GATEWAY, army, ATTACK_ARMY)
    if t >= ATTACK_TIME and army >= ATTACK_ARMY:
        return "%d 秒 ≥ %d 且兵 %d ≥ %d，开打" % (t, ATTACK_TIME, army, ATTACK_ARMY)
    return "以上都不满足，继续量产"
