# -*- coding: utf-8 -*-
"""用户（项目作者）钦定的狂热者快攻战术 —— v2，2026-11-20 19:03 冻结。

来源：33 条人工标注拟合出的 v1（六条规则，33/33 全中），随后按用户口述修订：
  · 删掉"兵营<4 且矿≥250 且兵<10 → 再补兵营"（用户："意义不明"）
  · 进攻条件从"7:00 且兵≥10"改为 **兵 ≥ 8**（去掉时间门）
  · 新增前置条件：水晶 = 0 → 补水晶（兵营必须建在能量场里）

优先级（从高到低）：
  1. 水晶 = 0                     → 补水晶
  2. 人口余量 ≤ 9                 → 补水晶
  3. 农民 ≤ 14                    → 补农民
  4. 兵营 < 3                     → 补兵营
  5. 兵 ≥ 8                       → 全面进攻
  6. 其余                         → 量产狂热者

与 V5 老师的差异：农民 14（V5 16±1）；进攻门槛 8 且无时间门（V5 10±1、随时）；
V5 没有"水晶=0"这个前置判断。

用法：
    from doctrine import decide, CODES
    sid = decide(t=420, used=50, cap=70, workers=25, army=6, gw=3, minerals=300, pylons=2)
    code = CODE_OF[sid]          # one-hot，投给模型看/收模型答都用它
"""

# 阈值（改动前先重新标定）
GAP_PYLON = 9        # 人口余量 ≤9 → 补水晶
WORKERS_MIN = 14     # 农民 ≤14 → 补农民
GATEWAY_MIN = 3      # 兵营 <3 → 补兵营
ATTACK_ARMY = 8      # 兵 ≥8 → 全面进攻

# 冻结的 one-hot 编码（2026-11-20 用户指定：避免数字/字母的顺序先验）
CODES = {
    "00001": "S2",   # 补水晶
    "00010": "S1",   # 补农民
    "00100": "S9",   # 补兵营
    "01000": "S3",   # 量产狂热者
    "10000": "S6",   # 全面进攻
}
CODE_OF = {v: k for k, v in CODES.items()}

NAMES = {
    "S2": "补水晶", "S1": "补农民", "S9": "补兵营",
    "S3": "量产狂热者", "S6": "全面进攻",
}


def decide(t, used, cap, workers, army, gw, minerals, pylons=1, **_):
    """返回策略编号（S1/S2/S9/S3/S6）。pylons 缺省 1，表示"至少有一个水晶"。"""
    if pylons <= 0:
        return "S2"                     # 没有能量场，任何建造都不成立
    if cap - used <= GAP_PYLON:
        return "S2"
    if workers <= WORKERS_MIN:
        return "S1"
    if gw < GATEWAY_MIN:
        return "S9"
    if army >= ATTACK_ARMY:
        return "S6"
    return "S3"


# ---- v3：五位独立判断（2026-11-20 用户改为多选）----
# 顺序固定：1 补水晶 / 2 补农民 / 3 补兵营 / 4 量产狂热者 / 5 全面进攻
BITS = ["pylon", "probe", "gateway", "zealot", "attack"]
BIT_NAMES = ["补水晶", "补农民", "补兵营", "量产狂热者", "全面进攻"]

# 量产的条件（用户 19:09 指定）：兵<24 且 矿≥100 且 农民≥10 且 兵营≥1 且 余量≥2
ZEALOT_MAX = 24
ZEALOT_MINERALS = 100
ZEALOT_WORKERS = 10


def decide_bits(t, used, cap, workers, army, gw, minerals, pylons=1, **_):
    """返回 5 位 0/1 串。可同时为 1（基地造农民、兵营造兵、农民造建筑互不冲突）。

    唯一的依赖：没有水晶就没有能量场，兵营放不下去 → 第 3 位强制 0。
    """
    gap = cap - used
    b1 = 1 if (pylons <= 0 or gap <= GAP_PYLON) else 0
    b2 = 1 if workers <= WORKERS_MIN else 0
    b3 = 1 if (gw < GATEWAY_MIN and pylons > 0) else 0
    b4 = 1 if (army < ZEALOT_MAX and minerals >= ZEALOT_MINERALS
               and workers >= ZEALOT_WORKERS and gw >= 1 and gap >= 2) else 0
    b5 = 1 if army >= ATTACK_ARMY else 0
    return "%d%d%d%d%d" % (b1, b2, b3, b4, b5)


def explain_bits(t, used, cap, workers, army, gw, minerals, pylons=1, **_):
    return [b == "1" for b in decide_bits(t, used, cap, workers, army, gw, minerals, pylons)]


def explain(t, used, cap, workers, army, gw, minerals, pylons=1, **_):
    if pylons <= 0:
        return "没有任何水晶，兵营无处可建，必须先补水晶"
    if cap - used <= GAP_PYLON:
        return "人口余量 %d ≤ %d，先补水晶" % (cap - used, GAP_PYLON)
    if workers <= WORKERS_MIN:
        return "农民 %d ≤ %d，先补农民" % (workers, WORKERS_MIN)
    if gw < GATEWAY_MIN:
        return "兵营 %d < %d，先补产能" % (gw, GATEWAY_MIN)
    if army >= ATTACK_ARMY:
        return "兵 %d ≥ %d，全面进攻" % (army, ATTACK_ARMY)
    return "以上都不满足，继续量产"
