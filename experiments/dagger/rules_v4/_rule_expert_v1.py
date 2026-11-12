#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAgger 规则专家 v1（HY26 / 2026-11-08）

版本串：rule_v1_20261108
职责：只判断「该不该干预、干预成什么动作」，不构造标签。
标签由 _dagger_build.py 用真实数据的同类动作模板生成，保证六段 one-hot 合法。

输入：summ = {
    "food_used": int, "food_cap": int,
    "n_probe": int, "n_pylon": int, "n_gateway": int, "n_nexus": int,
    "n_army": int, "minerals": float,
}
输出：(intervene: bool, action_name: str|None, reason: str)
"""
from __future__ import annotations

VERSION = "rule_v1_20261108"

# SC2 unit_type（raw_units 里的整数）
NEXUS = 59
PYLON = 60
ASSIMILATOR = 61
GATEWAY = 62
PROBE = 84

ACTIONS = {
    "build_pylon": "Build_Pylon_pt",
    "build_gateway": "Build_Gateway_pt",
    "train_stalker": "Train_Stalker_quick",
    "train_zealot": "Train_Zealot_quick",
    "attack": "Attack_pt",
}


def decide(summ):
    """严格最小干预：只在明确缺口上给动作，其余返回不干预。"""
    food_used = int(summ.get("food_used", 0))
    food_cap = int(summ.get("food_cap", 0))
    n_pylon = int(summ.get("n_pylon", 0))
    n_gateway = int(summ.get("n_gateway", 0))
    n_army = int(summ.get("n_army", 0))
    n_probe = int(summ.get("n_probe", 0))
    minerals = float(summ.get("minerals", 0.0))

    # 1) 卡人口：最高优先级（探针实测模型在此状态仍选 Train_Probe）
    if food_cap > 0 and food_used >= food_cap and n_pylon == 0:
        return True, ACTIONS["build_pylon"], "supply_blocked_no_pylon"

    # 2) 有电没兵营（observation 无法可靠读取矿物，v1 按结构决策，不设矿门槛）
    if n_gateway == 0 and n_pylon > 0:
        return True, ACTIONS["build_gateway"], "no_gateway"

    # 3) 有兵营且人口有余 → 造兵
    if n_gateway > 0 and food_cap - food_used >= 2:
        return True, ACTIONS["train_stalker"], "train_army"

    # 4) 有兵 → 进攻
    if n_army >= 6:
        return True, ACTIONS["attack"], "attack_with_army"

    return False, None, "keep_policy"


def summarize(obs):
    """从 pysc2 feature observation 造 summ，不依赖 player 段（实测该段是缩放值，不可靠）。"""
    ru = obs.get("raw_units", [])
    mine = [u for u in ru if int(u["alliance"]) == 1]
    cnt = {}
    for u in mine:
        t = int(u["unit_type"])
        cnt[t] = cnt.get(t, 0) + 1
    n_probe = cnt.get(PROBE, 0)
    n_pylon = cnt.get(PYLON, 0)
    n_gateway = cnt.get(GATEWAY, 0)
    n_nexus = cnt.get(NEXUS, 0)
    # 建筑不算人口；战斗单位按 2 人口粗算，probe 按 1
    buildings = {NEXUS, PYLON, ASSIMILATOR, GATEWAY}
    n_army = sum(v for k, v in cnt.items() if k not in buildings and k != PROBE)
    food_used = n_probe + 2 * n_army
    food_cap = 15 * n_nexus + 8 * n_pylon
    return {
        "food_used": food_used,
        "food_cap": food_cap,
        "n_probe": n_probe,
        "n_pylon": n_pylon,
        "n_gateway": n_gateway,
        "n_nexus": n_nexus,
        "n_army": n_army,
        "minerals": 0.0,  # v1 不读取矿物（observation 该段是缩放值）；规则按结构决策
        "n_units": len(mine),
    }


if __name__ == "__main__":
    cases = [
        dict(food_used=15, food_cap=15, n_pylon=0, n_gateway=0, n_army=0, n_probe=15, minerals=800),
        dict(food_used=15, food_cap=23, n_pylon=1, n_gateway=0, n_army=0, n_probe=15, minerals=800),
        dict(food_used=17, food_cap=23, n_pylon=1, n_gateway=1, n_army=0, n_probe=15, minerals=800),
        dict(food_used=17, food_cap=23, n_pylon=1, n_gateway=1, n_army=8, n_probe=9, minerals=100),
        dict(food_used=10, food_cap=23, n_pylon=1, n_gateway=1, n_army=0, n_probe=10, minerals=50),
    ]
    for c in cases:
        ok, act, why = decide(c)
        print("%-28s -> %-6s %-24s %s" % (why, ok, act or "-", c))
