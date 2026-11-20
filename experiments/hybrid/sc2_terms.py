# -*- coding: utf-8 -*-
"""神族术语的别名解析：官方简体 / 官方繁体 / 玩家俗称 / 英文 —— 都认成同一个单位。

为什么要它：同一个东西有三种叫法（官方「传送门」、口语「兵营」、英文「Gateway」），
prompt 里用口语、日志里可能写官方名，一旦代码里各写各的就会像 2026-11-20 那次
把 Stalker 写成「使徒」一样出错。这里统一解析。

用法：
    from sc2_terms import resolve, canonical_cn, ALIASES
    resolve("兵营")      -> "Gateway"
    resolve("傳送之門")  -> "Gateway"
    canonical_cn("Gateway", style="colloquial") -> "兵营"
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE = os.path.join(_HERE, "sc2_terms_protoss.json")

_DATA = json.load(open(_TABLE, encoding="utf-8"))
TERMS = _DATA["terms"]

# 英文名 -> 词条
BY_EN = {t["en"]: t for t in TERMS}

# 别名 -> 英文名（英文、官方简体、官方繁体、玩家俗称、以及去空格的写法）
ALIASES = {}
for t in TERMS:
    for key in (t["en"], t["cn"], t.get("tw", ""), t.get("colloquial", "")):
        if not key:
            continue
        ALIASES[key] = t["en"]
        ALIASES[key.replace(" ", "")] = t["en"]
    # 口语里常见的简写
ALIASES.update({
    "水晶": "Pylon", "兵营": "Gateway", "基地": "Nexus", "气矿": "Assimilator",
    "农民": "Probe", "追猎": "Stalker", "叉子": "Zealot", "白球": "Archon",
    "OB": "Observer", "叮当": "Observer", "棱镜": "WarpPrism",
})


def resolve(name):
    """任意叫法 -> 英文规范名；认不出返回 None。"""
    if not name:
        return None
    n = str(name).strip()
    return ALIASES.get(n) or ALIASES.get(n.replace(" ", "")) or None


def canonical_cn(en, style="colloquial"):
    """英文规范名 -> 中文。style: colloquial（口语，默认）/ official（官方简体）/ tw（官方繁体）。"""
    t = BY_EN.get(en)
    if not t:
        return None
    if style == "official":
        return t["cn"]
    if style == "tw":
        return t.get("tw")
    return t.get("colloquial") or t["cn"]


def check_all():
    """自检：每个词条的所有叫法都能解析回自己。"""
    bad = []
    for t in TERMS:
        for key in (t["en"], t["cn"], t.get("tw", ""), t.get("colloquial", "")):
            if key and resolve(key) != t["en"]:
                bad.append((key, resolve(key), t["en"]))
    return bad


if __name__ == "__main__":
    print("词条 %d，别名 %d" % (len(TERMS), len(ALIASES)))
    for k in ["兵营", "傳送之門", "Gateway", "水晶", "追猎者", "使徒", "Adept"]:
        print("  %-8s -> %-14s (口语=%s)"
              % (k, resolve(k), canonical_cn(resolve(k)) if resolve(k) else "-"))
    bad = check_all()
    print("自检：%s" % ("全部通过" if not bad else bad))
