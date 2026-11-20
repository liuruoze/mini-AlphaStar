#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把一帧观测压成给 LLM 的"状态摘要"（CoS 思路，自己实现）。

输入两种：
  1) --trace <jsonl>：用评测采样的时序文件（离线、无需 SC2）；
  2) 代码里调用 summarize_obs(obs)：接 pysc2 观测（pipeline 用）。

输出是一段 10 行以内的中文，稳定、可 diff，便于做 prompt 版本管理。
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from experiments.macro_metrics import _unit_name_map
except Exception:
    _unit_name_map = lambda: {}

NAMES = _unit_name_map()
TECH = ["Forge", "CyberneticsCore", "TwilightCouncil", "RoboticsFacility",
        "Stargate", "TemplarArchives", "FleetBeacon"]
CN = {
    "Pylon": "水晶", "Gateway": "兵营", "Nexus": "基地", "Assimilator": "气矿",
    "Forge": "锻炉", "CyberneticsCore": "控制核心", "TwilightCouncil": "暮光议会",
    "RoboticsFacility": "机械台", "Stargate": "星门", "TemplarArchives": "圣堂档案馆",
    "FleetBeacon": "舰队航标", "Zealot": "狂热者", "Stalker": "追猎者",
    "Adept": "使徒", "Sentry": "哨兵", "Immortal": "不朽", "Colossus": "巨像",
    "Probe": "农民", "VoidRay": "虚空辉光舰", "Phoenix": "凤凰",
}


def _cn(name):
    return CN.get(name, name or "?")


def summarize_record(rec):
    """rec 是 trace 的一行（dict）→ 中文摘要。"""
    own = rec.get("own", {}) or {}
    by_name = {}
    for uid, cnt in own.items():
        nm = NAMES.get(int(uid), str(uid))
        by_name[nm] = by_name.get(nm, 0) + int(cnt)
    buildings = {k: v for k, v in by_name.items()
                 if k in ("Pylon", "Gateway", "Nexus", "Assimilator", "Forge", "CyberneticsCore",
                          "TwilightCouncil", "RoboticsFacility", "Stargate", "TemplarArchives",
                          "FleetBeacon")}
    army = sum(v for k, v in by_name.items()
               if k not in buildings and k not in ("Probe",))
    tech = [t for t in TECH if by_name.get(t)]
    mins = rec.get("t_s", 0) / 60.0
    lines = [
        "时间 %d:%02d｜人口 %d/%d｜矿 %d 气 %d" % (
            int(mins), int(rec.get("t_s", 0)) % 60,
            rec.get("food_used", 0), rec.get("food_cap", 0),
            rec.get("minerals", 0), rec.get("vespene", 0)),
        "农民 %d｜军队 %d｜空闲农民 %d" % (
            by_name.get("Probe", 0), army, rec.get("idle_workers", -1)),
        "建筑: " + ("、".join("%s×%d" % (_cn(k), v) for k, v in sorted(buildings.items())) or "无"),
        "科技: " + ("、".join(_cn(t) for t in tech) or "无") + "（%d/%d）" % (len(tech), len(TECH)),
    ]
    return "\n".join(lines)


def summarize_obs(obs):
    """接 pysc2 观测（duck-typed）：字段缺失时降级，不抛异常。"""
    pl = getattr(obs, "player", None)
    rec = {
        "t_s": float(getattr(obs, "game_loop", [0])[0]) / 22.4,
        "food_used": int(getattr(pl, "food_used", 0) or 0),
        "food_cap": int(getattr(pl, "food_cap", 0) or 0),
        "minerals": int(getattr(pl, "minerals", 0) or 0),
        "vespene": int(getattr(pl, "vespene", 0) or 0),
        "idle_workers": int(getattr(pl, "idle_worker_count", -1) or -1),
        "own": {},
    }
    try:
        cnt = {}
        for u in obs.raw_units:
            if u.alliance == 1:
                cnt[str(int(u.unit_type))] = cnt.get(str(int(u.unit_type)), 0) + 1
        rec["own"] = cnt
    except Exception:
        pass
    return summarize_record(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="评测采样 .jsonl，取最后一行做演示")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.trace, errors="ignore") if l.strip()]
    print(summarize_record(rows[-1]))


if __name__ == "__main__":
    main()
