"""Minimal demo: strategy_engine GameState -> Advice, no SC2, no live bot."""
import json
import os
import sys

SRC = "/home/liuruoze/bvb/expert_probe/sc2_bots_src"
sys.path.insert(0, SRC)

from strategy_engine import StrategicAdvisor  # noqa: E402
from strategy_engine.state import GameState as SEState  # noqa: E402

CASES = {
    "A_supply_blocked": dict(game_time=100, worker_count=12, base_count=1,
                             minerals=500, supply_used=13, supply_cap=14,
                             supply_left=1, production_structures=1),
    "B_supply_ok": dict(game_time=100, worker_count=12, base_count=1,
                        minerals=500, supply_used=12, supply_cap=26,
                        supply_left=14, production_structures=1),
    "C_tech_army": dict(game_time=400, worker_count=12, base_count=1,
                        minerals=1000, vespene=300, supply_used=24,
                        supply_cap=30, supply_left=6, production_structures=2,
                        tech_structures=1, army_supply=8, enemy_army_supply=4),
}

ADV = StrategicAdvisor()
OUT = []
for name, kw in CASES.items():
    st = SEState(**kw)
    advice = ADV.advise(st)
    hits = [getattr(h, "rule", str(h)) for h in getattr(advice, "rule_hits", [])]
    item = {
        "case": name,
        "summary": advice.summary(),
        "rule_hits": [str(h) for h in hits],
        "top_investment": str(getattr(getattr(advice, "investment", None), "top", None)),
        "efficiency": str(getattr(advice, "efficiency", None))[:120],
    }
    OUT.append(item)
    print("[%s] %s" % (name, item["summary"].replace("\n", " | ")[:220]), flush=True)
    print("     rule_hits=%s" % item["rule_hits"], flush=True)

with open(os.path.join(os.path.dirname(SRC), "strategy_advisor_demo.json"), "w") as f:
    json.dump(OUT, f, indent=2)
