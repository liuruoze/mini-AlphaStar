"""Quantitative cross-checks on existing products (no new experiments)."""
import json
import os
import pickle
from collections import Counter

P = "/home/liuruoze/bvb/expert_probe"


def j(name):
    with open(os.path.join(P, name)) as f:
        return json.load(f)


rows = j("state_sweep.json")
gate = [r for r in rows if r["build_supply_hit"]]
both = [r for r in gate if 881 in r["aids"]]
inv_supply = [r for r in rows if r["invest"] == "SUPPLY"]
inv_supply_881 = [r for r in inv_supply if 881 in r["aids"]]
zero = [r for r in rows if not r["aids"]]
zero_gate = [r for r in zero if r["build_supply_hit"]]

print("=== upper Advice vs lower concrete actions (90 states) ===")
print("build_supply_hit states        =", len(gate))
print("  ... of which produced 881    =", len(both))
print("invest=SUPPLY states           =", len(inv_supply))
print("  ... of which produced 881    =", len(inv_supply_881))
print("zero-action states             =", len(zero))
print("  ... with build_supply_hit    =", len(zero_gate))
for inv in ["SUPPLY", "ECONOMY", "ARMY"]:
    sub = [r for r in rows if r["invest"] == inv]
    c = Counter(a for r in sub for a in r["aids"])
    print("invest=%-8s n=%-3d actions=%s" % (inv, len(sub), dict(c)))

with open(os.path.join(P, "matrix_actions_voidray_C_tech_army.pkl"), "rb") as f:
    raws = pickle.load(f)
acts = []
for b in raws:
    from s2clientprotocol import sc2api_pb2 as sc_pb
    a = sc_pb.Action()
    a.ParseFromString(b)
    uc = a.action_raw.unit_command
    acts.append((uc.ability_id, tuple(uc.unit_tags)))
tags = [t for _, t in acts]
print("=== voidray C step: are the 12 Harvest_Gather actions duplicates? ===")
print("n_actions=", len(acts), "unique (ability,tags) pairs=", len(set(acts)))
print("ability histogram=", dict(Counter(a for a, _ in acts)))
print("distinct unit_tags=", len(set(tags)), "example tags=", tags[:6])

out = {
    "build_supply_hit_states": len(gate),
    "hit_and_881": len(both),
    "invest_supply_states": len(inv_supply),
    "invest_supply_and_881": len(inv_supply_881),
    "zero_action_states": len(zero),
    "zero_action_with_hit": len(zero_gate),
    "voidray_C_actions": len(acts),
    "voidray_C_unique_pairs": len(set(acts)),
    "voidray_C_distinct_tags": len(set(tags)),
}
with open(os.path.join(P, "consistency_analysis.json"), "w") as f:
    json.dump(out, f, indent=2)
