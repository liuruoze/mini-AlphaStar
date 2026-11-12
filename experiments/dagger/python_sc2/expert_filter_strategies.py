"""Compare candidate rules for which actions from one step enter training."""
import json
import os
from collections import Counter

P = "/home/liuruoze/bvb/expert_probe"
with open(os.path.join(P, "matrix_translate.json")) as f:
    rows = json.load(f)

HIGH = {
    "Build_Pylon_pt", "Build_Assimilator_unit", "Build_Gateway_pt",
    "Build_CyberneticsCore_pt", "Train_Probe_quick",
}
LOW = {"Harvest_Gather_unit"}


def strategies(names):
    high = [n for n in names if n in HIGH]
    return {
        "index0_current": names[:1],
        "all": list(names),
        "drop_harvest": [n for n in names if n not in LOW],
        "high_only": high,
        "high_first_one": high[:1] if high else names[:1],
    }


results = {k: {"n_actions": 0, "n_high": 0, "groups_with_high": 0}
           for k in ["index0_current", "all", "drop_harvest", "high_only", "high_first_one"]}
for r in rows:
    names = [x.get("function") for x in r["translated"] if x.get("ok")]
    for key, kept in strategies(names).items():
        results[key]["n_actions"] += len(kept)
        results[key]["n_high"] += sum(1 for n in kept if n in HIGH)
        results[key]["groups_with_high"] += int(any(n in HIGH for n in kept))

print("strategy            actions  high  high%%  groups_with_high(/9)")
for k, v in results.items():
    pct = 100.0 * v["n_high"] / max(v["n_actions"], 1)
    print("%-18s %5d  %5d  %5.1f%%  %d" % (k, v["n_actions"], v["n_high"], pct,
                                            v["groups_with_high"]))
with open(os.path.join(P, "filter_strategies.json"), "w") as f:
    json.dump(results, f, indent=2)
