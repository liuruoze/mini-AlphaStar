"""Quantify what function_calls[0] keeps vs a high-value-first selection rule."""
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

OUT = []
for r in rows:
    names = [x.get("function") for x in r["translated"] if x.get("ok")]
    kept0 = names[:1]
    high = [n for n in names if n in HIGH]
    low = [n for n in names if n in LOW]
    best = high[:1] if high else kept0
    OUT.append({
        "bot": r["bot"], "case": r["case"], "n": len(names),
        "names": names,
        "kept_by_index0": kept0,
        "index0_is_high": bool(kept0 and kept0[0] in HIGH),
        "high_count": len(high), "low_count": len(low),
        "kept_by_priority": best,
        "priority_is_high": bool(best and best[0] in HIGH),
    })

n_multi = sum(1 for o in OUT if o["n"] > 1)
i0_high = sum(1 for o in OUT if o["index0_is_high"])
pr_high = sum(1 for o in OUT if o["priority_is_high"])
lost_high = sum(o["high_count"] - (1 if o["index0_is_high"] else 0) for o in OUT)

print("=== groups=%d multi-action groups=%d ===" % (len(OUT), n_multi))
print("index[0] keeps a high-value action in %d/%d groups" % (i0_high, len(OUT)))
print("high-value-first keeps a high-value action in %d/%d groups" % (pr_high, len(OUT)))
print("high-value actions dropped by index[0] = %d" % lost_high)
print("=== per-group detail (multi-action only) ===")
for o in OUT:
    if o["n"] > 1:
        print("  %-9s %-16s n=%-2d first=%-24s high=%d low=%d priority=%s"
              % (o["bot"], o["case"], o["n"], str(o["kept_by_index0"][0])[:24],
                 o["high_count"], o["low_count"], o["kept_by_priority"]))
print("=== first-action histogram ===")
print(" ", dict(Counter(o["kept_by_index0"][0] for o in OUT if o["kept_by_index0"])))

with open(os.path.join(P, "action_priority.json"), "w") as f:
    json.dump({"summary": {"groups": len(OUT), "multi_action_groups": n_multi,
                           "index0_high": i0_high, "priority_high": pr_high,
                           "high_actions_dropped_by_index0": lost_high},
               "rows": OUT}, f, indent=2)
