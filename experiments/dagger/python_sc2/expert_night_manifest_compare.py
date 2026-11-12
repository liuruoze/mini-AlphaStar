"""Relate this round's action-type filtering to the J145 night dataset."""
import json
import os
from collections import Counter

D = "/home/liuruoze/mini-AlphaStar/data"
N = D + "/replay_data_m30_dagger_hier/manifest.json"
B = D + "/replay_data_m30_dagger_hier_balanced/manifest.json"

n = json.load(open(N))
b = json.load(open(B))
before = b.get("expert_action_before", {})
after = b.get("expert_action_after", {})

LOW = {"Harvest_Gather_unit"}
REPEAT = {"Build_Assimilator_unit", "Train_Probe_quick"}


def report(tag, d):
    tot = sum(d.values())
    rep = sum(d.get(k, 0) for k in REPEAT)
    low = sum(d.get(k, 0) for k in LOW)
    other = tot - rep - low
    print("%-12s total=%-5d repeat(Assim+Probe)=%-5d (%.1f%%) harvest=%-3d other_build_train=%-5d (%.1f%%)"
          % (tag, tot, rep, 100.0 * rep / max(tot, 1), low, other,
             100.0 * other / max(tot, 1)))
    return {"total": tot, "repeat": rep, "harvest": low,
            "other": other, "other_pct": 100.0 * other / max(tot, 1)}


print("=== J145 night expert frames (manifest) ===")
r_before = report("original", before)
r_after = report("balanced", after)

print("=== effect of THIS round's filter strategies on the ORIGINAL distribution ===")
drop_harvest = {k: v for k, v in before.items() if k not in LOW}
r_drop = report("drop_harvest", drop_harvest)
drop_repeat = {k: v for k, v in before.items() if k not in LOW and k not in REPEAT}
r_drop_repeat = report("drop_low+repeat", drop_repeat)

print("=== conclusion inputs ===")
print("J145 quota downsampling raised other-build/train from %.1f%% to %.1f%%"
      % (r_before["other_pct"], r_after["other_pct"]))
print("drop_harvest alone only removes %d frames (%.1f%% of original)"
      % (r_before["harvest"], 100.0 * r_before["harvest"] / max(r_before["total"], 1)))
print("=> conversion-layer filtering (this round) and dataset-layer quota (J145) "
      "are complementary, not substitutes.")

with open("/home/liuruoze/bvb/expert_probe/night_manifest_compare.json", "w") as f:
    json.dump({"original": r_before, "balanced": r_after,
               "drop_harvest": r_drop, "drop_low_repeat": r_drop_repeat,
               "before_actions": before, "after_actions": after}, f, indent=2)
