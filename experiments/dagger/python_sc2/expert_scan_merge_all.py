#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge ALL three read-only replay scans and classify by recording batch.

Samples (20 distinct games out of the 90):
  A replay_action_scan.json        : 12 games, one per recording date
  B replay_action_scan_extra.json  : 6 games (4 from concurrent batches + 2 control)
  C replay_action_scan_census.json : 2 games (remaining 12-14 batch)
Union of the two concurrent batches 12-13 (3 games) + 12-14 (5 games) = 8/8
is fully covered -> a complete census of those batches, not a sample.
"""
import json
import os

P = os.path.expanduser("~/bvb/expert_probe")
FILES = ["replay_action_scan.json", "replay_action_scan_extra.json",
         "replay_action_scan_census.json"]
KEYS = ["steps", "steps_with_actions", "total_actions", "multi_action_steps",
        "valid_actions", "kept_by_index0", "dropped_valid",
        "multi_valid_steps", "first_filtered_steps",
        "replays_with_multi_action"]

rows, parts = [], {}
for fn in FILES:
    d = json.load(open(os.path.join(P, fn)))
    parts[fn] = d["total"]
    rows.extend(d["rows"])
names = [r["replay"] for r in rows]
assert len(names) == len(set(names)), "duplicate replay in merged rows"

merged = {k: sum(d[k] for d in parts.values()) for k in KEYS}
merged["n_replays"] = sum(d["n_replays"] for d in parts.values())
merged["drop_rate_of_valid"] = round(
    100.0 * merged["dropped_valid"] / merged["valid_actions"], 2)
merged["pct_action_steps_first_filtered"] = round(
    100.0 * merged["first_filtered_steps"] / merged["steps_with_actions"], 2)
merged["errors"] = [e for d in parts.values() for e in d.get("errors", [])]

CONCURRENT = {"2021-12-13", "2021-12-14"}
by_date, buckets = {}, {"concurrent_batches": {"replays": 0, "games_with_multi": 0},
                        "other_dates": {"replays": 0, "games_with_multi": 0}}
for r in rows:
    date = r["replay"].split("_")[1][:10]
    g = by_date.setdefault(date, {"replays": 0, "valid": 0, "dropped": 0,
                                  "multi_action_steps": 0,
                                  "games_with_multi": 0})
    g["replays"] += 1
    g["valid"] += r["valid_actions"]
    g["dropped"] += r["dropped_valid"]
    g["multi_action_steps"] += r["multi_action_steps"]
    g["games_with_multi"] += int(r["multi_action_steps"] > 0)
    b = buckets["concurrent_batches"] if date in CONCURRENT else buckets["other_dates"]
    b["replays"] += 1
    b["games_with_multi"] += int(r["multi_action_steps"] > 0)

for name, b in buckets.items():
    sub = [r for r in rows
           if ((r["replay"].split("_")[1][:10] in CONCURRENT) ==
               (name == "concurrent_batches"))]
    b["valid_actions"] = sum(r["valid_actions"] for r in sub)
    b["dropped_valid"] = sum(r["dropped_valid"] for r in sub)
    b["multi_action_steps"] = sum(r["multi_action_steps"] for r in sub)
    b["drop_rate"] = round(100.0 * b["dropped_valid"] / max(b["valid_actions"], 1), 2)

out = {
    "parts": {k: {"n_replays": v["n_replays"], "valid": v["valid_actions"],
                  "dropped": v["dropped_valid"],
                  "drop_rate": round(100.0 * v["dropped_valid"]
                                     / max(v["valid_actions"], 1), 2)}
              for k, v in parts.items()},
    "merged_20_games": merged,
    "by_date": by_date,
    "buckets": buckets,
    "census_note": "12-13 (3/3) + 12-14 (5/5) = 8/8 concurrent games fully "
                   "sampled; all other 10 dates (12 games) have zero multi-action steps.",
    "caveat": "20 of 90 games, deliberately oversampled on the two concurrent "
              "batches; 4.15%-class merged rates are NOT unbiased estimates "
              "for all 90. The 12-date stratified 2.74% is the closest to unbiased.",
}
with open(os.path.join(P, "replay_action_scan_merged_all.json"), "w") as f:
    json.dump(out, f, indent=2)

print("parts:")
for k, v in out["parts"].items():
    print("  %-38s n=%-3d valid=%-5d dropped=%-4d rate=%.2f%%"
          % (k, v["n_replays"], v["valid"], v["dropped"], v["drop_rate"]))
print("MERGED 20 games: steps=%d actions=%d multi_steps=%d valid=%d dropped=%d (%.2f%%) first_filt=%.1f%%"
      % (merged["steps"], merged["total_actions"], merged["multi_action_steps"],
         merged["valid_actions"], merged["dropped_valid"],
         merged["drop_rate_of_valid"], merged["pct_action_steps_first_filtered"]))
print("buckets:")
for k, v in buckets.items():
    print("  %-20s games_with_multi=%d/%d valid=%-5d dropped=%-4d rate=%.2f%% multi_steps=%d"
          % (k, v["games_with_multi"], v["replays"], v["valid_actions"],
             v["dropped_valid"], v["drop_rate"], v["multi_action_steps"]))
print("by date:")
for d, g in sorted(by_date.items()):
    print("  %s games=%d multi_games=%d multi_steps=%-5d valid=%-5d dropped=%d"
          % (d, g["replays"], g["games_with_multi"], g["multi_action_steps"],
             g["valid"], g["dropped"]))
