#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge the two read-only replay scans into one reproducible summary.

IMPORTANT: the two samples are NOT independent random samples of the 90 games:
 - replay_action_scan.json      : one replay per recording date (12 dates)
 - replay_action_scan_extra.json: 6 targeted replays, 4 of them deliberately
                                  from the two concurrent batches
So the merged rate is a descriptive number for the 18 sampled games, NOT an
unbiased estimate for all 90 games. The stratified 12-date rate (2.74%) is the
closer-to-unbiased figure.
"""
import json
import os

P = os.path.expanduser("~/bvb/expert_probe")
KEYS = ["steps", "steps_with_actions", "total_actions", "multi_action_steps",
        "valid_actions", "kept_by_index0", "dropped_valid",
        "multi_valid_steps", "first_filtered_steps",
        "replays_with_multi_action"]

a = json.load(open(os.path.join(P, "replay_action_scan.json")))
b = json.load(open(os.path.join(P, "replay_action_scan_extra.json")))
rows = a["rows"] + b["rows"]
merged = {k: a["total"][k] + b["total"][k] for k in KEYS}
merged["n_replays"] = a["total"]["n_replays"] + b["total"]["n_replays"]
merged["drop_rate_of_valid"] = round(
    100.0 * merged["dropped_valid"] / merged["valid_actions"], 2)
merged["pct_action_steps_first_filtered"] = round(
    100.0 * merged["first_filtered_steps"] / merged["steps_with_actions"], 2)
merged["errors"] = a["total"].get("errors", []) + b["total"].get("errors", [])

by_date = {}
for r in rows:
    d = r["replay"].split("_")[1][:10]
    g = by_date.setdefault(d, {"replays": 0, "valid": 0, "dropped": 0,
                               "multi_action_steps": 0})
    g["replays"] += 1
    g["valid"] += r["valid_actions"]
    g["dropped"] += r["dropped_valid"]
    g["multi_action_steps"] += r["multi_action_steps"]

out = {
    "stratified_12date": {"total": a["total"],
                          "drop_rate": round(100.0 * a["total"]["dropped_valid"]
                                             / a["total"]["valid_actions"], 2)},
    "targeted_extra_6": {"total": b["total"],
                         "drop_rate": round(100.0 * b["total"]["dropped_valid"]
                                           / b["total"]["valid_actions"], 2)},
    "merged_18_sampled": merged,
    "by_date": by_date,
    "caveat": "18 sampled games, NOT all 90; the 6 extra replays deliberately "
              "oversample the two concurrent batches, so 4.15% is biased "
              "upward. The 12-date stratified 2.74% is the better estimate.",
}
with open(os.path.join(P, "replay_action_scan_merged.json"), "w") as f:
    json.dump(out, f, indent=2)
print("12-date stratified : valid=%d dropped=%d (%.2f%%) replays_with_multi=%d"
      % (a["total"]["valid_actions"], a["total"]["dropped_valid"],
         out["stratified_12date"]["drop_rate"],
         a["total"]["replays_with_multi_action"]))
print("targeted extra 6   : valid=%d dropped=%d (%.2f%%) replays_with_multi=%d"
      % (b["total"]["valid_actions"], b["total"]["dropped_valid"],
         out["targeted_extra_6"]["drop_rate"],
         b["total"]["replays_with_multi_action"]))
print("merged 18 sampled  : valid=%d dropped=%d (%.2f%%) multi_action_steps=%d first_filt=%.1f%%"
      % (merged["valid_actions"], merged["dropped_valid"],
         merged["drop_rate_of_valid"], merged["multi_action_steps"],
         merged["pct_action_steps_first_filtered"]))
print("by date:")
for d, g in sorted(by_date.items()):
    print("  %s replays=%d multi_steps=%-5d valid=%-5d dropped=%d"
          % (d, g["replays"], g["multi_action_steps"], g["valid"], g["dropped"]))
