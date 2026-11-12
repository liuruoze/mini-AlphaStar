#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot integrity check for all read-only products of this investigation.

Fails loudly on: duplicate replay rows, scan errors, missing/empty products,
cross-env digest mismatch, or inconsistent merged totals.
"""
import hashlib
import json
import os
import sys

P = os.path.expanduser("~/bvb/expert_probe")
SCANS = ["replay_action_scan.json", "replay_action_scan_extra.json",
         "replay_action_scan_census.json"]
PRODUCTS = [
    "action_loss_compare.json", "replay_multivalid_detail.json",
    "replay_action_error_probe.json", "feature_layer_probe.json",
    "selection_probe.json", "build_executor_demo.json",
    "three_layer_demo.json", "bot_strength_evidence.json",
    "protoss_profiles.json", "replay_action_scan_merged.json",
    "replay_action_scan_merged_all.json", "matrix_translate.json",
    "action_roundtrip.json", "state_sweep.json", "filter_strategies.json",
]


def j(name):
    with open(os.path.join(P, name)) as f:
        return json.load(f)


def main():
    problems = []
    rows, totals = [], []
    for s in SCANS:
        d = j(s)
        rows.extend(d["rows"])
        totals.append(d["total"])
        if d["total"].get("errors"):
            problems.append("%s has errors: %s" % (s, d["total"]["errors"]))
    names = [r["replay"] for r in rows]
    if len(names) != len(set(names)):
        problems.append("duplicate replay rows: %d" % (len(names) - len(set(names))))
    n_games = sum(t["n_replays"] for t in totals)
    n_steps = sum(t["steps"] for t in totals)
    ma = j("replay_action_scan_merged_all.json")
    if ma["merged_20_games"]["n_replays"] != n_games:
        problems.append("merged n_replays %d != %d"
                        % (ma["merged_20_games"]["n_replays"], n_games))
    if ma["merged_20_games"]["steps"] != n_steps:
        problems.append("merged steps %d != %d"
                        % (ma["merged_20_games"]["steps"], n_steps))
    for name in PRODUCTS:
        p = os.path.join(P, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            problems.append("missing/empty product: %s" % name)
    a = open(os.path.join(P, "crossenv_sc2.json"), "rb").read()
    b = open(os.path.join(P, "crossenv_pysc2.json"), "rb").read()
    if a != b:
        problems.append("crossenv digests differ")
    b16 = j("build_executor_demo.json")["summary"]
    out = {
        "ok": not problems,
        "n_scans": len(SCANS), "n_games_sampled": n_games,
        "n_steps_observed": n_steps,
        "unique_replays": len(set(names)),
        "merged_20_drop_rate": ma["merged_20_games"]["drop_rate_of_valid"],
        "buckets": {k: [v["games_with_multi"], v["replays"]]
                    for k, v in ma["buckets"].items()},
        "build_executor": [b16["n_protoss_guides"], b16["total_walked"],
                           b16["total_with_ability"]],
        "crossenv_sha256": hashlib.sha256(a).hexdigest()[:16],
        "products_checked": len(PRODUCTS),
        "problems": problems,
    }
    with open(os.path.join(P, "probe_integrity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
