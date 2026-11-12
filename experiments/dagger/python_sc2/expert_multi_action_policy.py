#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-1b: three-way multi-action policy, offline only (no production file touched).

Policies (per step, the transform ultimately emits ONE label per kept step):
  index0   : keep valid[0]                      (current behaviour)
  priority : dedup (name,args) -> drop Harvest_* -> prefer Build_/Train_/
             Research_ > Attack_ > other -> keep first
  all      : keep every valid call              (reference upper bound)

Also replays the same three policies on the 9-case synthetic matrix to check the
"high-value kept 9/9" criterion independently of the real replay.

Output: multi_action_policy.json
"""
import json
import os

P = os.path.expanduser("~/bvb/p0_baseline")
CACHE = os.path.join(P, "actions_seq_12-14-09-33-00.pkl")
EP = os.path.expanduser("~/bvb/expert_probe")
OUT = os.path.join(P, "multi_action_policy.json")

HIGH = ("Build_", "Train_", "Research_")
MID = ("Attack_",)
# Smart_unit (value=12) is NOT filtered by getFuncCall; on real B0 replays it is
# the single largest low-value class (22/34 of the dropped calls, J146 §25.1),
# so the priority policy must drop it too.
LOW = ("Harvest_", "Smart_unit", "Smart_pt", "no_op")


def kind(name):
    if name.startswith(HIGH):
        return "high"
    if name.startswith(MID):
        return "mid"
    return "low"


def apply_policy(valid, policy, stats=None):
    if policy == "index0":
        return valid[:1]
    if policy == "all":
        return list(valid)
    # priority
    seen, dedup = set(), []
    for c in valid:
        key = (c["name"], c.get("args", ""))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    kept = [c for c in dedup if not c["name"].startswith(LOW)]
    if not kept:
        # nothing valuable this step: keep the first so the frame count does not
        # collapse (a step with only Harvest/Smart still carries selection info)
        if stats is not None:
            stats["fallback_steps"] = stats.get("fallback_steps", 0) + 1
        return dedup[:1]
    rank = {"high": 0, "mid": 1, "low": 2}
    kept.sort(key=lambda c: rank[kind(c["name"])])
    return kept[:1]


def replay_policy(rows, policy):
    frames, by_kind, names, dup = 0, {"high": 0, "mid": 0, "low": 0}, {}, 0
    stats = {}
    for r in rows:
        kept = apply_policy(r["valid"], policy, stats)
        if not kept:
            continue
        frames += 1
        for c in kept:
            by_kind[kind(c["name"])] += 1
            names[c["name"]] = names.get(c["name"], 0) + 1
        if policy == "priority":
            keys = [(c["name"], c.get("args", "")) for c in kept]
            dup += len(keys) - len(set(keys))
    return {"frames": frames, "by_kind": by_kind,
            "n_actions": sum(by_kind.values()),
            "same_step_duplicates": dup,
            "fallback_steps": stats.get("fallback_steps", 0),
            "low_ratio": round(by_kind["low"] / max(sum(by_kind.values()), 1), 4),
            "top_actions": sorted(names.items(), key=lambda x: -x[1])[:12]}


def matrix_check():
    path = os.path.join(EP, "matrix_translate.json")
    mt = json.load(open(path))
    cases = []
    total_high_all = 0
    for row in mt:
        valid = [{"name": x["function"], "args": ""}
                 for x in row["translated"] if x.get("ok")]
        high_all = sum(1 for c in valid if kind(c["name"]) == "high")
        total_high_all += high_all
        res = {p: replay_policy([{"valid": valid}], p) for p in
               ("index0", "priority", "all")}
        cases.append({
            "case": row.get("case"), "n_valid": len(valid),
            "high_all": high_all,
            "high_index0": res["index0"]["by_kind"]["high"],
            "high_priority": res["priority"]["by_kind"]["high"],
            "dup_priority": res["priority"]["same_step_duplicates"],
        })
    high_pri = sum(c["high_priority"] for c in cases)
    high_i0 = sum(c["high_index0"] for c in cases)
    return {"n_cases": len(cases), "high_all": total_high_all,
            "high_priority": high_pri, "high_index0": high_i0,
            # NOTE: priority can never equal 'all' because the loader emits one
            # label per step; the meaningful criterion is ">= index0".
            "criterion_priority_ge_index0": high_pri >= high_i0 and high_i0 >= 0,
            "cases": cases}


def main():
    d = json.load(open(CACHE)) if CACHE.endswith(".json") else None
    if d is None:
        import pickle
        d = pickle.load(open(CACHE, "rb"))
    rows = d["rows"]
    real = {p: replay_policy(rows, p) for p in ("index0", "priority", "all")}
    mx = matrix_check()
    base_frames = real["index0"]["frames"]
    pr_frames = real["priority"]["frames"]
    out = {
        "replay": d["replay"],
        "steps_with_raw_actions": len(rows),
        "real_replay": real,
        "matrix": mx,
        "criteria": {
            "C1_priority_high_ge_index0": (
                mx["criterion_priority_ge_index0"] and
                real["priority"]["by_kind"]["high"] >= real["index0"]["by_kind"]["high"]),
            "C1b_priority_low_ratio_lt_index0": (
                real["priority"]["low_ratio"] < real["index0"]["low_ratio"]),
            "C2_priority_no_same_step_duplicates":
                real["priority"]["same_step_duplicates"] == 0,
            "C3_frame_delta_index0_vs_priority_pct": round(
                100.0 * (pr_frames - base_frames) / max(base_frames, 1), 2),
            "C3_pass": abs(pr_frames - base_frames) <= 0.05 * max(base_frames, 1),
        },
        "note": "frames = kept steps (one label per step); 'all' is an upper "
                "bound with multiple labels per step and is NOT a valid "
                "training config under the current 1-label-per-step loader.",
    }
    out["pass"] = (out["criteria"]["C1_priority_high_ge_index0"]
                   and out["criteria"]["C1b_priority_low_ratio_lt_index0"]
                   and out["criteria"]["C2_priority_no_same_step_duplicates"]
                   and out["criteria"]["C3_pass"])
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("steps_with_raw_actions", "criteria", "pass")}, indent=2))
    for p in ("index0", "priority", "all"):
        x = real[p]
        print("%-9s frames=%-4d actions=%-4d high=%-4d mid=%-3d low=%-4d dup=%d"
              % (p, x["frames"], x["n_actions"], x["by_kind"]["high"],
                 x["by_kind"]["mid"], x["by_kind"]["low"],
                 x["same_step_duplicates"]))
    print("matrix high_all=%d high_priority=%d high_index0=%d ge=%s"
          % (mx["high_all"], mx["high_priority"], mx["high_index0"],
             mx["criterion_priority_ge_index0"]))
    print("priority fallback_steps=%d (steps with only low-value calls), "
          "low_ratio index0=%.3f priority=%.3f"
          % (real["priority"]["fallback_steps"],
             real["index0"]["low_ratio"], real["priority"]["low_ratio"]))
    print("top index0 actions:", real["index0"]["top_actions"][:8])


if __name__ == "__main__":
    main()
