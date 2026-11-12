#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only comparison: how much does function_calls[0] lose on (a) the real
B0 90-game DAgger replays vs (b) our newly wrapped python-sc2 expert?

Inputs are products already written by other scripts. No SC2, no training.
"""
import json
import os

P = os.path.expanduser("~/bvb/expert_probe")


def j(name):
    with open(os.path.join(P, name)) as f:
        return json.load(f)


def main():
    scan = j("replay_action_scan.json")
    rows = scan["rows"]
    total = scan["total"]
    real = {
        "source": "real B0 90-game DAgger replays, one per recording date",
        "n_replays": total["n_replays"],
        "steps": total["steps"],
        "steps_with_actions": total["steps_with_actions"],
        "total_actions": total["total_actions"],
        "multi_action_steps": total["multi_action_steps"],
        "first_filtered_steps": total["first_filtered_steps"],
        "valid_actions": total["valid_actions"],
        "multi_valid_steps": total["multi_valid_steps"],
        "dropped_valid": total["dropped_valid"],
        "drop_rate_of_valid": round(
            100.0 * total["dropped_valid"] / max(total["valid_actions"], 1), 2),
        "pct_action_steps_first_filtered": round(
            100.0 * total["first_filtered_steps"]
            / max(total["steps_with_actions"], 1), 2),
        "replays_with_multi_action": total["replays_with_multi_action"],
        "per_replay": [{
            "replay": r["replay"],
            "date": r["replay"].split("_")[1][:10],
            "steps_with_actions": r["steps_with_actions"],
            "multi_action_steps": r["multi_action_steps"],
            "multi_valid_steps": r["multi_valid_steps"],
            "valid_actions": r["valid_actions"],
            "dropped_valid": r["dropped_valid"],
        } for r in rows],
    }
    # new expert: 3x3 matrix, all actions were valid (getFuncCall returned all)
    mt = j("matrix_translate.json")
    ml = j("multiaction_loss.json")
    n_actions = sum(r["n_actions"] for r in mt)
    n_ok = sum(1 for r in mt for x in r["translated"] if x.get("ok"))
    exp = {
        "source": "offline-wrapped python-sc2 BotAI, 3 bots x 3 states",
        "n_cases": len(mt),
        "n_actions": n_actions,
        "n_translated_ok": n_ok,
        "n_all_valid": n_actions,  # getFuncCall filtered nothing in these cases
        "dropped_by_index0": n_actions - len(mt),
        "drop_rate_of_valid": round(
            100.0 * (n_actions - len(mt)) / max(n_actions, 1), 2),
        "synthetic_step_dropped": [r["dropped"] for r in ml.get("results", [])],
    }
    ratio = (exp["drop_rate_of_valid"] / real["drop_rate_of_valid"]
             if real["drop_rate_of_valid"] else None)
    out = {
        "real_b0": real, "new_expert": exp,
        "drop_rate_ratio_new_vs_real": round(ratio, 1) if ratio else None,
        "conclusion": (
            "getFuncCall already drops no_op(0)/camera(168)/Smart_pt(1) before "
            "returning its list, so function_calls[0] keeps the FIRST VALID "
            "action. On real B0 replays, steps carrying >=2 valid calls are "
            "rare; on the newly wrapped python-sc2 expert, one step routinely "
            "carries 3-13 valid calls, so [0] would drop most of them. "
            "The fix is required for the NEW pipeline, not to re-explain B0."),
        "caveat": "B0 scan samples one replay per recording date (12 dates); "
                  "not all 90 games. New-expert numbers are synthetic/offline, "
                  "not a played concurrent game.",
    }
    with open(os.path.join(P, "action_loss_compare.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("real B0: replays=%d valid=%d multi_valid_steps=%d dropped=%d (%.2f%%)"
          % (real["n_replays"], real["valid_actions"], real["multi_valid_steps"],
             real["dropped_valid"], real["drop_rate_of_valid"]))
    print("         first-action-filtered=%d/%d action steps (%.1f%%)"
          % (real["first_filtered_steps"], real["steps_with_actions"],
             real["pct_action_steps_first_filtered"]))
    print("new expert: cases=%d actions=%d all_valid=%d dropped_by_[0]=%d (%.2f%%)"
          % (exp["n_cases"], exp["n_actions"], exp["n_all_valid"],
             exp["dropped_by_index0"], exp["drop_rate_of_valid"]))
    print("ratio new/real =", out["drop_rate_ratio_new_vs_real"])
    print("per-date dropped:", [(x["date"], x["dropped_valid"])
                                for x in real["per_replay"]])


if __name__ == "__main__":
    main()
