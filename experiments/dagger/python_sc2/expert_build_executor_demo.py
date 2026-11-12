#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline proof: strategy_engine.BuildExecutor emits concrete, state-conditioned
Protoss build/train/research actions with no SC2 client and no pysc2 dependency.

Runs in venv_sc2_084 (python-sc2 0.8.4, Py3.8) with PYTHONPATH pointing at the
sc2_bots source tree. Writes only build_executor_demo.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/bvb/expert_probe/sc2_bots_src"))
sys.path.insert(0, os.path.expanduser("~/mini-AlphaStar/experiments"))

from strategy_engine import build_guides as bg

OUT = os.path.expanduser("~/bvb/expert_probe")
ABIL = json.load(open(os.path.expanduser(
    "~/mini-AlphaStar/experiments/sc2_unit_ability.json")))["units"]
TOKEN_UID = json.load(open(os.path.join(OUT, "protoss_token_unitid.json")))


def ability_for(token):
    uid = TOKEN_UID.get(token)
    if uid is None:
        return None
    return ABIL.get(str(uid))


def walk(build, max_steps=40):
    ex = bg.BuildExecutor(build, reproducible_only=True)
    have = {}
    seq = []
    for step in range(max_steps):
        act = ex.next_action(have)
        if act is None:
            break
        token = act.token or act.name
        # whether the script's supply/time trigger has been reached
        due_supply = bg.BuildExecutor.is_due(act, act.at_supply, None)
        due_time = bg.BuildExecutor.is_due(act, None, act.at_second)
        seq.append({
            "i": step,
            "action": act.action, "name": act.name, "token": token,
            "unit_id": TOKEN_UID.get(token),
            "ability_id": ability_for(token),
            "at_supply": act.at_supply, "at_second": act.at_second,
            "due_supply": due_supply, "due_time": due_time,
            "have_before": dict(have),
        })
        have[token] = have.get(token, 0) + act.count
    return {
        "build_id": build.id, "title": build.title,
        "matchup": build.matchup,
        "coverage": build.coverage(),
        "n_steps": len(build.build_steps()),
        "n_actions_walked": len(seq),
        "n_with_ability": sum(1 for s in seq if s["ability_id"] is not None),
        "distinct_actions": sorted({s["action"] for s in seq}),
        "distinct_tokens": sorted({s["token"] for s in seq}),
        "sequence": seq,
    }


def main():
    protoss = [b for b in bg.BUILD_GUIDES.values() if b.race == "Protoss"]
    results = [walk(b) for b in protoss]
    summary = {
        "n_protoss_guides": len(protoss),
        "guides": [{
            "id": r["build_id"], "title": r["title"],
            "steps": r["n_steps"], "coverage": r["coverage"]["fraction"],
            "walked": r["n_actions_walked"],
            "with_ability": r["n_with_ability"],
            "kinds": r["distinct_actions"],
            "tokens": r["distinct_tokens"],
        } for r in results],
        "total_walked": sum(r["n_actions_walked"] for r in results),
        "total_with_ability": sum(r["n_with_ability"] for r in results),
    }
    with open(os.path.join(OUT, "build_executor_demo.json"), "w") as f:
        json.dump({"summary": summary, "walks": results}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("=== first guide, first 8 state-conditioned steps ===")
    for s in results[0]["sequence"][:8]:
        print("  i=%d %-8s %-26s token=%-20s aid=%-6s supply=%s have=%s"
              % (s["i"], s["action"], s["name"], s["token"],
                 s["ability_id"], s["at_supply"], s["have_before"]))


if __name__ == "__main__":
    main()
