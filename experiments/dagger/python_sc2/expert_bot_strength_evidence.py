#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only: quantify the strength evidence shipped with the sc2_bots repo.

Parses phoenix/results/history.jsonl (real recorded games) and reports win
rates by difficulty / opponent race. No SC2 client, no training, no writes
outside ~/bvb/expert_probe/bot_strength_evidence.json.

NOTE: these games were played by the upstream authors on AI Arena maps with
ares-sc2 + a newer SC2 client. They are NOT comparable to our difficulty-7
Simple64 results; they only establish that a state-conditioned scripted
Protoss bot can beat cheat-level built-in AI.
"""
import collections
import json
import os

SRC = os.path.expanduser(
    "~/bvb/expert_probe/sc2_bots_src/phoenix/results/history.jsonl")
OUT = os.path.expanduser("~/bvb/expert_probe/bot_strength_evidence.json")


def rate(items):
    c = collections.Counter(r.get("result") for r in items)
    n = c["Victory"] + c["Defeat"] + c["Tie"]
    return {
        "victory": c["Victory"], "defeat": c["Defeat"], "tie": c["Tie"],
        "error": c["Error"], "decided": n,
        "winrate": round(100.0 * c["Victory"] / n, 1) if n else None,
    }


def main():
    rows = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_diff = {}
    for d in sorted({str(r.get("difficulty")) for r in rows}):
        by_diff[d] = rate([r for r in rows if str(r.get("difficulty")) == d])
    by_race_ci = {}
    sub = [r for r in rows if r.get("difficulty") == "CheatInsane"]
    for race in sorted({str(r.get("opponent_race")) for r in sub}):
        by_race_ci[race] = rate(
            [r for r in sub if str(r.get("opponent_race")) == race])
    groups = collections.OrderedDict()
    for r in sub:
        groups.setdefault(str(r.get("run_id")), []).append(r)
    by_run = {k: rate(v) for k, v in groups.items()}
    out = {
        "source": SRC,
        "n_records": len(rows),
        "overall": rate(rows),
        "by_difficulty": by_diff,
        "cheatinsane_by_race": by_race_ci,
        "cheatinsane_runs": len(by_run),
        "cheatinsane_last_runs": dict(list(by_run.items())[-6:]),
        "caveat": "upstream authors, AI Arena maps, ares-sc2 + newer SC2; "
                  "NOT comparable with our difficulty-7 Simple64 results",
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("records=%d overall=%s" % (len(rows), out["overall"]))
    for k, v in sorted(by_diff.items()):
        print("  difficulty=%-12s %s" % (k, v))
    for k, v in sorted(by_race_ci.items()):
        print("  CheatInsane race=%-9s %s" % (k, v))
    print("caveat:", out["caveat"])


if __name__ == "__main__":
    main()
