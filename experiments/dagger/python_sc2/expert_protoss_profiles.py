#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only: index the upstream repo's Protoss ladder-bot scouting profiles.

These are *descriptions* of strong Protoss bots (Elo, opening, composition,
record), not executable bots. They are a tactical-diversity source for choosing
which build a machine expert should emulate -- directly addressing the
"HierNet only does Zealot rush" gap.
"""
import json
import os
import re

ROOT = os.path.expanduser(
    "~/bvb/expert_probe/sc2_bots_src/bot_profiles")
OUT = os.path.expanduser("~/bvb/expert_probe/protoss_profiles.json")


def field(text, key):
    m = re.search(r"\|\s*\*\*%s\*\*\s*\|\s*([^|]+?)\s*\|" % re.escape(key), text)
    return m.group(1).strip() if m else None


def main():
    rows = []
    for name in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, name, "PROFILE.md")
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        race = field(text, "Race")
        if race != "Protoss":
            continue
        opening = re.search(r"\*\*Opening:\*\*\s*(.+)", text)
        summary = re.search(r"## Summary\s*\n\s*\n(.+)", text)
        elo_raw = field(text, "AI Arena Elo") or ""
        m = re.search(r"(\d{3,4})", elo_raw)
        rows.append({
            "name": name,
            "race": race,
            "bot_type": field(text, "Bot type"),
            "elo": int(m.group(1)) if m else None,
            "elo_raw": elo_raw,
            "source_public": field(text, "Source public"),
            "opening": opening.group(1).strip()[:220] if opening else None,
            "summary": summary.group(1).strip()[:220] if summary else None,
        })
    buckets = {}
    for r in rows:
        o = (r["opening"] or "").lower()
        if "forge" in o or "cannon" in o:
            k = "forge/cannon"
        elif "stargate" in o or "tempest" in o or "carrier" in o or "void ray" in o:
            k = "stargate/air"
        elif "robo" in o or "immortal" in o:
            k = "robo"
        elif "gateway" in o or "stalker" in o or "adept" in o or "zealot" in o:
            k = "gateway"
        else:
            k = "unspecified"
        buckets[k] = buckets.get(k, 0) + 1
    out = {
        "source": ROOT,
        "n_profiles_total": len([n for n in os.listdir(ROOT)
                                 if os.path.isfile(os.path.join(ROOT, n, "PROFILE.md"))]),
        "n_protoss": len(rows),
        "elo_min": min([r["elo"] for r in rows if r["elo"]], default=None),
        "elo_max": max([r["elo"] for r in rows if r["elo"]], default=None),
        "opening_buckets": buckets,
        "profiles": rows,
        "caveat": "scouting descriptions, NOT executable bots; upstream AI Arena "
                  "ladder data, not our Simple64 / SC2 3.16.1 setting",
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("n_protoss=%d elo=%s..%s buckets=%s"
          % (out["n_protoss"], out["elo_min"], out["elo_max"], buckets))
    for r in rows[:8]:
        print("  %-22s Elo=%-5s %s" % (r["name"], r["elo"],
                                       (r["opening"] or "")[:70]))
    print("caveat:", out["caveat"])


if __name__ == "__main__":
    main()
