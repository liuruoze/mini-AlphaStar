"""Extract declarative action tokens from PiGBot builds for portability estimate."""
import json
import re
from collections import Counter

import yaml

P = "/home/liuruoze/bvb/expert_probe"
d = yaml.safe_load(open(P + "/protoss_builds.yml"))
B = d["Builds"]

KNOWN = {
    "PYLON": "Build_Pylon_pt", "GATEWAY": "Build_Gateway_pt",
    "ASSIMILATOR": "Build_Assimilator_unit", "CYBERNETICSCORE": "Build_CyberneticsCore_pt",
    "FORGE": "Build_Forge_pt", "STARGATE": "Build_Stargate_pt",
    "NEXUS": "Build_Nexus_pt", "ROBOTICSFACILITY": "Build_RoboticsFacility_pt",
    "PROBE": "Train_Probe_quick", "ZEALOT": "Train_Zealot_quick",
    "STALKER": "Train_Stalker_quick", "SENTRY": "Train_Sentry_quick",
    "ADEPT": "Train_Adept_quick", "VOIDRAY": "Train_VoidRay_quick",
    "CHRONO": "Effect_ChronoBoostEnergyCost", "WORKER_SCOUT": "(scout, non-build)",
    "WARPGATE": "Morph_WarpGate", "SHIELD_BATTERY": "Build_ShieldBattery_pt",
}

KNOWN_EXT = dict(KNOWN)
KNOWN_EXT.update({
    "SHIELDBATTERY": "Build_ShieldBattery_pt",
    "WARPGATERESEARCH": "Research_WarpGate",
    "IMMORTAL": "Train_Immortal_quick",
    "TWILIGHTCOUNCIL": "Build_TwilightCouncil_pt",
    "OBSERVER": "Train_Observer_quick",
    "ROBOTICSBAY": "Build_RoboticsBay_pt",
    "BLINKTECH": "Research_Blink",
    "WARPPRISM": "Train_WarpPrism_quick",
    "EXPAND": "Build_Nexus_pt (+placement)",
    "GAS": "Build_Assimilator_unit / Harvest_Gather (context-dependent)",
})

tokens = Counter()
per_build = {}
for k, v in B.items():
    steps = v.get("OpeningBuildOrder", [])
    local = Counter()
    for step in steps:
        s = str(step)
        for w in re.findall(r"\b[A-Z][A-Z_]{2,}\b", s):
            tokens[w] += 1
            local[w] += 1
    per_build[k] = dict(local)

print("=== top declarative tokens across 12 builds ===")
for w, n in tokens.most_common(28):
    print("  %-24s %-4d  mapped=%s" % (w, n, KNOWN.get(w, "NO")))
mapped = sum(n for w, n in tokens.items() if w in KNOWN)
total = sum(tokens.values())
mapped_ext = sum(n for w, n in tokens.items() if w in KNOWN_EXT)
print("token occurrences=%d conservative-mapped=%d (%.1f%%) extended-mapped=%d (%.1f%%)"
      % (total, mapped, 100.0 * mapped / max(total, 1),
         mapped_ext, 100.0 * mapped_ext / max(total, 1)))
print("distinct tokens=%d, conservative distinct=%d, extended distinct=%d"
      % (len(tokens), sum(1 for w in tokens if w in KNOWN),
         sum(1 for w in tokens if w in KNOWN_EXT)))
with open(P + "/pigbot_mapping.json", "w") as f:
    json.dump({"tokens": dict(tokens), "per_build": per_build,
               "known": KNOWN, "known_ext": KNOWN_EXT,
               "total": total, "mapped": mapped, "mapped_ext": mapped_ext}, f, indent=2)
