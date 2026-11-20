# -*- coding: utf-8 -*-
"""导出 pysc2 的神族单位/建筑英文名与 ID（权威的英文名来源）。"""
import json
import sys

from pysc2.lib import units

WANT = {
    "Probe", "Zealot", "Stalker", "Adept", "Sentry", "HighTemplar", "DarkTemplar",
    "Immortal", "Colossus", "Disruptor", "Phoenix", "Oracle", "VoidRay", "Carrier",
    "Tempest", "Observer", "WarpPrism", "Mothership", "Archon", "DarkArchon",
    "Nexus", "Pylon", "Gateway", "WarpGate", "Forge", "CyberneticsCore",
    "TwilightCouncil", "RoboticsFacility", "RoboticsBay", "Stargate", "FleetBeacon",
    "TemplarArchives", "DarkShrine", "Assimilator", "PhotonCannon", "ShieldBattery",
}

rows = {}
for u in list(units.Protoss) + list(units.Neutral):
    if u.name in WANT:
        rows[u.name] = int(u.value)

print("n=%d" % len(rows))
for k in sorted(rows, key=lambda x: rows[x]):
    print("%-20s %d" % (k, rows[k]))
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/protoss_ids.json", "w"),
          ensure_ascii=False, indent=1)
