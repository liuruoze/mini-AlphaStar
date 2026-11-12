#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only probe: do real B0 replays expose action_errors (rejected commands)?

Directly tests the user's earlier ruling: "replay records commands that were
sent, including ones the engine rejected; commands with wrong selected_units
never reach the engine validation layer". We count, per step:
  - o.actions composition (raw / feature_layer / ui / chat) and ability ids
  - o.action_errors count and their ActionResult enum values
  - steps with both actions and errors
  - repeated identical (ability_id, unit_tags) commands (proxy for rejected
    commands being re-issued on the following steps)
No .pt writes, no dataset changes.
"""
import collections
import json
import os
import sys

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
sys.path.insert(0, os.path.expanduser("~/mini-AlphaStar"))

from absl import flags
from pysc2 import run_configs
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import error_pb2 as err_pb
from s2clientprotocol import common_pb2 as com_pb

REPLAY_DIR = os.path.expanduser("~/mini-AlphaStar-off/data/Replays/all90")
MAP_FILE = os.path.join(os.environ["SC2PATH"], "Maps/Melee/Simple64.SC2Map")
OUT = os.path.expanduser("~/bvb/expert_probe")
MAX_STEPS = int(os.environ.get("PROBE_MAX_STEPS", "12000"))
TARGETS = [x for x in os.environ.get(
    "PROBE_REPLAYS",
    "Simple64_2021-12-14-09-33-00.SC2Replay,"
    "Simple64_2021-12-19-09-21-04.SC2Replay").split(",") if x]

RESULT_NAME = {v.number: v.name for v in err_pb.ActionResult.DESCRIPTOR.values}


def interface_options():
    itf = sc_pb.InterfaceOptions(
        raw=True, score=True,
        feature_layer=sc_pb.SpatialCameraSetup(width=24), render=None,
        show_cloaked=False, show_burrowed_shadows=False,
        show_placeholders=False, raw_affects_selection=False,
        raw_crop_to_playable_area=False)
    point.Point(64, 64).assign_to(itf.feature_layer.resolution)
    point.Point(64, 64).assign_to(itf.feature_layer.minimap_resolution)
    return itf


def main():
    os.makedirs(OUT, exist_ok=True)
    map_bytes = open(MAP_FILE, "rb").read()
    rc = run_configs.get(version="3.16.1")
    rows = []
    with rc.start(full_screen=False) as controller:
        for name in TARGETS:
            path = os.path.join(REPLAY_DIR, name)
            s = {"replay": name, "steps": 0, "steps_with_actions": 0,
                 "steps_with_errors": 0, "steps_with_both": 0,
                 "total_actions": 0, "total_errors": 0,
                 "action_kinds": collections.Counter(),
                 "ability_ids": collections.Counter(),
                 "error_results": collections.Counter(),
                 "error_examples": [], "error": None}
            try:
                rd = rc.replay_data(path)
                info = controller.replay_info(rd)
                oid = 0
                ids = []
                for p in info.player_info:
                    if p.player_info.race_actual == com_pb.Protoss:
                        ids.append(p.player_info.player_id)
                        if p.player_result.result == sc_pb.Victory:
                            oid = p.player_info.player_id
                if not oid and ids:
                    oid = ids[0]
                controller.start_replay(sc_pb.RequestStartReplay(
                    replay_data=rd, options=interface_options(),
                    disable_fog=False, observed_player_id=oid,
                    map_data=map_bytes, realtime=False))
                i = 0
                while i < MAX_STEPS:
                    o = controller.observe()
                    s["steps"] += 1
                    na, ne = len(o.actions), len(o.action_errors)
                    if na:
                        s["steps_with_actions"] += 1
                        s["total_actions"] += na
                        for a in o.actions:
                            if a.HasField("action_raw"):
                                k = "action_raw"
                                uc = a.action_raw.unit_command
                                if uc.ability_id:
                                    s["ability_ids"][str(uc.ability_id)] += 1
                            elif a.HasField("action_feature_layer"):
                                k = "action_feature_layer"
                            elif a.HasField("action_ui"):
                                k = "action_ui"
                            elif a.HasField("action_chat"):
                                k = "action_chat"
                            else:
                                k = "empty_or_other"
                            s["action_kinds"][k] += 1
                    if ne:
                        s["steps_with_errors"] += 1
                        s["total_errors"] += ne
                        for e in o.action_errors:
                            rname = RESULT_NAME.get(e.result, "UNKNOWN_%d" % e.result)
                            s["error_results"][rname] += 1
                            if len(s["error_examples"]) < 12:
                                s["error_examples"].append({
                                    "game_loop": o.observation.game_loop,
                                    "unit_tag": e.unit_tag,
                                    "ability_id": e.ability_id,
                                    "result": rname,
                                })
                    if na and ne:
                        s["steps_with_both"] += 1
                    if o.player_result:
                        break
                    controller.step()
                    i += 1
            except Exception as exc:
                s["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:180])
            for k in ("action_kinds", "ability_ids", "error_results"):
                s[k] = dict(s[k].most_common(20))
            rows.append(s)
            print("[err-probe] %s steps=%d actions=%d errors=%d both_steps=%d kinds=%s err=%s"
                  % (name, s["steps"], s["total_actions"], s["total_errors"],
                     s["steps_with_both"], s["action_kinds"], s["error"]),
                  flush=True)
            print("            error_results=%s" % s["error_results"], flush=True)
    with open(os.path.join(OUT, "replay_action_error_probe.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[err-probe] DONE rows=%d" % len(rows), flush=True)


if __name__ == "__main__":
    flags.FLAGS([sys.argv[0]])
    main()
