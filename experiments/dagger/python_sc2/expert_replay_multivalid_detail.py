#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only detail scan: when one step has >=2 *valid* calls, what gets dropped
by function_calls[0]? Only observes replays; writes no .pt / no dataset changes.

Deliberately a separate file from expert_replay_action_scan.py so the running
12-date scan script is never overwritten.
"""
import collections
import json
import os
import sys
import traceback

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
sys.path.insert(0, os.path.expanduser("~/mini-AlphaStar"))

from absl import flags
from pysc2 import run_configs
from pysc2.lib import features as F
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as com_pb

REPLAY_DIR = os.path.expanduser("~/mini-AlphaStar-off/data/Replays/all90")
MAP_FILE = os.path.join(os.environ["SC2PATH"], "Maps/Melee/Simple64.SC2Map")
OUT = os.path.expanduser("~/bvb/expert_probe")
MAX_STEPS = int(os.environ.get("DETAIL_MAX_STEPS", "12000"))
TARGETS = [x for x in os.environ.get(
    "DETAIL_REPLAYS",
    "Simple64_2021-12-13-08-48-02.SC2Replay").split(",") if x]

FILTERED_IDS = {0, 168, 1}  # exactly transform_replay_data.getFuncCall


def interface_options():
    itf = sc_pb.InterfaceOptions(
        raw=True, score=True,
        feature_layer=sc_pb.SpatialCameraSetup(width=24),
        render=None, show_cloaked=False, show_burrowed_shadows=False,
        show_placeholders=False, raw_affects_selection=False,
        raw_crop_to_playable_area=False)
    point.Point(64, 64).assign_to(itf.feature_layer.resolution)
    point.Point(64, 64).assign_to(itf.feature_layer.minimap_resolution)
    itf.feature_layer.crop_to_playable_area = False
    return itf


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(MAP_FILE, "rb") as f:
        map_bytes = f.read()
    rows = []
    rc = run_configs.get(version="3.16.1")
    with rc.start(full_screen=False) as controller:
        for name in TARGETS:
            path = os.path.join(REPLAY_DIR, name)
            if not os.path.exists(path):
                print("[detail] MISSING", path, flush=True)
                continue
            item = {"replay": name, "steps": 0, "multi_valid": 0,
                    "examples": [], "pair_counter": collections.Counter(),
                    "first_filtered": 0, "error": None}
            try:
                rd = rc.replay_data(path)
                info = controller.replay_info(rd)
                ids = [p.player_info.player_id for p in info.player_info
                       if p.player_info.race_actual == com_pb.Protoss]
                oid = 0
                for p in info.player_info:
                    if (p.player_info.race_actual == com_pb.Protoss
                            and p.player_result.result == sc_pb.Victory):
                        oid = p.player_info.player_id
                        break
                if not oid and ids:
                    oid = ids[0]
                controller.start_replay(sc_pb.RequestStartReplay(
                    replay_data=rd, options=interface_options(),
                    disable_fog=False, observed_player_id=oid,
                    map_data=map_bytes, realtime=False))
                feat = F.features_from_game_info(
                    game_info=controller.game_info(), raw_resolution=64,
                    crop_to_playable_area=False, raw_crop_to_playable_area=False,
                    hide_specific_actions=[], use_feature_units=True,
                    use_raw_units=True, use_unit_counts=True,
                    use_raw_actions=True, show_cloaked=True,
                    show_burrowed_shadows=True, show_placeholders=True)
                i = 0
                while i < MAX_STEPS:
                    o = controller.observe()
                    item["steps"] += 1
                    if o.actions:
                        obs = feat.transform_obs(o)
                        names = []
                        first_filtered = False
                        for k, action in enumerate(o.actions):
                            try:
                                fc = feat.reverse_raw_action(action, obs)
                                v = fc.function.value
                                if k == 0 and v in FILTERED_IDS:
                                    first_filtered = True
                                if v not in FILTERED_IDS:
                                    names.append(fc.function.name)
                            except Exception:
                                pass
                        if first_filtered:
                            item["first_filtered"] += 1
                        if len(names) >= 2:
                            item["multi_valid"] += 1
                            item["pair_counter"][tuple(names)] += 1
                            if len(item["examples"]) < 40:
                                item["examples"].append({
                                    "game_loop": o.observation.game_loop,
                                    "n_actions": len(o.actions),
                                    "valid": names,
                                    "kept": names[0],
                                    "dropped": names[1:],
                                })
                    if o.player_result:
                        break
                    controller.step()
                    i += 1
            except Exception as exc:
                item["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                traceback.print_exc()
            item["pair_counter"] = [
                {"valid": list(k), "n": v}
                for k, v in item["pair_counter"].most_common(20)]
            rows.append(item)
            print("[detail] %s steps=%d multi_valid=%d first_filt=%d err=%s"
                  % (name, item["steps"], item["multi_valid"],
                     item["first_filtered"], item["error"]), flush=True)
    with open(os.path.join(OUT, "replay_multivalid_detail.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[detail] DONE rows=%d" % len(rows), flush=True)


if __name__ == "__main__":
    flags.FLAGS([sys.argv[0]])
    main()
