#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only scan: how many actions per step do real B0 (90 human/DAgger) replays carry?

This only observes replays, never writes .pt / never modifies any dataset.
It replicates the project's replay-watching setup (run_configs + InterfaceOptions),
then calls the project's own getFuncCall-equivalent filtering per step, so the
"would be dropped by function_calls[0]" count is measured, not guessed.
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
from pysc2.lib.actions import RAW_FUNCTIONS
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as com_pb

REPLAY_DIR = os.path.expanduser("~/mini-AlphaStar-off/data/Replays/all90")
MAP_FILE = os.path.join(os.environ["SC2PATH"], "Maps/Melee/Simple64.SC2Map")
OUT = os.path.expanduser("~/bvb/expert_probe")
N_REPLAYS = int(os.environ.get("SCAN_N", "4"))
MAX_STEPS = int(os.environ.get("SCAN_MAX_STEPS", "12000"))
OUT_NAME = os.environ.get("SCAN_OUT", "replay_action_scan.json")
SCAN_LIST = [x for x in os.environ.get("SCAN_LIST", "").split(",") if x]

# Exact filter used by transform_replay_data.getFuncCall (lines 384-394):
# value 0=no_op, 168=raw_move_camera, 1=Smart_pt. Smart_unit(12) is NOT filtered there.
FILTERED_IDS = {0, 168, 1}


def interface_options():
    camera_width = 24
    itf = sc_pb.InterfaceOptions(
        raw=True, score=True,
        feature_layer=sc_pb.SpatialCameraSetup(width=camera_width),
        render=None, show_cloaked=False, show_burrowed_shadows=False,
        show_placeholders=False,
        raw_affects_selection=False, raw_crop_to_playable_area=False,
    )
    point.Point(64, 64).assign_to(itf.feature_layer.resolution)
    point.Point(64, 64).assign_to(itf.feature_layer.minimap_resolution)
    itf.feature_layer.crop_to_playable_area = False
    return itf


def pick_replays(n):
    fs = sorted(f for f in os.listdir(REPLAY_DIR) if f.endswith(".SC2Replay"))
    mode = os.environ.get("SCAN_MODE", "date")
    if mode == "list":
        missing = [x for x in SCAN_LIST if x not in fs]
        if missing:
            raise ValueError("SCAN_LIST files not found: %s" % missing)
        return SCAN_LIST[:n] if n < len(SCAN_LIST) else SCAN_LIST
    if mode == "date":
        # one replay per recording date: covers every DAgger iteration batch
        groups = collections.OrderedDict()
        for f in fs:
            key = f.split("_")[1][:10] if "_" in f else f[:10]
            groups.setdefault(key, []).append(f)
        picked = [g[len(g) // 2] for g in groups.values()]
        return picked[:n] if n < len(picked) else picked
    if n >= len(fs):
        return fs
    idx = [int(round(i * (len(fs) - 1) / (n - 1))) for i in range(n)]
    return [fs[i] for i in idx]


def observer_id(controller, replay_data):
    info = controller.replay_info(replay_data)
    ids, results = [], []
    for p in info.player_info:
        if p.player_info.race_actual == com_pb.Protoss:
            ids.append(p.player_info.player_id)
            results.append(p.player_result.result)
    for rid, res in zip(ids, results):
        if res == sc_pb.Victory:
            return rid, len(ids), str(results)
    return (ids[0] if ids else 0), len(ids), str(results)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(MAP_FILE, "rb") as f:
        map_bytes = f.read()
    replays = pick_replays(N_REPLAYS)
    print("[scan] replays=%s map_bytes=%d" % (replays, len(map_bytes)), flush=True)
    rows = []
    run_config = run_configs.get(version="3.16.1")
    with run_config.start(full_screen=False) as controller:
        for name in replays:
            path = os.path.join(REPLAY_DIR, name)
            stat = {
                "replay": name, "steps": 0, "steps_with_actions": 0,
                "total_actions": 0, "multi_action_steps": 0,
                "valid_actions": 0, "kept_by_index0": 0,
                "dropped_valid": 0, "multi_valid_steps": 0,
                "first_filtered_steps": 0,
                "n_dist": collections.Counter(), "valid_dist": collections.Counter(),
                "func_names": collections.Counter(), "error": None,
            }
            try:
                rd = run_config.replay_data(path)
                oid, n_proto, results = observer_id(controller, rd)
                stat["observer_id"] = oid
                stat["n_protoss"] = n_proto
                stat["results"] = results
                sr = sc_pb.RequestStartReplay(
                    replay_data=rd, options=interface_options(),
                    disable_fog=False, observed_player_id=oid,
                    map_data=map_bytes, realtime=False)
                controller.start_replay(sr)
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
                    stat["steps"] += 1
                    n = len(o.actions)
                    if n:
                        stat["steps_with_actions"] += 1
                        stat["total_actions"] += n
                        stat["n_dist"][str(n)] += 1
                        if n > 1:
                            stat["multi_action_steps"] += 1
                        obs = feat.transform_obs(o)
                        valid = []
                        first_filtered = False
                        for k, action in enumerate(o.actions):
                            try:
                                fc = feat.reverse_raw_action(action, obs)
                                v = fc.function.value
                                if k == 0 and v in FILTERED_IDS:
                                    first_filtered = True
                                if v not in FILTERED_IDS:
                                    valid.append(fc.function.name)
                                    stat["func_names"][fc.function.name] += 1
                            except Exception:
                                pass
                        stat["valid_dist"][str(len(valid))] += 1
                        if first_filtered:
                            stat["first_filtered_steps"] += 1
                        if len(valid) > 1:
                            stat["multi_valid_steps"] += 1
                        stat["valid_actions"] += len(valid)
                        if valid:
                            stat["kept_by_index0"] += 1
                            stat["dropped_valid"] += len(valid) - 1
                    if o.player_result:
                        break
                    controller.step()
                    i += 1
            except Exception as exc:
                stat["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                traceback.print_exc()
            stat["n_dist"] = dict(stat["n_dist"])
            stat["valid_dist"] = dict(stat["valid_dist"])
            stat["func_names"] = dict(stat["func_names"].most_common(20))
            rows.append(stat)
            print("[scan] %s steps=%d act_steps=%d actions=%d multi=%d first_filt=%d valid=%d multi_valid=%d dropped=%d err=%s"
                  % (name, stat["steps"], stat["steps_with_actions"],
                     stat["total_actions"], stat["multi_action_steps"],
                     stat["first_filtered_steps"], stat["valid_actions"],
                     stat["multi_valid_steps"], stat["dropped_valid"],
                     stat["error"]), flush=True)
    total = {
        "n_replays": len(rows),
        "steps": sum(r["steps"] for r in rows),
        "steps_with_actions": sum(r["steps_with_actions"] for r in rows),
        "total_actions": sum(r["total_actions"] for r in rows),
        "multi_action_steps": sum(r["multi_action_steps"] for r in rows),
        "valid_actions": sum(r["valid_actions"] for r in rows),
        "kept_by_index0": sum(r["kept_by_index0"] for r in rows),
        "dropped_valid": sum(r["dropped_valid"] for r in rows),
        "multi_valid_steps": sum(r["multi_valid_steps"] for r in rows),
        "first_filtered_steps": sum(r["first_filtered_steps"] for r in rows),
        "replays_with_multi_action": sum(1 for r in rows if r["multi_action_steps"]),
        "errors": [r["error"] for r in rows if r["error"]],
    }
    with open(os.path.join(OUT, OUT_NAME), "w") as f:
        json.dump({"rows": rows, "total": total}, f, indent=2)
    print("[scan] TOTAL %s" % json.dumps(total), flush=True)


if __name__ == "__main__":
    # pysc2's run_configs reads --sc2_run_config via absl flags; parse once first.
    flags.FLAGS([sys.argv[0]])
    main()
