#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cache the per-step action list of ONE replay, decoded by the project's own
getFuncCall, so later steps (P0-1b three-way policy) never need another SC2 run.

Output: <out>.pkl  = list of dicts
  {i, game_loop, n_actions, n_valid, valid: [{name, value, hex}], first_name}
Read-only: no .pt, no dataset change.
"""
import json
import os
import pickle
import sys

M = os.path.expanduser("~/mini-AlphaStar")
sys.path.insert(0, M)
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

from absl import flags  # noqa: E402
from pysc2 import run_configs  # noqa: E402
from pysc2.lib import features as F  # noqa: E402
from pysc2.lib import point  # noqa: E402
from s2clientprotocol import sc2api_pb2 as sc_pb  # noqa: E402
from s2clientprotocol import common_pb2 as com_pb  # noqa: E402

from alphastarmini.core.sl.transform_replay_data import getFuncCall  # noqa: E402

REPLAY_DIR = os.path.expanduser("~/mini-AlphaStar-off/data/Replays/all90")
MAP_FILE = os.path.join(os.environ["SC2PATH"], "Maps/Melee/Simple64.SC2Map")
OUT_PKL = os.environ.get(
    "CACHE_OUT", os.path.expanduser("~/bvb/p0_baseline/actions_seq_12-14-09-33-00.pkl"))
TARGET = os.environ.get("CACHE_REPLAY",
                        "Simple64_2021-12-14-09-33-00.SC2Replay")
MAX_STEPS = int(os.environ.get("CACHE_MAX_STEPS", "12000"))


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
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    map_bytes = open(MAP_FILE, "rb").read()
    rc = run_configs.get(version="3.16.1")
    rows = []
    with rc.start(full_screen=False) as controller:
        rd = rc.replay_data(os.path.join(REPLAY_DIR, TARGET))
        info = controller.replay_info(rd)
        oid, ids = 0, []
        for p in info.player_info:
            if p.player_info.race_actual == com_pb.Protoss:
                ids.append(p.player_info.player_id)
                if p.player_result.result == sc_pb.Victory:
                    oid = p.player_info.player_id
        if not oid and ids:
            oid = ids[0]
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=rd, options=interface_options(), disable_fog=False,
            observed_player_id=oid, map_data=map_bytes, realtime=False))
        feat = F.features_from_game_info(
            game_info=controller.game_info(), raw_resolution=64,
            crop_to_playable_area=False, raw_crop_to_playable_area=False,
            hide_specific_actions=[], use_feature_units=True,
            use_raw_units=True, use_unit_counts=True, use_raw_actions=True,
            show_cloaked=True, show_burrowed_shadows=True,
            show_placeholders=True)
        i = 0
        while i < MAX_STEPS:
            o = controller.observe()
            if o.actions:
                obs = feat.transform_obs(o)
                fcs = getFuncCall(o, feat, obs)
                valid = []
                if fcs:
                    for fc in fcs:
                        # pysc2 FunctionCall is a namedtuple, not a protobuf:
                        # keep name/value plus a compact repr of arguments so
                        # same-name/same-target repeats can be detected later.
                        valid.append({
                            "name": fc.function.name,
                            "value": int(fc.function.value),
                            "args": str(fc.arguments)[:240],
                        })
                rows.append({
                    "i": i, "game_loop": o.observation.game_loop,
                    "n_actions": len(o.actions), "n_valid": len(valid),
                    "valid": valid,
                    "first_name": valid[0]["name"] if valid else None,
                })
            if o.player_result:
                break
            controller.step()
            i += 1
    with open(OUT_PKL, "wb") as f:
        pickle.dump({"replay": TARGET, "rows": rows}, f, protocol=4)
    n_act = sum(r["n_actions"] for r in rows)
    n_val = sum(r["n_valid"] for r in rows)
    multi = sum(1 for r in rows if r["n_valid"] > 1)
    summary = {"replay": TARGET, "steps": i + 1, "action_steps": len(rows),
               "total_actions": n_act, "valid_actions": n_val,
               "multi_valid_steps": multi, "out": OUT_PKL}
    with open(OUT_PKL.replace(".pkl", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[cache] %s" % json.dumps(summary), flush=True)


if __name__ == "__main__":
    flags.FLAGS([sys.argv[0]])
    main()
