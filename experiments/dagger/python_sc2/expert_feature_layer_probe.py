#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only: what are the feature_layer actions in a real B0 concurrent replay?

Project getFuncCall() calls Features.reverse_raw_action(), which only decodes
action_raw; everything else becomes no_op and is filtered. This probe classifies
those dropped non-raw actions to decide whether expert (human) build/train
commands are being lost. No .pt writes, no dataset changes.
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
from s2clientprotocol import common_pb2 as com_pb

REPLAY_DIR = os.path.expanduser("~/mini-AlphaStar-off/data/Replays/all90")
MAP_FILE = os.path.join(os.environ["SC2PATH"], "Maps/Melee/Simple64.SC2Map")
OUT = os.path.expanduser("~/bvb/expert_probe")
MAX_STEPS = int(os.environ.get("FL_MAX_STEPS", "12000"))
TARGET = os.environ.get(
    "FL_REPLAY", "Simple64_2021-12-14-09-33-00.SC2Replay")


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
    map_bytes = open(MAP_FILE, "rb").read()
    rc = run_configs.get(version="3.16.1")
    fl_fields = collections.Counter()
    ui_fields = collections.Counter()
    fl_ability = collections.Counter()
    n_fl = n_ui = n_raw = 0
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
        i = 0
        while i < MAX_STEPS:
            o = controller.observe()
            for a in o.actions:
                if a.HasField("action_raw"):
                    n_raw += 1
                elif a.HasField("action_feature_layer"):
                    n_fl += 1
                    fl = a.action_feature_layer
                    for f, v in fl.ListFields():
                        fl_fields[f.name] += 1
                        # ability_id may live on unit_command (or nested target)
                        aid = getattr(v, "ability_id", None)
                        if aid:
                            fl_ability["%s:%s" % (f.name, aid)] += 1
                elif a.HasField("action_ui"):
                    n_ui += 1
                    for f, _ in a.action_ui.ListFields():
                        ui_fields[f.name] += 1
            if o.player_result:
                break
            controller.step()
            i += 1
    out = {
        "replay": TARGET, "steps": i + 1,
        "n_raw": n_raw, "n_feature_layer": n_fl, "n_ui": n_ui,
        "feature_layer_fields": dict(fl_fields.most_common()),
        "ui_fields": dict(ui_fields.most_common()),
        "feature_layer_ability_ids": dict(fl_ability.most_common(25)),
        "note": "getFuncCall decodes only action_raw; all feature_layer / ui "
                "actions become no_op and are filtered out.",
    }
    with open(os.path.join(OUT, "feature_layer_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    flags.FLAGS([sys.argv[0]])
    main()
