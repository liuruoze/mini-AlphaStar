#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only: does a human interface selection actually change is_selected?

Follows up ERRATA J147: if the expert's intervention is carried by the
selection set, then steps containing a feature_layer selection action should
show a different number of is_selected raw units on the NEXT observation.
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
from s2clientprotocol import common_pb2 as com_pb
from s2clientprotocol import spatial_pb2 as sp_pb

REPLAY_DIR = os.path.expanduser("~/mini-AlphaStar-off/data/Replays/all90")
MAP_FILE = os.path.join(os.environ["SC2PATH"], "Maps/Melee/Simple64.SC2Map")
OUT = os.path.expanduser("~/bvb/expert_probe")
MAX_STEPS = int(os.environ.get("SEL_MAX_STEPS", "12000"))
TARGET = os.environ.get(
    "SEL_REPLAY", "Simple64_2021-12-14-09-33-00.SC2Replay")

TYPE_NAME = {v.number: v.name
             for v in sp_pb.ActionSpatialUnitSelectionPoint.Type.DESCRIPTOR.values} \
    if hasattr(sp_pb.ActionSpatialUnitSelectionPoint, "Type") else {}


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


def n_selected(o):
    return sum(1 for u in o.observation.raw_data.units
               if getattr(u, "is_selected", False))


def main():
    map_bytes = open(MAP_FILE, "rb").read()
    rc = run_configs.get(version="3.16.1")
    sel_types = collections.Counter()
    rect_add = collections.Counter()
    sel_counts = collections.Counter()
    sel_counts_when_sel = collections.Counter()
    sel_counts_no_sel = collections.Counter()
    steps_with_selection = 0
    changed_next = 0
    prev_had_selection = False
    prev_n = None
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
            n = n_selected(o)
            sel_counts[str(n)] += 1
            if prev_had_selection and prev_n is not None and n != prev_n:
                changed_next += 1
            has_sel = False
            for a in o.actions:
                if not a.HasField("action_feature_layer"):
                    continue
                fl = a.action_feature_layer
                if fl.HasField("unit_selection_point"):
                    has_sel = True
                    t = fl.unit_selection_point.type
                    sel_types[TYPE_NAME.get(t, str(t))] += 1
                elif fl.HasField("unit_selection_rect"):
                    has_sel = True
                    rect_add[str(fl.unit_selection_rect.selection_add)] += 1
            if has_sel:
                steps_with_selection += 1
                sel_counts_when_sel[str(n)] += 1
            else:
                sel_counts_no_sel[str(n)] += 1
            prev_had_selection, prev_n = has_sel, n
            if o.player_result:
                break
            controller.step()
            i += 1
    out = {
        "replay": TARGET, "steps": i + 1,
        "steps_with_selection_action": steps_with_selection,
        "selection_point_types": dict(sel_types),
        "selection_rect_add": dict(rect_add),
        "n_selected_distribution": dict(sel_counts.most_common(15)),
        "n_selected_WHEN_selection_action_present": dict(
            sel_counts_when_sel.most_common(15)),
        "n_selected_WHEN_no_selection_action": dict(
            sel_counts_no_sel.most_common(15)),
        "mean_n_selected_with_sel": round(
            sum(int(k) * v for k, v in sel_counts_when_sel.items())
            / max(sum(sel_counts_when_sel.values()), 1), 2),
        "mean_n_selected_without_sel": round(
            sum(int(k) * v for k, v in sel_counts_no_sel.items())
            / max(sum(sel_counts_no_sel.values()), 1), 2),
        "steps_after_selection_with_changed_n_selected": changed_next,
        "change_rate": round(100.0 * changed_next / max(steps_with_selection, 1), 1),
        "note": "is_selected read from raw_data.units on each observation; "
                "change is measured on the NEXT observation after a selection action.",
    }
    with open(os.path.join(OUT, "selection_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    flags.FLAGS([sys.argv[0]])
    main()
