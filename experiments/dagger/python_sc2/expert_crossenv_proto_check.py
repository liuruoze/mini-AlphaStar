#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-environment proto bridge check.

Run the SAME script under both interpreters on the SAME serialized bytes:
  venv_sc2_084 (python-sc2, protobuf 3.20.3) vs th_1_7 (pysc2, protobuf 3.8.0)
Output is a deterministic text digest; the two outputs must be byte-identical.
"""
import hashlib
import json
import os
import pickle

from s2clientprotocol import sc2api_pb2 as sc_pb

P = os.path.expanduser("~/bvb/expert_probe")


def load(name):
    return pickle.load(open(os.path.join(P, name), "rb"))


def as_observation(b):
    """Auto-detect either ResponseObservation bytes or a top-level Response."""
    ro = sc_pb.ResponseObservation()
    ro.ParseFromString(b)
    top = sc_pb.Response()
    try:
        top.ParseFromString(b)
    except Exception:
        top = None
    best = ro.observation
    if top is not None and top.HasField("observation"):
        cand = top.observation
        if (len(cand.raw_data.units) > len(best.raw_data.units)
                or cand.game_loop > best.game_loop):
            best = cand
    return ro, best


def digest_obs(name):
    b = load(name)
    ro, ob = as_observation(b)
    units = sorted(
        (u.tag, u.unit_type, u.owner, int(u.pos.x * 100), int(u.pos.y * 100))
        for u in ob.raw_data.units
    )
    return {
        "bytes": len(b),
        "game_loop": ob.game_loop,
        "n_units": len(ob.raw_data.units),
        "units_sha": hashlib.sha256(json.dumps(units).encode()).hexdigest()[:16],
        "minerals": ob.player_common.minerals,
        "food_used": ob.player_common.food_used,
        "food_cap": ob.player_common.food_cap,
        "n_actions": len(ro.actions),
        "n_errors": len(ro.action_errors),
        "player_result": [r.result for r in ro.player_result],
    }


def digest_data():
    b = load("response_data.pkl")
    d = sc_pb.ResponseData()
    d.ParseFromString(b)
    ids = sorted(u.unit_id for u in d.units)
    return {
        "n_abilities": len(d.abilities), "n_units": len(d.units),
        "n_upgrades": len(d.upgrades),
        "unit_ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
        "has_ability_881": any(a.ability_id == 881 for a in d.abilities),
        "bytes": len(b),
    }


def digest_info():
    b = load("response_game_info.pkl")
    g = sc_pb.ResponseGameInfo()
    g.ParseFromString(b)
    return {
        "map_size": [g.start_raw.map_size.x, g.start_raw.map_size.y],
        "placement": [g.start_raw.placement_grid.size.x,
                      g.start_raw.placement_grid.size.y],
        "n_players": len(g.player_info),
        "bytes": len(b),
    }


def digest_action(name):
    obj = load(name)
    b = obj[0] if isinstance(obj, (list, tuple)) else obj
    a = sc_pb.Action()
    a.ParseFromString(b)
    uc = a.action_raw.unit_command
    return {
        "ability_id": uc.ability_id,
        "unit_tags": list(uc.unit_tags),
        "target_pt": [uc.target_world_space_pos.x, uc.target_world_space_pos.y]
        if uc.HasField("target_world_space_pos") else None,
        "queued": uc.queue_command,
        "bytes": len(b),
    }


if __name__ == "__main__":
    out = {}
    jobs = [
        ("real_obs_000", lambda: digest_obs("real_obs_000.pkl")),
        ("obs_000", lambda: digest_obs("obs_000.pkl")),
        ("response_data", digest_data),
        ("response_game_info", digest_info),
        ("action_A", lambda: digest_action("actions_A_supply_blocked.pkl")),
        ("action_B", lambda: digest_action("actions_B_supply_ok.pkl")),
    ]
    for name, fn in jobs:
        try:
            out[name] = fn()
        except Exception as exc:
            out[name] = {"ERROR": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
    print(json.dumps(out, sort_keys=True, separators=(",", ":")))
