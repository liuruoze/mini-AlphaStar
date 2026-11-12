"""P2 extension: 3 python-sc2 bots x 3 synthetic states, fully offline.

Checks whether the state->actions wrapper is generic across the local bot
scripts, and whether one state can yield several different action types.
"""
import asyncio
import importlib.util
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc2_compat  # noqa: F401

from sc2.game_data import GameData
from sc2.game_info import GameInfo
from sc2.game_state import GameState
from sc2.action import combine_actions
from sc2.data import ActionResult
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

P = os.path.expanduser("~/bvb/expert_probe")
CAPTURED = []


def load_proto(name, cls):
    with open(os.path.join(P, name), "rb") as f:
        raw = pickle.load(f)
    obj = cls()
    obj.ParseFromString(raw)
    return obj


GD = GameData(load_proto("response_data.pkl", sc_pb.ResponseData))
GI = GameInfo(load_proto("response_game_info.pkl", sc_pb.ResponseGameInfo))


class StubClient:
    async def actions(self, action, game_data=None, return_successes=False):
        acts = action if isinstance(action, list) else [action]
        for a in combine_actions(acts, game_data):
            CAPTURED.append(sc_pb.Action(action_raw=a).SerializeToString())
        return None

    async def query_available_abilities(self, unit):
        return []

    async def query_building_placement(self, ability, positions, ignore_resources=True):
        return [ActionResult.Success for _ in positions]

    async def chat_send(self, *args, **kwargs):
        return None

    async def query_pathing(self, start, end):
        return 0.0

    async def query_available_abilities_with_tag(self, *args, **kwargs):
        return []


async def _noop(*args, **kwargs):
    return None


def mk_unit(unit_type, tag, x, y, alliance=1, progress=1.0):
    u = raw_pb.Unit()
    u.unit_type = unit_type
    u.alliance = alliance
    u.tag = tag
    u.owner = 1 if alliance == 1 else 0
    u.pos.x = float(x)
    u.pos.y = float(y)
    u.pos.z = 0.0
    u.facing = 0.0
    u.radius = 0.5
    u.build_progress = progress
    u.display_type = 2
    u.health = 100.0
    u.health_max = 100.0
    u.shield = 0.0
    u.shield_max = 0.0
    u.energy = 0.0
    u.energy_max = 0.0
    u.is_powered = True
    return u


def mk_obs(food_used, food_cap, minerals, vespene, units):
    ro = sc_pb.ResponseObservation()
    ro.observation.game_loop = 100
    pc = ro.observation.player_common
    pc.player_id = 1
    pc.minerals = minerals
    pc.vespene = vespene
    pc.food_used = food_used
    pc.food_cap = food_cap
    pc.food_workers = food_used
    pc.food_army = 0
    for u in units:
        ro.observation.raw_data.units.add().CopyFrom(u)
    return ro


def load_bot(path, cls_name):
    spec = importlib.util.spec_from_file_location("bot_" + cls_name, os.path.expanduser(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


BOTS = [
    ("warpgate", "~/sc2bots/warpgate_push.py", "WarpGateBot"),
    ("cannon", "~/sc2bots/cannon_rush.py", "CannonRushBot"),
    ("voidray", "~/sc2bots/threebase_voidray.py", "ThreebaseVoidrayBot"),
]

BASE = [mk_unit(59, 1001, 20, 20)]
BASE += [mk_unit(84, 2000 + i, 21 + 0.5 * i, 20) for i in range(12)]
MINERALS = [mk_unit(665, 3000 + i, 15 + i, 20, alliance=3) for i in range(4)]
GASES = [mk_unit(342, 3100 + i, 18 + 4 * i, 26, alliance=3) for i in range(2)]

TECH = list(BASE) + MINERALS + GASES
TECH += [mk_unit(60, 4000 + i, 24 + i, 22) for i in range(2)]       # pylons
TECH += [mk_unit(62, 4100, 26, 24)]                                  # gateway
TECH += [mk_unit(69, 4200, 28, 24)]                                  # cyber core
TECH += [mk_unit(73, 4300 + i, 30 + 0.5 * i, 26) for i in range(4)]   # stalkers

CASES = [
    ("A_supply_blocked", 13, 14, 500, 0, list(BASE) + MINERALS + GASES),
    ("B_supply_ok", 12, 26, 500, 0, list(BASE) + MINERALS + GASES),
    ("C_tech_army", 24, 30, 1000, 300, TECH),
]


def decode(raw):
    a = sc_pb.Action()
    a.ParseFromString(raw)
    uc = a.action_raw.unit_command
    return {
        "ability_id": uc.ability_id,
        "unit_tags": list(uc.unit_tags),
        "has_target_pos": uc.HasField("target_world_space_pos"),
        "target_unit_tag": uc.target_unit_tag if uc.HasField("target_unit_tag") else None,
    }


async def run(bot_cls, bot_name, case):
    name, fu, fc, mins, ves, units = case
    CAPTURED.clear()
    ro = mk_obs(fu, fc, mins, ves, units)
    gs = GameState(ro, GD)
    bot = bot_cls()
    bot._prepare_start(StubClient(), 1, GI, GD)
    bot.on_start()
    bot._prepare_step(gs)
    bot._prepare_first_step()
    bot.distribute_workers = _noop
    err = None
    try:
        await bot.on_step(0)
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, str(exc)[:180])
    raws = list(CAPTURED)
    stem = "%s_%s" % (bot_name, name)
    with open(os.path.join(P, "matrix_actions_%s.pkl" % stem), "wb") as f:
        pickle.dump(raws, f, protocol=2)
    return {
        "bot": bot_name, "case": name,
        "n_actions": len(raws),
        "actions": [decode(r) for r in raws],
        "error": err,
    }


async def main():
    res = []
    for bot_name, path, cls_name in BOTS:
        bot_cls = load_bot(path, cls_name)
        for case in CASES:
            r = await run(bot_cls, bot_name, case)
            res.append(r)
            print("[%s/%s] n=%d err=%s aids=%s"
                  % (bot_name, r["case"], r["n_actions"], r["error"],
                     [a["ability_id"] for a in r["actions"]]), flush=True)
    for case in CASES:
        name, fu, fc, mins, ves, units = case
        ro = mk_obs(fu, fc, mins, ves, units)
        with open(os.path.join(P, "matrix_obs_%s.pkl" % name), "wb") as f:
            pickle.dump(ro.SerializeToString(), f, protocol=2)
    with open(os.path.join(P, "matrix_result.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
