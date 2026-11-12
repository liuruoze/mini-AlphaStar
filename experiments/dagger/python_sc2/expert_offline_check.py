"""Offline verification: python-sc2 BotAI as a pure state -> actions function.

Uses the ResponseData / ResponseGameInfo protos captured earlier, builds a
synthetic ResponseObservation (two distinct game states), injects a fresh
WarpGateBot instance via _prepare_start/_prepare_step/on_step with a stub
client, and records every action passed to Client.actions.

No SC2 process, no network, no game loop. Fully reproducible.
"""
import asyncio
import importlib.util
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc2_compat  # noqa: F401  (patch ability-id handling BEFORE GameData)

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


async def _noop(*args, **kwargs):
    return None


def mk_unit(unit_type, tag, x, y, progress=1.0):
    u = raw_pb.Unit()
    u.unit_type = unit_type
    u.alliance = 1
    u.tag = tag
    u.owner = 1
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


def mk_obs(food_used, food_cap, minerals, units):
    ro = sc_pb.ResponseObservation()
    ro.observation.game_loop = 100
    pc = ro.observation.player_common
    pc.player_id = 1
    pc.minerals = minerals
    pc.vespene = 0
    pc.food_used = food_used
    pc.food_cap = food_cap
    pc.food_workers = food_used
    pc.food_army = 0
    for u in units:
        ro.observation.raw_data.units.add().CopyFrom(u)
    return ro


def load_bot_class():
    path = os.path.expanduser("~/sc2bots/warpgate_push.py")
    spec = importlib.util.spec_from_file_location("wg_bot_offline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.WarpGateBot


def decode_action(raw):
    a = sc_pb.Action()
    a.ParseFromString(raw)
    uc = a.action_raw.unit_command
    out = {
        "ability_id": uc.ability_id,
        "unit_tags": list(uc.unit_tags),
        "target_unit_tag": uc.target_unit_tag if uc.HasField("target_unit_tag") else None,
        "has_target_pos": uc.HasField("target_world_space_pos"),
    }
    return out


async def run_case(bot_cls, name, food_used, food_cap, minerals, units):
    CAPTURED.clear()
    ro = mk_obs(food_used, food_cap, minerals, units)
    gs = GameState(ro, GD)
    bot = bot_cls()
    bot._prepare_start(StubClient(), 1, GI, GD)
    bot.on_start()
    bot._prepare_step(gs)
    bot._prepare_first_step()
    bot.distribute_workers = _noop
    try:
        await bot.on_step(0)
        err = None
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
    raws = list(CAPTURED)
    with open(os.path.join(P, "actions_%s.pkl" % name), "wb") as f:
        pickle.dump(raws, f, protocol=2)
    with open(os.path.join(P, "obs_case_%s.pkl" % name), "wb") as f:
        pickle.dump(ro.SerializeToString(), f, protocol=2)
    return {
        "case": name,
        "supply_left": food_cap - food_used,
        "minerals": minerals,
        "n_units": len(units),
        "n_actions": len(raws),
        "actions": [decode_action(r) for r in raws],
        "error": err,
    }


async def main():
    print("[setup] GameData abilities=%d units=%d | GameInfo start_locations=%s"
          % (len(GD.abilities), len(GD.units), GI.start_locations), flush=True)
    bot_cls = load_bot_class()
    base = [mk_unit(59, 1001, 20, 20)]
    base += [mk_unit(84, 2000 + i, 21 + 0.5 * i, 20) for i in range(12)]
    res = []
    res.append(await run_case(bot_cls, "A_supply_blocked", 13, 14, 500, base))
    res.append(await run_case(bot_cls, "B_supply_ok", 12, 26, 500, base))
    with open(os.path.join(P, "offline_result.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
