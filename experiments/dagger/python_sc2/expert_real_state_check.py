"""Drive the offline python-sc2 expert with REAL captured game observations."""
import asyncio
import importlib.util
import json
import os
import pickle
import sys

sys.path.insert(0, "/home/liuruoze/mini-AlphaStar/experiments")
spec = importlib.util.spec_from_file_location(
    "offline_matrix",
    "/home/liuruoze/mini-AlphaStar/experiments/dagger/python_sc2/expert_offline_matrix.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

from sc2.game_state import GameState  # noqa: E402
from s2clientprotocol import sc2api_pb2 as sc_pb  # noqa: E402

P = "/home/liuruoze/bvb/expert_probe"
FRAMES = [0, 5, 10, 15, 20, 25, 29]
BOT_CLS = M.load_bot("~/sc2bots/warpgate_push.py", "WarpGateBot")
OUT = []


def load_ro(it):
    with open(os.path.join(P, "real_obs_%03d.pkl" % it), "rb") as f:
        b = pickle.load(f)
    ro = sc_pb.ResponseObservation()
    ro.ParseFromString(b)
    return ro


async def one(it, boost=False):
    ro = load_ro(it)
    if boost:
        ro.observation.player_common.minerals = 500
    gs = GameState(ro, M.GD)
    bot = BOT_CLS()
    bot._prepare_start(M.StubClient(), 1, M.GI, M.GD)
    bot.on_start()
    bot._prepare_step(gs)
    bot._prepare_first_step()
    bot.distribute_workers = M._noop
    M.CAPTURED.clear()
    err = None
    try:
        await bot.on_step(0)
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, str(exc)[:150])
    raws = list(M.CAPTURED)
    tag = "real_actions_boost_%03d.pkl" % it if boost else "real_actions_%03d.pkl" % it
    with open(os.path.join(P, tag), "wb") as f:
        pickle.dump(raws, f, protocol=2)
    item = {
        "frame": it,
        "boost_minerals_500": boost,
        "game_loop": ro.observation.game_loop,
        "n_units": len(ro.observation.raw_data.units),
        "minerals": bot.minerals,
        "supply_left": bot.supply_left,
        "workers": bot.workers.amount,
        "n_actions": len(raws),
        "aids": [M.decode(r)["ability_id"] for r in raws],
        "error": err,
    }
    OUT.append(item)
    print("[real %03d boost=%s] gl=%s units=%s minerals=%s supply_left=%s actions=%s err=%s"
          % (it, boost, item["game_loop"], item["n_units"], item["minerals"],
             item["supply_left"], item["aids"], err), flush=True)


async def main():
    for it in FRAMES:
        await one(it, False)
    for it in FRAMES:
        await one(it, True)
    with open(os.path.join(P, "real_state_result.json"), "w") as f:
        json.dump(OUT, f, indent=2)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
