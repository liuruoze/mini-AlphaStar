"""Find the trigger thresholds of the upper strategy layer vs the lower bot."""
import asyncio
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/liuruoze/mini-AlphaStar/experiments")
sys.path.insert(0, "/home/liuruoze/bvb/expert_probe/sc2_bots_src")
spec = importlib.util.spec_from_file_location(
    "sweep", "/home/liuruoze/mini-AlphaStar/experiments/dagger/python_sc2/expert_state_sweep.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

M = S.M
from sc2.game_state import GameState  # noqa: E402
from strategy_engine import StrategicAdvisor  # noqa: E402

ADV = StrategicAdvisor()
BOT_CLS = M.load_bot("~/sc2bots/warpgate_push.py", "WarpGateBot")
P = "/home/liuruoze/bvb/expert_probe"


async def one(s):
    ro = S.build_obs(s, 500, 12)
    gs = GameState(ro, M.GD)
    bot = BOT_CLS()
    bot._prepare_start(M.StubClient(), 1, M.GI, M.GD)
    bot.on_start()
    bot._prepare_step(gs)
    bot._prepare_first_step()
    bot.distribute_workers = M._noop
    adv = ADV.advise_bot(bot, {})
    invest = str(getattr(getattr(adv, "investment", None), "top", "?")).split(".")[-1]
    hits = [str(getattr(h, "rule", h)) for h in getattr(adv, "rule_hits", [])]
    M.CAPTURED.clear()
    await bot.on_step(0)
    aids = [M.decode(r)["ability_id"] for r in M.CAPTURED]
    return {"supply_left": s, "invest": invest,
            "build_supply_hit": "build_supply" in hits,
            "pylon_881": 881 in aids, "aids": aids}


async def main():
    rows = [await one(s) for s in range(0, 21)]
    print("s  invest    hit   881  aids")
    for r in rows:
        print("%-2d %-8s %-6s %-4s %s" % (
            r["supply_left"], r["invest"], r["build_supply_hit"],
            r["pylon_881"], r["aids"]), flush=True)
    upper_off = next((r["supply_left"] for r in rows if not r["build_supply_hit"]), None)
    lower_off = next((r["supply_left"] for r in rows if not r["pylon_881"]), None)
    mism = [r["supply_left"] for r in rows if r["build_supply_hit"] != r["pylon_881"]]
    print("upper build_supply first OFF at supply_left=", upper_off, flush=True)
    print("lower Build_Pylon  first OFF at supply_left=", lower_off, flush=True)
    print("mismatch supply_left values=", mism, flush=True)
    with open(os.path.join(P, "threshold_analysis.json"), "w") as f:
        json.dump({"rows": rows, "upper_off": upper_off,
                   "lower_off": lower_off, "mismatch": mism}, f, indent=2)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
