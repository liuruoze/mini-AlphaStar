"""Verify strategy_engine.advise_bot() can read our offline python-sc2 BotAI directly.

Combines the two expert layers on the SAME injected state:
  strategy_engine -> strategic Advice (what to prioritise / whether to engage)
  python-sc2 BotAI -> concrete sc_pb.Action protos (what to click)
"""
import asyncio
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/liuruoze/mini-AlphaStar/experiments")
sys.path.insert(0, "/home/liuruoze/bvb/expert_probe/sc2_bots_src")

spec = importlib.util.spec_from_file_location(
    "offline_matrix",
    "/home/liuruoze/mini-AlphaStar/experiments/dagger/python_sc2/expert_offline_matrix.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

from strategy_engine import StrategicAdvisor  # noqa: E402

ADV = StrategicAdvisor()
BOT_CLS = M.load_bot("~/sc2bots/warpgate_push.py", "WarpGateBot")
OUT = []


async def one(case):
    name, fu, fc, mins, ves, units = case
    ro = M.mk_obs(fu, fc, mins, ves, units)
    gs = M.GameState(ro, M.GD)
    bot = BOT_CLS()
    bot._prepare_start(M.StubClient(), 1, M.GI, M.GD)
    bot.on_start()
    bot._prepare_step(gs)
    bot._prepare_first_step()
    bot.distribute_workers = M._noop

    advice = ADV.advise_bot(bot, {})
    invest = str(getattr(getattr(advice, "investment", None), "top", "?"))
    hits = [str(getattr(h, "rule", h)) for h in getattr(advice, "rule_hits", [])]
    engagement = str(getattr(getattr(advice, "engagement", None), "stance", "?"))

    M.CAPTURED.clear()
    err = None
    try:
        await bot.on_step(0)
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    aids = [M.decode(r)["ability_id"] for r in M.CAPTURED]

    item = {
        "case": name,
        "bot_minerals": bot.minerals,
        "bot_supply_left": bot.supply_left,
        "bot_workers": bot.workers.amount,
        "investment_top": invest,
        "build_supply_hit": "build_supply" in hits,
        "engagement_stance": engagement,
        "rule_hits": hits,
        "concrete_action_aids": aids,
        "error": err,
    }
    OUT.append(item)
    print("[%s] supply_left=%d invest=%s build_supply=%s actions=%s"
          % (name, item["bot_supply_left"], invest, item["build_supply_hit"], aids),
          flush=True)


async def main():
    for case in M.CASES:
        await one(case)
    with open("/home/liuruoze/bvb/expert_probe/advisor_on_bot.json", "w") as f:
        json.dump(OUT, f, indent=2)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
