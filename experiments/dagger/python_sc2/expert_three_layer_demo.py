#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Three-layer expert integration skeleton (offline, no SC2 client).

  L1 strategy_engine.StrategicAdvisor.advise_bot -> strategic Advice
  L2 python-sc2 BotAI.on_step                    -> reactive concrete actions
  L3 strategy_engine.BuildExecutor.next_action  -> long-horizon opening step

The L1->L3 guide choice is a HEURISTIC demo mapping, NOT a validated policy.
The point is to test whether all three can read the SAME injected state and
whether the reactive layer (needed for DAgger-style correction) is the one
that actually responds to a stuck state.
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
from strategy_engine import build_guides as bg  # noqa: E402

ADV = StrategicAdvisor()
BOT_CLS = M.load_bot("~/sc2bots/warpgate_push.py", "WarpGateBot")
ABIL = json.load(open(os.path.expanduser(
    "~/mini-AlphaStar/experiments/sc2_unit_ability.json")))["units"]
TOKEN_UID = json.load(open(os.path.expanduser(
    "~/bvb/expert_probe/protoss_token_unitid.json")))

# DELIBERATELY heuristic: shows the wiring, not a tuned strategy selector.
GUIDE_BY_INVEST = {
    "SUPPLY": 203352,     # PiG 8-worker opener (short, build-only)
    "ECONOMY": 203088,    # Harstem PvT Gasless
    "ARMY": 203051,       # Beginner Protoss (longer, has army tech)
}

OUT = []


def have_from_bot(bot):
    """Approximate build state from the injected observation."""
    n_pylon = max(1, int(bot.supply_cap // 8)) if bot.supply_cap else 1
    return {
        "PYLON": n_pylon,
        "NEXUS": max(1, bot.townhalls.amount),
        "GATEWAY": max(1, bot.structures(M.UnitTypeId.GATEWAY).amount)
        if hasattr(M, "UnitTypeId") else 1,
        "PROBE": bot.workers.amount,
    }


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
    build_supply = "build_supply" in hits

    M.CAPTURED.clear()
    err = None
    try:
        await bot.on_step(0)
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    aids = [M.decode(r)["ability_id"] for r in M.CAPTURED]

    key = next((k for k in GUIDE_BY_INVEST if k in invest.upper()), "ECONOMY")
    build = bg.get_build(GUIDE_BY_INVEST[key])
    ex = bg.BuildExecutor(build, reproducible_only=True)
    have = have_from_bot(bot)
    step = ex.next_action(have)
    l3 = None
    if step is not None:
        uid = TOKEN_UID.get(step.token)
        l3 = {
            "action": step.action, "name": step.name, "token": step.token,
            "ability_id": ABIL.get(str(uid)) if uid else None,
            "at_supply": step.at_supply, "at_second": step.at_second,
        }

    item = {
        "case": name,
        "minerals": bot.minerals, "supply_left": bot.supply_left,
        "workers": bot.workers.amount,
        "l1_investment_top": invest, "l1_build_supply_hit": build_supply,
        "l2_reactive_aids": aids,
        "l3_guide_choice": key, "l3_guide_id": build.id,
        "l3_next_step": l3,
        "l3_have": have,
        "error": err,
    }
    OUT.append(item)
    print("[%s] supply_left=%-3d invest=%-8s build_supply=%-5s L2=%-18s L3=%s/%s aid=%s"
          % (name, bot.supply_left, invest[:8], build_supply, str(aids),
             key, l3["name"] if l3 else None,
             l3["ability_id"] if l3 else None), flush=True)


async def main():
    for case in M.CASES:
        await one(case)
    out = {
        "note": "L1->L3 guide mapping is a heuristic wiring demo, NOT a "
                "validated policy; all three layers read the same injected state.",
        "cases": OUT,
    }
    with open("/home/liuruoze/bvb/expert_probe/three_layer_demo.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
