"""Systematic sweep: how does the offline expert's decision change with state?

Sweeps supply_left x minerals x worker_count; records both python-sc2 BotAI
concrete actions and strategy_engine Advice for the SAME injected instance.
"""
import asyncio
import importlib.util
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/home/liuruoze/mini-AlphaStar/experiments")
sys.path.insert(0, "/home/liuruoze/bvb/expert_probe/sc2_bots_src")

spec = importlib.util.spec_from_file_location(
    "offline_matrix",
    "/home/liuruoze/mini-AlphaStar/experiments/dagger/python_sc2/expert_offline_matrix.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

from sc2.game_state import GameState  # noqa: E402
from strategy_engine import StrategicAdvisor  # noqa: E402
from s2clientprotocol import sc2api_pb2 as sc_pb  # noqa: E402

P = "/home/liuruoze/bvb/expert_probe"
BOT_CLS = M.load_bot("~/sc2bots/warpgate_push.py", "WarpGateBot")
ADV = StrategicAdvisor()

SUPPLY = [0, 1, 2, 3, 6, 14]
MINERALS = [0, 50, 100, 300, 1000]
WORKERS = [6, 12, 22]
FOOD_USED = 12

AID_NAME = {
    881: "Build_Pylon_pt", 882: "Build_Assimilator_unit",
    883: "Build_Gateway_pt", 894: "Build_CyberneticsCore_pt",
    1006: "Train_Probe_quick", 3666: "Harvest_Gather_unit",
}


def build_obs(supply_left, minerals, n_workers):
    units = [M.mk_unit(59, 1001, 20, 20)]
    units += [M.mk_unit(84, 2000 + i, 21 + 0.4 * i, 20) for i in range(n_workers)]
    units += [M.mk_unit(665, 3000 + i, 15 + i, 20, alliance=3) for i in range(4)]
    units += [M.mk_unit(342, 3100 + i, 18 + 4 * i, 26, alliance=3) for i in range(2)]
    ro = M.mk_obs(FOOD_USED, FOOD_USED + supply_left, minerals, 0, units)
    return ro


async def one(supply_left, minerals, n_workers):
    ro = build_obs(supply_left, minerals, n_workers)
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
    err = None
    try:
        await bot.on_step(0)
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, str(exc)[:100])
    aids = [M.decode(r)["ability_id"] for r in M.CAPTURED]
    return {
        "supply_left": supply_left, "minerals": minerals, "workers": n_workers,
        "aids": aids, "names": [AID_NAME.get(a, str(a)) for a in aids],
        "invest": invest, "build_supply_hit": "build_supply" in hits,
        "error": err,
    }


async def main():
    rows = []
    for s in SUPPLY:
        for m in MINERALS:
            for w in WORKERS:
                rows.append(await one(s, m, w))
    with open(os.path.join(P, "state_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)

    print("=== total combinations: %d, errors: %d ==="
          % (len(rows), sum(1 for r in rows if r["error"])), flush=True)
    print("=== by supply_left (n_actions / pct with Build_Pylon / pct with 0 actions) ===")
    for s in SUPPLY:
        sub = [r for r in rows if r["supply_left"] == s]
        print("  supply_left=%-2d n=%-3d mean_actions=%.2f pylon=%d/%d zero=%d/%d"
              % (s, len(sub), sum(len(r["aids"]) for r in sub) / len(sub),
                 sum(1 for r in sub if 881 in r["aids"]), len(sub),
                 sum(1 for r in sub if not r["aids"]), len(sub)), flush=True)
    print("=== by minerals (mean actions) ===")
    for m in MINERALS:
        sub = [r for r in rows if r["minerals"] == m]
        print("  minerals=%-4d mean_actions=%.2f zero=%d/%d"
              % (m, sum(len(r["aids"]) for r in sub) / len(sub),
                 sum(1 for r in sub if not r["aids"]), len(sub)), flush=True)
    print("=== action histogram ===")
    c = Counter(a for r in rows for a in r["aids"])
    print(" ", {AID_NAME.get(k, k): v for k, v in c.most_common()}, flush=True)
    print("=== invest histogram ===")
    print(" ", dict(Counter(r["invest"] for r in rows)), flush=True)
    gate = [r for r in rows if r["supply_left"] < 2 and r["minerals"] >= 100]
    print("=== falsifiable gate: supply_left<2 & minerals>=100 -> Build_Pylon ===")
    print("  %d/%d cases produced 881" % (sum(1 for r in gate if 881 in r["aids"]), len(gate)), flush=True)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
