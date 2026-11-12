"""Compare tactical diversity of the 3 local python-sc2 bots across 90 states."""
import asyncio
import importlib.util
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/home/liuruoze/mini-AlphaStar/experiments")
spec = importlib.util.spec_from_file_location(
    "sweep", "/home/liuruoze/mini-AlphaStar/experiments/dagger/python_sc2/expert_state_sweep.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

M = S.M
P = "/home/liuruoze/bvb/expert_probe"

BOTS = [
    ("warpgate", "~/sc2bots/warpgate_push.py", "WarpGateBot"),
    ("cannon", "~/sc2bots/cannon_rush.py", "CannonRushBot"),
    ("voidray", "~/sc2bots/threebase_voidray.py", "ThreebaseVoidrayBot"),
]
SUPPLY = [0, 1, 2, 3, 6, 14]
MINERALS = [0, 50, 100, 300, 1000]
WORKERS = [6, 12, 22]


async def run_bot(cls, name):
    rows = []
    for s in SUPPLY:
        for m in MINERALS:
            for w in WORKERS:
                ro = S.build_obs(s, m, w)
                gs = M.GameState(ro, M.GD)
                bot = cls()
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
                    err = "%s: %s" % (type(exc).__name__, str(exc)[:80])
                aids = [M.decode(r)["ability_id"] for r in M.CAPTURED]
                rows.append({"s": s, "m": m, "w": w, "aids": aids, "error": err})
    return rows


async def main():
    out = {}
    for name, path, cls_name in BOTS:
        cls = M.load_bot(path, cls_name)
        rows = await run_bot(cls, name)
        hist = Counter(a for r in rows for a in r["aids"])
        distinct = sorted(set(tuple(r["aids"]) for r in rows))
        out[name] = {
            "n_states": len(rows),
            "n_errors": sum(1 for r in rows if r["error"]),
            "total_actions": sum(len(r["aids"]) for r in rows),
            "zero_action_states": sum(1 for r in rows if not r["aids"]),
            "action_hist": {S.AID_NAME.get(k, str(k)): v for k, v in hist.most_common()},
            "distinct_decisions": len(distinct),
            "error_samples": sorted(set(r["error"] for r in rows if r["error"]))[:3],
        }
        print("[%s] states=%d errors=%d actions=%d zero=%d distinct_decisions=%d"
              % (name, out[name]["n_states"], out[name]["n_errors"],
                 out[name]["total_actions"], out[name]["zero_action_states"],
                 out[name]["distinct_decisions"]), flush=True)
        print("     hist=%s" % out[name]["action_hist"], flush=True)
        if out[name]["error_samples"]:
            print("     err_sample=%s" % out[name]["error_samples"][0][:100], flush=True)
    with open(os.path.join(P, "bot_diversity.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
