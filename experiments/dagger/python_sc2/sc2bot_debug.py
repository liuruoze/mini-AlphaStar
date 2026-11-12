"""诊断：把 SC2 3.16 返回的 game_data 结构与 python-sc2 的期望对照打印出来。"""
import os
import sys

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sc2_compat  # noqa: E402,F401
from sc2 import run_game, maps, Race, Difficulty  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402
from sc2.bot_ai import BotAI  # noqa: E402


class Dbg(BotAI):
    def on_start(self):
        gd = self._game_data
        print("[dbg] units=%d abilities=%d upgrades=%d"
              % (len(gd.units), len(gd.abilities), len(gd.upgrades)), flush=True)
        for uid in (84, 59, 60):     # Probe / Nexus / Pylon
            u = gd.units.get(uid)
            if u is None:
                print("[dbg] unit %s MISSING" % uid, flush=True)
                continue
            p = u._proto
            fields = [f.name for f in p.DESCRIPTOR.fields]
            print("[dbg] unit %s name=%s avail=%s fields=%s"
                  % (uid, p.name, getattr(p, "available", "?"),
                     ",".join(fields)), flush=True)
            print("[dbg]    ability_id=%s creation_ability=%s cost=%s/%s/%s"
                  % (getattr(p, "ability_id", "NOFIELD"),
                     getattr(p, "creation_ability", "NOFIELD"),
                     getattr(p, "mineral_cost", "?"), getattr(p, "vespene_cost", "?"),
                     getattr(p, "food_required", "?")), flush=True)
            print("[dbg]    creation_ability_obj=%r" % (u.creation_ability,), flush=True)
        print("[dbg] ability sample=%s" % (sorted(gd.abilities)[:12],), flush=True)

    async def on_step(self, iteration):
        if iteration == 0:
            st = self.state
            rows = [(int(u._proto.unit_type), int(u._proto.alliance), bool(u.is_mineral_field))
                    for u in st.units]
            print("[dbg2] state_units=%d own=%d mineral=%d"
                  % (len(rows), sum(1 for r in rows if r[1] == 1),
                     sum(1 for r in rows if r[2])), flush=True)
            print("[dbg2] rows=%s" % (rows[:14],), flush=True)
            print("[dbg2] MINERAL_FIELD ids=%s"
                  % (sorted(int(x) for x in __import__("sc2").constants.MINERAL_FIELD),), flush=True)
        return


def main():
    run_game(maps.get("Simple64"),
             [Bot(Race.Protoss, Dbg()),
              Computer(Race.Protoss, Difficulty.Easy)],
             realtime=False, game_time_limit=3,
             save_replay_as="/home/liuruoze/bvb/matrix_logs/dbg_sc2bot.SC2Replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
