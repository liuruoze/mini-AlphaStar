"""python-sc2 冒烟测试（适配 SC2 4.8 / Base55958 + python-sc2 0.8.4）。

只验证：能否创建游戏、能否拿到 state、能否下一个建造命令。
"""
import os
import sys

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import sc2.controller as _ctrl  # noqa: E402
from s2clientprotocol import sc2api_pb2 as _sc_pb  # noqa: E402
from sc2 import run_game, maps, Race, Difficulty  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402
from sc2.bot_ai import BotAI  # noqa: E402
from sc2.ids.unit_typeid import UnitTypeId  # noqa: E402


# ---- SC2 4.8 兼容补丁：地图用二进制传，不依赖路径解析 ----------------------
async def _create_game_patched(self, game_map, players, realtime):
    req = _sc_pb.RequestCreateGame(
        local_map=_sc_pb.LocalMap(map_path="Maps/Melee/Simple64.SC2Map",
                                  map_data=game_map.data),
        realtime=realtime,
    )
    for player in players:
        p = req.player_setup.add()
        p.type = player.type.value
        if isinstance(player, _ctrl.Computer):
            p.race = player.race.value
            p.difficulty = player.difficulty.value
    return await self._execute(create_game=req)


_ctrl.Controller.create_game = _create_game_patched


class Smoke(BotAI):
    async def on_step(self, iteration: int) -> None:
        if iteration == 0:
            self.client.game_step = 4
            print("[smoke] map_size=%s start=%s workers=%d minerals=%d"
                  % (self.game_info.map_size, self.start_location,
                     self.workers.amount, self.minerals), flush=True)
        if iteration == 5:
            w = self.workers.first
            if w is not None:
                target = self.start_location.towards(self.game_info.map_center, 6)
                ok = await self.build(UnitTypeId.PYLON, near=target)
                print("[smoke] build_pylon near %s -> %s" % (target, ok), flush=True)
        if iteration % 100 == 0 and iteration:
            print("[smoke] it=%d pylon=%d workers=%d minerals=%d"
                  % (iteration, self.structures(UnitTypeId.PYLON).amount,
                     self.workers.amount, self.minerals), flush=True)


def main() -> int:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_game(
        maps.get("Simple64"),
        [Bot(Race.Protoss, Smoke()),
         Computer(Race.Terran, Difficulty.VeryHard)],
        realtime=False,
        game_time_limit=seconds,
        save_replay_as="/home/liuruoze/bvb/matrix_logs/smoke_sc2bot.SC2Replay",
    )
    print("[smoke] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
