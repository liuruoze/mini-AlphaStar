"""通用 python-sc2 对局驱动：加载任意现成 bot 脚本，跟内置 AI 打 N 局。

适配 SC2 3.16.1 / Base55958 + python-sc2 0.8.4：
  * 老版本 SC2 的 local_map 不认绝对路径，改用 map_data 传地图二进制；
  * python-sc2 0.8.4 的 client.step() 固定 8 步，无需额外设置。

用法：
  python sc2bot_run.py --bot-file ~/sc2bots/warpgate_push.py --bot-class WarpGateBot \
      --games 5 --minutes 15 --difficulty very_hard --enemy protoss \
      --out ~/bvb/sc2bot_replays/warpgate
"""
import argparse
import asyncio
import importlib.util
import os
import sys
import time

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import sc2.controller as _ctrl  # noqa: E402
from s2clientprotocol import sc2api_pb2 as _sc_pb  # noqa: E402
from sc2 import run_game, maps, Race, Difficulty  # noqa: E402
from sc2.player import Bot, Computer  # noqa: E402


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc2_compat  # noqa: E402,F401  兼容补丁（地图/ability 字段）


# 0.8.4 的 _host_game 对非人类玩家返回 None，这里把 _play_game 的返回值打出来
import sc2.main as _sc2main  # noqa: E402

_orig_play_game = _sc2main._play_game


_LAST_RESULT = []


async def _play_game_logged(*args, **kwargs):
    res = await _orig_play_game(*args, **kwargs)
    _LAST_RESULT.append(res)
    print("[result] %s" % getattr(res, "name", res), flush=True)
    return res


_sc2main._play_game = _play_game_logged


DIFF = {
    "very_easy": Difficulty.VeryEasy,
    "easy": Difficulty.Easy,
    "medium": Difficulty.Medium,
    "medium_hard": Difficulty.MediumHard,
    "hard": Difficulty.Hard,
    "harder": Difficulty.Harder,
    "very_hard": Difficulty.VeryHard,
    "cheat_vision": Difficulty.CheatVision,
    "cheat_money": Difficulty.CheatMoney,
    "cheat_insane": Difficulty.CheatInsane,
}
RACES = {"protoss": Race.Protoss, "terran": Race.Terran, "zerg": Race.Zerg}


def load_bot_class(path, cls_name):
    spec = importlib.util.spec_from_file_location("candidate_bot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if cls_name:
        return getattr(mod, cls_name)
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.endswith("Bot"):
            return obj
    raise SystemExit("无法在 %s 中找到 Bot 类" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-file", required=True)
    ap.add_argument("--bot-class", default="")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--difficulty", default="very_hard", choices=sorted(DIFF))
    ap.add_argument("--enemy", default="protoss", choices=sorted(RACES))
    ap.add_argument("--out", default=os.path.expanduser("~/bvb/sc2bot_replays"))
    ap.add_argument("--port-base", type=int, default=46000,
                    help="每局占用 10 个连续端口，避免 portpicker 竞态")
    ap.add_argument("--gap", type=float, default=6.0, help="局间等待秒数")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    bot_cls = load_bot_class(a.bot_file, a.bot_class)
    print("[run] bot=%s games=%d enemy=%s %s limit=%.0fmin"
          % (bot_cls.__name__, a.games, a.enemy, a.difficulty, a.minutes), flush=True)

    wins = 0
    done = 0
    for g in range(1, a.games + 1):
        name = "None"
        for attempt in range(1, 7):          # 老版本 SC2 偶发端口冲突，重试
            replay = os.path.join(a.out, "g%d.SC2Replay" % g)
            try:
                _LAST_RESULT.clear()
                res = run_game(
                    maps.get("Simple64"),
                    [Bot(Race.Protoss, bot_cls()),
                     Computer(RACES[a.enemy], DIFF[a.difficulty])],
                    realtime=False,
                    game_time_limit=a.minutes * 60,
                    save_replay_as=replay,
                )
                if res is None and _LAST_RESULT:
                    res = _LAST_RESULT[-1]
            except Exception as exc:  # noqa: BLE001
                print("[game %d] attempt %d ERROR %r" % (g, attempt, exc), flush=True)
                name = "None"
                time.sleep(6)
                continue
            name = getattr(res, "name", str(res))
            if name != "None":
                break
            print("[game %d] attempt %d aborted (no result)" % (g, attempt), flush=True)
            time.sleep(6)

        done += 1
        if name.lower().startswith("victory"):
            wins += 1
        print("[game %d] result=%s" % (g, name), flush=True)
        if g < a.games:
            time.sleep(a.gap)

    print("[summary] %d/%d win (games completed=%d)" % (wins, a.games, done), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
