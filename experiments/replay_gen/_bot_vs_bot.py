#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：两个 SC2 内置脚本 AI（Computer）能否互相对打并保存 replay。

背景
----
pysc2 的高层 API `sc2_env.SC2Env` 在 sc2_env.py:219-223 硬性要求
「至少一个 Agent」：

    self._num_agents = sum(1 for p in players if isinstance(p, Agent))
    if not 1 <= num_players <= 2 or not self._num_agents:
        raise ValueError("Only 1 or 2 players with at least one agent is ...")

所以 `players=[Bot, Bot]` 必然 ValueError。这是 **pysc2 封装的限制**，
不是 SC2 引擎的限制。

本脚本绕过 SC2Env，直接用 pysc2 的底层 controller 原语：
    create_game(RequestCreateGame(player_setup=[Computer, Computer]))
    join_game(RequestJoinGame(observed_player_id=1))   # 本地端以观战身份
    step(count=N) 循环推进
    save_replay() 落盘

用法（必须带 DISPLAY / SC2PATH）：
    DISPLAY=:1024 SC2PATH=$HOME/StarCraftII \
      python experiments/replay_gen/_bot_vs_bot.py --diff-p easy --diff-t easy \
             --max-loops 12000 --out-dir /tmp/bvb

退出码：0 = 成功打完并存下 replay；2 = 创建/加入对局失败；3 = 打到上限仍未结束。
"""

import argparse
import os
import sys
import time

from absl import flags

from pysc2 import run_configs
from s2clientprotocol import common_pb2 as com_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(["bot_vs_bot"])

MAP_PATH = "Maps/Melee/Simple64.SC2Map"

RACES = {
    "protoss": com_pb.Protoss,
    "terran": com_pb.Terran,
    "zerg": com_pb.Zerg,
    "random": com_pb.Random,
}
DIFFS = {
    "veryeasy": sc_pb.VeryEasy,
    "easy": sc_pb.Easy,
    "medium": sc_pb.Medium,
    "mediumhard": sc_pb.MediumHard,
    "hard": sc_pb.Hard,
    "harder": sc_pb.Harder,
    "veryhard": sc_pb.VeryHard,
    "cheatvision": sc_pb.CheatVision,
    "cheatmoney": sc_pb.CheatMoney,
    "cheatinsane": sc_pb.CheatInsane,
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-p", default="protoss", choices=sorted(RACES))
    ap.add_argument("--race-t", default="terran", choices=sorted(RACES))
    ap.add_argument("--diff-p", default="easy", choices=sorted(DIFFS))
    ap.add_argument("--diff-t", default="easy", choices=sorted(DIFFS))
    ap.add_argument("--max-loops", type=int, default=12000,
                    help="硬上限（game loop）；22.4 loop ≈ 1 秒，12000 ≈ 8.9 分钟")
    ap.add_argument("--chunk", type=int, default=64, help="每次 step 推进的 loop 数")
    ap.add_argument("--out-dir", default="/tmp/bvb")
    ap.add_argument("--tag", default="bvb")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rc = run_configs.get()
    log("run_config = %s" % type(rc).__name__)

    # 直接用地图文件的二进制（LocalMap.map_data），
    # 既避开 SC2 的相对路径解析，也避开 pysc2 只认内置地图表的 map_data()。
    map_file = os.path.expanduser(
        os.environ.get("BVB_MAP_FILE", "~/StarCraftII/Maps/Melee/Simple64.SC2Map"))
    with open(map_file, "rb") as f:
        map_data = f.read()
    log("map file = %s（%.1f KB）" % (map_file, len(map_data) / 1024.0))

    req_create = sc_pb.RequestCreateGame(
        local_map=sc_pb.LocalMap(map_data=map_data),
        player_setup=[
            sc_pb.PlayerSetup(type=sc_pb.Computer, race=RACES[args.race_p],
                              difficulty=DIFFS[args.diff_p], player_name="botP"),
            sc_pb.PlayerSetup(type=sc_pb.Computer, race=RACES[args.race_t],
                              difficulty=DIFFS[args.diff_t], player_name="botT"),
            # 关键：SC2 引擎要求「至少一个 participant 或 observer」，
            # 纯 2×Computer 会被 InvalidPlayerSetup 拒绝。
            # 加一个 Observer 槽（观战不占玩家位），本地客户端就以旁观者身份接入。
            sc_pb.PlayerSetup(type=sc_pb.Observer, player_name="spectator"),
        ],
        realtime=False,
    )

    t0 = time.time()
    # 注意：本版 pysc2 的 RunConfig.start() 直接返回 RemoteController
    # （不是 StarcraftProcess），所以不用再取 .controller。
    with rc.start() as c:
        log("controller = %s" % type(c).__name__)

        # ---- 1) 创建对局：两个 Player 都是 Computer ----
        try:
            c.create_game(req_create)
            log("create_game OK（2 × Computer：%s/%s vs %s/%s）"
                % (args.race_p, args.diff_p, args.race_t, args.diff_t))
        except Exception as e:
            log("create_game FAILED: %s: %s" % (type(e).__name__, e))
            return 2

        # ---- 2) 本地客户端以「观战」身份加入（关键：不能作为 Participant） ----
        try:
            c.join_game(sc_pb.RequestJoinGame(
                observed_player_id=1,
                options=sc_pb.InterfaceOptions(),
            ))
            log("join_game OK（observed_player_id=1，本地为观战者）")
        except Exception as e:
            log("join_game FAILED: %s: %s" % (type(e).__name__, e))
            return 2

        # ---- 3) 推进游戏 ----
        loops = 0
        ended = False
        results = None
        while loops < args.max_loops:
            try:
                c.step(count=args.chunk)
            except Exception as e:
                log("step 结束于 loop=%d：%s: %s" % (loops, type(e).__name__, e))
                ended = True
                break
            loops += args.chunk
            obs = c.observe()
            gl = obs.observation.game_loop
            if obs.player_result:
                results = [(r.player_id, r.result) for r in obs.player_result]
            if gl == 2 ** 32 - 1 or results:
                ended = True
                break
            if loops % 2000 == 0:
                log("  进度 loop=%d（%.1f 分钟游戏时间）"
                    % (loops, loops / 22.4 / 60.0))

        log("对局结束=%s  loop=%d（%.1f 分钟游戏时间）墙钟=%.0fs"
            % (ended, loops, loops / 22.4 / 60.0, time.time() - t0))
        if results:
            log("player_result = %s" % results)

        # ---- 4) 保存 replay ----
        out = None
        try:
            data = c.save_replay()
            out = rc.save_replay(data, args.out_dir, prefix=args.tag)
            log("save_replay OK → %s（%.1f KB）" % (out, os.path.getsize(out) / 1024.0))
        except Exception as e:
            log("save_replay FAILED: %s: %s" % (type(e).__name__, e))

        try:
            c.leave()
        except Exception:
            pass

    if out is None:
        return 2
    log("DONE")
    return 0 if ended else 3


if __name__ == "__main__":
    sys.exit(main())
