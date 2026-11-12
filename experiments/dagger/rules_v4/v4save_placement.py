#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 Simple64 官方 placement_grid 到 npy（88x96，1=可建）。

用途：off 仓评测链路（rl_eval_sl.py）里拿不到整洁的 game_info 注入点，
但 Simple64 是固定地图 ⇒ 导一次、由 _expert_intervene.py 直接 np.load。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                    # noqa: E402
from pysc2.env import sc2_env                         # noqa: E402
from pysc2.lib import features as F                   # noqa: E402


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "simple64_placement.npy")
    aif = sc2_env.AgentInterfaceFormat(
        feature_dimensions=F.Dimensions(screen=64, minimap=64),
        use_feature_units=True, use_raw_units=True, use_raw_actions=True,
        max_raw_actions=512, max_selected_units=512, raw_resolution=64)
    with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
            agent_interface_format=aif, step_mul=8, game_steps_per_episode=200,
            visualize=False) as env:
        gi = env.game_info
        if isinstance(gi, (list, tuple)):
            gi = gi[0]
        pg = gi.start_raw.placement_grid
        arr = np.frombuffer(pg.data, dtype=np.uint8).reshape(pg.size.y, pg.size.x)
        grid = (arr > 0).astype(np.uint8)
        np.save(out, grid)
        print("[save] %s shape=%s buildable=%d" % (out, grid.shape, int(grid.sum())),
              flush=True)


if __name__ == "__main__":
    from absl import flags as _f
    _f.FLAGS(sys.argv[:1])
    main()
