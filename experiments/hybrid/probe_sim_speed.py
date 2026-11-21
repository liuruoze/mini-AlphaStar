# -*- coding: utf-8 -*-
"""实测仿真倍速：创建与评测同样的 SC2Env，跑 N 步 no-op，量 steps/s。

仿真倍速 = 步/秒 × STEP_MUL / 22.4
（SC2 里 22.4 个 game step = 1 游戏秒；STEP_MUL=8 表示每个 env.step 推进 8 个 game step）

用法：
  python experiments/hybrid/probe_sim_speed.py [步数]
"""

import os
import sys
import time

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pysc2.run_configs                            # noqa: E402  定义 --sc2_run_config
from absl import flags as _absl_flags               # noqa: E402
_absl_flags.FLAGS([sys.argv[0]])                     # 先解析，否则 SC2Env 取不到 run_config

from pysc2.env.sc2_env import (SC2Env, AgentInterfaceFormat, Agent, Race, Bot,  # noqa: E402
                               Difficulty)
from pysc2.lib import actions as sc2_actions        # noqa: E402
from alphastarmini.lib.hyper_parameters import (     # noqa: E402
    AlphaStar_Agent_Interface_Format_Params as AAIFP,
    StarCraft_Hyper_Parameters as SCHP)

STEP_MUL = 8


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    env = SC2Env(map_name=SCHP.map_name,
                 players=[Agent(Race.protoss, "p0"),
                          Bot(Race.protoss, Difficulty.easy)],
                 step_mul=STEP_MUL, game_steps_per_episode=None,
                 agent_interface_format=[aif], version=SCHP.game_version,
                 random_seed=1)
    env.reset()
    noop = [sc2_actions.FunctionCall(0, [])]
    t0 = time.time()
    for _ in range(steps):
        env.step([noop], step_mul=STEP_MUL)
    dt = time.time() - t0
    sps = steps / dt
    print("步数 %d ｜ 用时 %.2fs ｜ %.2f 步/秒" % (steps, dt, sps))
    print("仿真倍速 ≈ %.2f×（%.2f × %d / 22.4）" % (sps * STEP_MUL / 22.4, sps, STEP_MUL))
    env.close()


if __name__ == "__main__":
    main()
