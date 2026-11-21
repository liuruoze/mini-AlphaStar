# -*- coding: utf-8 -*-
"""验证 state_from_obs 提取的单位数是否与游戏真实值一致。"""
import os, sys, time
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments", "hybrid"))

import pysc2.run_configs
from absl import flags as F
F.FLAGS([sys.argv[0]])
from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent, Race, Bot, Difficulty
from pysc2.lib import actions as A
from alphastarmini.lib.hyper_parameters import (AlphaStar_Agent_Interface_Format_Params as AAF,
                                                StarCraft_Hyper_Parameters as SHP)
from commander_v3 import state_from_obs

env = SC2Env(map_name=SHP.map_name, players=[Agent(Race.protoss, "p0"), Bot(Race.protoss, Difficulty(1))],
             step_mul=8, game_steps_per_episode=None,
             agent_interface_format=[AgentInterfaceFormat(**AAF._asdict())],
             version=SHP.game_version, random_seed=1)
obs = env.reset()[0]
noop = [A.FunctionCall(0, [])]
for i in range(60):
    obs = env.step([noop], step_mul=8)[0]
st = state_from_obs(obs.observation)
pl = obs.observation.player
print("游戏真实: food_used=%d cap=%d army=%d" % (pl.food_used, pl.food_cap, pl.army_count))
print("state_from_obs: %s" % {k: st[k] for k in ("used", "cap", "workers", "army", "gw", "pylons")})
print("raw_units 条数: %d" % len(obs.observation.raw_units))
from collections import Counter
c = Counter()
for u in obs.observation.raw_units:
    if u.alliance == 1:
        c[int(u.unit_type)] += 1
print("己方单位构成(前6):", c.most_common(6))
env.close()
