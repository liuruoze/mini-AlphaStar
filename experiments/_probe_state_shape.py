"""T2 前置探测：MsState 三段的真实形状与体积。

四份计划都只用"1.31 MB/帧"这个笼统数字，没人给出跨进程传输的具体布局。
本脚本跑一次真实 env，打印 entity / statistical(list) / map 的逐段形状与字节数，
用来设计 batch_server 的共享内存布局。
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent as SC2Agent, Race, Bot, Difficulty

import param as P
P.on_server = True

from alphastarmini.core.arch.agent import Agent
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP


def make_env(idx=0):
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    return SC2Env(map_name="Simple64",
                  players=[SC2Agent(Race.protoss, "p%d" % idx),
                           Bot(Race.random, Difficulty(1))],
                  step_mul=8, agent_interface_format=[aif], random_seed=idx + 991)


def nbytes(t):
    n = 1
    for s in t.shape:
        n *= s
    return n * t.element_size()


def main():
    env = make_env(0)
    obs = env.reset()[0].observation
    for _ in range(3):
        from pysc2.lib import actions as A
        obs = env.step([A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])])[0].observation

    st = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
    total = 0
    print("entity_state      ", tuple(st.entity_state.shape), st.entity_state.dtype,
          "%.1f KB" % (nbytes(st.entity_state) / 1024))
    total += nbytes(st.entity_state)
    print("map_state         ", tuple(st.map_state.shape), st.map_state.dtype,
          "%.1f KB" % (nbytes(st.map_state) / 1024))
    total += nbytes(st.map_state)
    print("statistical_state : %d 项" % len(st.statistical_state))
    for i, s in enumerate(st.statistical_state):
        print("   [%2d] %-22s %-8s %.2f KB" % (i, str(tuple(s.shape)), str(s.dtype), nbytes(s) / 1024))
        total += nbytes(s)
    print("--- 合计 %.2f KB / 帧 ---" % (total / 1024))
    try:
        ru = obs["raw_units"] if not hasattr(obs, "raw_units") else obs.raw_units
        print("raw_units type:", type(ru), "count:", len(ru))
        print("raw_units[0]:", ru[0])
    except Exception as e:
        print("raw_units 探测失败:", repr(e))
    env.close()


if __name__ == "__main__":
    main()
