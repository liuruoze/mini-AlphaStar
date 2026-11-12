# -*- coding: utf-8 -*-
"""严格时序验证：建 Pylon -> 等它 build_progress==1 -> 再建 Gateway。

用于判定 Gateway 失败到底是「能量场未就绪 / 位置非法 / 命令被忽略」。
"""
from absl import flags as _f
try:
    _f.FLAGS(["gwseq"])
except Exception:
    _f.FLAGS.mark_as_parsed()

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))

import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features as F
from pysc2.lib.actions import FunctionCall
from pysc2.lib.actions import RAW_FUNCTIONS as R

import _rule_expert_v3 as RE


def env_new(steps=6000):
    aif = sc2_env.AgentInterfaceFormat(
        feature_dimensions=F.Dimensions(screen=64, minimap=64),
        use_feature_units=True, use_raw_units=True, use_raw_actions=True,
        max_raw_actions=512, max_selected_units=512, raw_resolution=64)
    return sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.protoss),
                 sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
        agent_interface_format=aif, step_mul=8, game_steps_per_episode=steps,
        random_seed=3038, visualize=False, ensure_available_actions=False)


def units(ro, utype):
    return [(int(u.tag), round(u.pos.x, 1), round(u.pos.y, 1), round(u.build_progress, 2), u.is_powered)
            for u in ro.observation.raw_data.units
            if int(u.alliance) == 1 and int(u.unit_type) == utype]


def main():
    env = env_new()
    try:
        o = env.reset()[0]
        summ = RE.summarize(o.observation)
        print("anchor(feat)=", summ["anchor"], flush=True)
        probes = [i for i, u in enumerate(o.observation["raw_units"])
                  if int(u["alliance"]) == 1 and int(u["unit_type"]) == 84]
        idx = probes[0]

        # 1) Pylon at anchor+(0,4)
        ax, ay = summ["anchor"]
        px, py = ax, ay + 4
        env.step([[FunctionCall(R.Build_Pylon_pt.id, [[0], [idx], [px, py]])]])
        for i in range(120):
            env.step([[FunctionCall(R.no_op.id, [])]])
            py_u = units(env._obs[0], 60)
            if py_u and max(x[3] for x in py_u) >= 1.0:
                print("[pylon] done at step %d: %s" % (i, py_u), flush=True)
                break
        else:
            print("[pylon] NOT done in 120 frames:", units(env._obs[0], 60), flush=True)

        # 2) Gateway 在 Pylon 周围逐个偏移试
        for dx, dy in [(4, 0), (-4, 0), (0, 3), (0, -3), (3, 3), (-3, 3), (2, 2), (-2, 2)]:
            gx, gy = ax + dx, ay + dy
            ru = env._obs[0].observation.raw_data.units
            cand = [u.tag for u in ru if int(u.alliance) == 1 and int(u.unit_type) == 84]
            if not cand:
                print("  no probe left"); break
            # 用 raw_units 的 index（transform_action 需要 index）
            # raw_units 顺序 = raw_data.units 顺序
            pidx = next(i for i, u in enumerate(ru)
                        if int(u.alliance) == 1 and int(u.unit_type) == 84)
            env.step([[FunctionCall(R.Build_Gateway_pt.id, [[0], [pidx], [gx, gy]])]])
            got = 0
            for _ in range(60):
                env.step([[FunctionCall(R.no_op.id, [])]])
                got = len(units(env._obs[0], 62))
                if got:
                    break
            print("  gw at (%d,%d) -> %s" % (gx, gy, units(env._obs[0], 62) or "NONE"),
                  flush=True)
            if got:
                print("  === SUCCESS at (%d,%d) ===" % (gx, gy), flush=True)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
