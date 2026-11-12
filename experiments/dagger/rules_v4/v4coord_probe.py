#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""坐标语义探针：pysc2 RAW_FUNCTIONS.Build_Pylon_pt 的 target 到底是 world 还是 minimap？

做法：以 Nexus 的 raw_units 坐标（minimap 尺度）为锚，在 anchor+(0,4) 下令建 Pylon，
然后从 raw proto（env._obs[0].observation.raw_data.units）读出建成 Pylon 的真实
world 坐标 pos.x/pos.y，与两种换算比较：
  - 若 pos ≈ (mmx*1.5+0.75, 96-(mmy*1.5+0.75)) ⇒ 传 minimap 坐标正确（pysc2 会转）
  - 若 pos ≈ (mmx, mmy)                      ⇒ 必须自己传 world 坐标
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                    # noqa: E402
from pysc2.env import sc2_env                         # noqa: E402
from pysc2.lib import actions as A                    # noqa: E402
from pysc2.lib import features as F                   # noqa: E402

import _rule_expert_v3 as RE                          # noqa: E402

MAP_H = 96.0


def to_world(mmx, mmy):
    return (mmx * 1.5 + 0.75, MAP_H - (mmy * 1.5 + 0.75))


def pylons(obs):
    out = {}
    for u in obs.observation["raw_units"]:
        if int(u["alliance"]) == 1 and int(u["unit_type"]) == RE.PYLON:
            out[int(u["tag"])] = (int(u["x"]), int(u["y"]))
    return out


def proto_units(env):
    """{tag: (pos.x, pos.y)} 来自原始 proto（真实 world 坐标）。"""
    out = {}
    try:
        ts = env._obs[0]
        for u in ts.observation.raw_data.units:
            out[int(u.tag)] = (float(u.pos.x), float(u.pos.y))
    except Exception as e:
        print("[proto] err", e, flush=True)
    return out


def main():
    aif = sc2_env.AgentInterfaceFormat(
        feature_dimensions=F.Dimensions(screen=64, minimap=64),
        use_feature_units=True, use_raw_units=True, use_raw_actions=True,
        max_raw_actions=512, max_selected_units=512, raw_resolution=64)
    with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
            agent_interface_format=aif, step_mul=8, game_steps_per_episode=4000,
            visualize=False) as env:
        obs = env.reset()[0]
        for _ in range(30):
            obs = env.step([A.RAW_FUNCTIONS.no_op()])[0]
        ru = obs.observation["raw_units"]
        nx = [u for u in ru if int(u["alliance"]) == 1
              and int(u["unit_type"]) == RE.NEXUS]
        pr = [(i, u) for i, u in enumerate(ru)
              if int(u["alliance"]) == 1 and int(u["unit_type"]) == RE.PROBE]
        if not nx or not pr:
            print("[probe] missing nexus/probe", len(nx), len(pr), flush=True)
            return
        ax, ay = int(nx[0]["x"]), int(nx[0]["y"])
        idx, pu = pr[0]
        tx, ty = ax + 0, ay + 4
        wx, wy = to_world(tx, ty)
        print("[probe] nexus_mm=(%d,%d) probe#%d mm=(%d,%d) tag=%d"
              % (ax, ay, idx, int(pu["x"]), int(pu["y"]), int(pu["tag"])), flush=True)
        print("[probe] target mm=(%d,%d)  back_pt->world=(%.2f,%.2f)"
              % (tx, ty, wx, wy), flush=True)
        pu_world = proto_units(env).get(int(pu["tag"]))
        if pu_world:
            print("[probe] probe real world=(%.2f,%.2f)" % pu_world, flush=True)

        before = pylons(obs)
        fc = A.RAW_FUNCTIONS.Build_Pylon_pt(0, [idx], [tx, ty])
        obs = env.step([fc])[0]
        new = {}
        for k in range(80):
            obs = env.step([A.RAW_FUNCTIONS.no_op()])[0]
            cur = pylons(obs)
            add = set(cur) - set(before)
            if add:
                new = {t: cur[t] for t in add}
                print("[probe] pylon appeared after %d steps" % (k + 1), flush=True)
                break
        if not new:
            print("[probe] NO new pylon in 80 steps -> command rejected", flush=True)
            return
        pw = proto_units(env)
        for t, mm in new.items():
            print("[probe] new pylon tag=%d mm=(%d,%d) real_world=%s"
                  % (t, mm[0], mm[1], pw.get(t)), flush=True)
            print("[probe]   |Δ| vs back_pt = %s"
                  % (None if t not in pw else
                     (round(abs(pw[t][0] - wx), 2), round(abs(pw[t][1] - wy), 2))),
                  flush=True)
            print("[probe]   |Δ| vs raw mm  = %s"
                  % (None if t not in pw else
                     (round(abs(pw[t][0] - tx), 2), round(abs(pw[t][1] - ty), 2))),
                  flush=True)


if __name__ == "__main__":
    from absl import flags as _f
    _f.FLAGS(sys.argv[:1])
    main()
