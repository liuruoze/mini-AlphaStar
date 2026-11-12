# -*- coding: utf-8 -*-
"""Gateway 建造探针：按 HierNet 的 get_available_area + 膨胀 选候选，逐个实测。

对每个候选：发 Build_Gateway_pt -> 等 200 帧 -> 看 Gateway 是否出现 / Probe orders。
输出成功率与每步细节，用于定位「为什么建不出来」。
"""
from absl import flags as _f
try:
    _f.FLAGS(["gwprobe"])
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


def env_new():
    aif = sc2_env.AgentInterfaceFormat(
        feature_dimensions=F.Dimensions(screen=64, minimap=64),
        use_feature_units=True, use_raw_units=True, use_raw_actions=True,
        max_raw_actions=512, max_selected_units=512, raw_resolution=64)
    return sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.protoss),
                 sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
        agent_interface_format=aif, step_mul=8, game_steps_per_episode=4000,
        random_seed=3038, visualize=False, ensure_available_actions=False)


def dilate(mask, r):
    """HierNet 的 dialted_area 等价实现（方形膨胀 r 格）。"""
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:, :] |= out[:-1, :]
        m[:-1, :] |= out[1:, :]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def candidates(summ, half=1, n=8, power_r=None):
    mm = summ["mm"]
    hm = np.asarray(mm["height_map"]).astype(np.int32)
    vm = np.asarray(mm["visibility_map"]).astype(np.int32)
    ut = np.asarray(mm["unit_type"]).astype(np.int32)
    avail = (hm == hm.max()) & (vm == vm.max()) & (ut == 0)
    not_avail = ~avail
    not_avail = dilate(not_avail, half)          # 建筑 footprint
    cand = ~not_avail
    ax, ay = summ["anchor"]
    ys, xs = np.nonzero(cand)
    if len(xs) == 0:
        return []
    d = np.abs(xs - ax) + np.abs(ys - ay)
    order = np.argsort(d)
    out = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if power_r is not None:
            continue  # 由调用方处理
        out.append((x, y))
        if len(out) >= n:
            break
    return out


def main():
    env = env_new()
    try:
        o = env.reset()[0]
        summ = RE.summarize(o.observation)
        cands = candidates(summ, half=1, n=6)
        print("anchor=", summ["anchor"], "candidates=", cands, flush=True)
        rows = []
        for (tx, ty) in cands:
            # 每试一个都新开一局，避免上一局的 Gateway 影响计数
            e = env_new()
            try:
                oo = e.reset()[0]
                s2 = RE.summarize(oo.observation)
                probes = [i for i, u in enumerate(oo.observation["raw_units"])
                          if int(u["alliance"]) == 1 and int(u["unit_type"]) == 84]
                if not probes:
                    rows.append((tx, ty, "no_probe"))
                    continue
                idx = probes[0]
                tag = int(oo.observation["raw_units"][idx]["tag"])
                fc = FunctionCall(R.Build_Gateway_pt.id, [[0], [idx], [tx, ty]])
                e.step([[fc]])
                seen = 0
                orders = None
                for step_i in range(200):
                    e.step([[FunctionCall(R.no_op.id, [])]])
                    ro = e._obs[0]
                    seen = sum(1 for u in ro.observation.raw_data.units
                               if int(u.alliance) == 1 and int(u.unit_type) == 62)
                    if seen:
                        break
                    if step_i == 4:
                        for u in ro.observation.raw_data.units:
                            if int(u.tag) == tag:
                                orders = [int(x.ability_id) for x in u.orders]
                rows.append((tx, ty, "gateway=%d" % seen, "orders=%s" % orders))
            finally:
                e.close()
            print("  try", rows[-1], flush=True)
        ok = sum(1 for r in rows if len(r) > 2 and str(r[2]).startswith("gateway=")
                 and int(str(r[2]).split("=")[1]) > 0)
        print("\n=== GW success %d/%d ===" % (ok, len(rows)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
