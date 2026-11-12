"""HierNet macro → 本项目 raw 动作（v3 执行层）的适配层。

两条硬约束（本夜源码级核实）：
  1. pysc2 3.0 `Features.reverse_raw_action` 只处理 `action_raw`；HierNet 用
     `*_screen` feature-layer 动作，直接转 replay 会全变 no_op ⇒ 专家动作必须
     以 **raw** 形式落地（这里输出的就是 v3 的 raw action dict）。
  2. HierNet `get_input` 的 41 维含 `assigned_harvesters` / `orders[].ability_id`
     等 raw-proto 独有字段，我们的 `.pt` 编码 state 里没有 ⇒ 这里从 pysc2 3.0
     实时 obs 构造**近似** 41 维，缺失项按 J143 训练时实测的「恒 0」填充
     （`tech_cost[5:9]` 整局为 0，填 0 反而忠实于训练分布）。

macro 索引（multi_agent.py:205/237）：
    tech 0 pylon / 1 gas / 2 gateway / 3 cyber
    pop  0 probe / 1 zealot / 2 stalker
这些名字与 v3 的 ACT key 完全一致，可直接复用 v3 的选址/选兵执行层。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rule_expert_v3 as RE  # noqa: E402

TECH_COST_BASE = np.array([100.0, 75.0, 150.0, 150.0], dtype=np.float32)


def build_features(summ, obs, difficulty=7, game_loop=0):
    """从 v3 summarize + 原始 obs 构造 HierNet 的 41 维输入（近似）。

    41 = high_input(20) + tech_cost(9) + pop_num(12)
    controller 用 high20；base/battle 用 41；tech 用 29；pop 用 32。
    """
    hi = np.zeros(20, dtype=np.float32)
    tc = np.zeros(9, dtype=np.float32)
    pn = np.zeros(12, dtype=np.float32)

    n_probe = float(summ.get("n_probe", 0))
    n_army = float(summ.get("n_army", 0))
    food_army = 2.0 * n_army                      # 地面兵均 2 人口（近似）
    food_workers = max(n_probe, 1.0)

    hi[0] = float(difficulty)
    hi[1] = float(game_loop)
    hi[2] = float(summ.get("minerals", 0))
    hi[3] = float(summ.get("vespene", 0))
    hi[4] = RE._score(obs, 11)                   # spent_minerals
    hi[5] = RE._score(obs, 12)                   # spent_vespene
    hi[6] = float(summ.get("food_cap", 0))
    hi[7] = float(summ.get("food_used", 0))
    hi[8] = food_army
    hi[9] = n_probe
    hi[10] = n_army
    hi[11] = n_probe
    hi[12] = float(summ.get("n_zealot", 0))
    hi[13] = float(summ.get("n_stalker", 0))
    hi[14] = float(summ.get("n_pylon", 0))
    hi[15] = float(summ.get("n_assim", 0))
    hi[16] = float(summ.get("n_gateway", 0)) + float(summ.get("n_warpgate", 0))
    hi[17] = float(summ.get("n_cyber", 0))
    # 敌方单位数（v3 summarize 的 raw 里带 alliance）
    hi[18] = float(sum(1 for u in summ.get("raw", []) if u.get("alliance") == 4))
    hi[19] = 0.0                                 # 正在攻击的兵数，训练时难精确

    tc[:4] = TECH_COST_BASE
    tc[4] = max(0.0, float(summ.get("food_cap", 0)) - float(summ.get("food_used", 0)))
    tc[5:9] = 0.0                                # J143：整局恒 0（在建数量特征失效）

    n_nexus = max(1, int(summ.get("n_nexus", 1)))
    pn[0] = min(n_probe, 16.0 * n_nexus)         # assigned_harvesters 近似
    pn[1] = 16.0 * n_nexus                       # ideal_harvesters 近似
    pn[2:6] = 0.0                                # 气矿采集数（无对应字段）
    pn[6] = food_army / food_workers
    assim = summ.get("assimilators") or []
    pn[7] = 1.0 if len(assim) >= 1 else 0.0
    pn[8] = 1.0 if len(assim) >= 2 else 0.0
    pn[9:12] = 0.0                               # 训练队列数（无对应字段）

    return np.concatenate([hi, tc, pn]).astype(np.float32)


def macro_to_action(macro, summ, tick, clamp_pylon=8):
    """HierNet macro → v3 decide 契约的 act dict；不满足条件时返回 None。"""
    if macro not in RE.ACT:
        return None

    # Pylon 钳制：HierNet 自身有 Pylon 循环（J143 实测涨到 21，训练目标 8）
    if macro == "build_pylon" and clamp_pylon > 0 \
            and summ.get("n_pylon", 0) >= clamp_pylon:
        return None

    def mk(name, kind, hint=None, place=None, target_tag=None):
        return {"name": name, "id": RE.ACT[name], "raw_name": RE.ACT_RAW[name],
                "kind": kind, "unit_type_hint": hint,
                "place": place, "target_tag": target_tag}

    if macro.startswith("build_"):
        if not RE._can(summ, macro):
            return None
        if macro == "build_assimilator":
            gey = summ.get("geysers") or []
            if not gey:
                return None
            return mk(macro, "build", RE.PROBE, None, int(gey[0][2]))
        loc = RE._pick(summ, tick, need_power=(macro == "build_pylon"), name=macro)
        if loc is None:
            return None
        return mk(macro, "build", RE.PROBE, loc)

    if macro.startswith("train_"):
        if not RE._can(summ, macro):
            return None
        hint = RE.NEXUS if macro == "train_probe" else RE.GATEWAY
        return mk(macro, "train", hint)

    if macro == "attack":
        if summ.get("n_army", 0) < 3:
            return None
        tgt = summ.get("enemy_near_pos") or summ.get("enemy_center") or (45, 47)
        return mk("attack", "attack", None, (int(tgt[0]), int(tgt[1])))

    return None


def selftest():
    # 用 v3 的 fake obs 走一遍，验证 41 维与 macro→action 不抛异常
    units = [(RE.NEXUS, 19, 23, 1.0, 0, 1)] + \
            [(RE.PROBE, 17 + i % 3, 21 + i // 3, 1.0, 0, 1) for i in range(12)]
    obs = RE._fake_obs(20, 400, 0, units)
    summ = RE.summarize(obs)
    f = build_features(summ, obs, difficulty=7, game_loop=1234)
    print("[hier_expert] feat41 shape=%s sum=%.1f first8=%s"
          % (f.shape, float(f.sum()), f[:8].tolist()))
    assert f.shape == (41,), "feat dim"
    for m in ("build_pylon", "build_gateway", "build_cyber",
              "train_probe", "train_zealot", "train_stalker", "attack"):
        a = macro_to_action(m, summ, tick=0)
        print("[hier_expert] %-18s -> %s" % (m, a["name"] if a else None))
    print("[hier_expert] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
