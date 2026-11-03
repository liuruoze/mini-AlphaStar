#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 58 轮：**真实轨迹**上的 rho 分布（DeepSeek）。

## 这一步补的是什么

DS10 §16.1 的诚实清单里有一条：「真实 rho 分布 —— 本轮只跑到 forward，
第 8 轮的 rho=1.0 是单步、非完整轨迹」。

本脚本用的输入不是手工拼的 tensor，而是 `rl_vs_inner_bot_mp.py` 在**真实训练循环**里
落盘的一条 16 步轨迹（`MAS_DUMP_TRAJ`），连带**当时的权重**。所以它测的是：

> batch 路径存进 `Trajectory.behavior_logits` 的数，能否被
> `rl_unroll -> mimic_forward` 的重算值对上（即 on-policy 的 rho 是否为 1）。

## 四条判据

| 判据 | 期望 | 不成立说明什么 |
|---|---|---|
| [A] `rl_unroll` 能吃这条轨迹 | 不报错，形状 `[T-1, 1, ...]` | 轨迹容器/字段不匹配 |
| [B] 同权重下 rho | **恒等于 1**（<1e-6） | 链路错位：hidden/state/action 有一处对不上 |
| [C] 扰动权重后 rho | 必须偏离 1（且 ≤1） | 若仍恒为 1，说明 [B] 的"全 1"是假的（比如两边用了同一份 logits） |
| [D] 逐字段统计 | action_type/delay/queue 各自给分布 | 哪一路错位能定位到具体 head |

用法：
    python _test_r58_real_rho.py log/traj_batch.pt [cpu|cuda:1]
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from pysc2.env.sc2_env import Race

from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
from alphastarmini.core.rl import rl_loss as RLOSS

FIELDS = ("action_type", "delay", "queue")


def build_agent(state_dict, device):
    a = AlphaStarAgent(name="r58", race=Race.protoss, initial_weights=None)
    a.agent_nn.model.load_state_dict(state_dict)
    a.agent_nn.model.to(device)
    a.agent_nn.model.eval()          # 必须 eval：dropout 会重采样，rho 必然乱
    return a


def rho_stats(traj, agent):
    """返回 {field: (n, min, max, mean, frac_eq_1)}。"""
    out = {}
    logits_all = agent.rl_unroll([traj], use_opponent_state=False)
    for f in FIELDS:
        # mask_provided 只有 units/target_unit/target_location 三路会用到，
        # 这三个 head 传 [None, None, None] 即可（函数内部 mask_used 保持 None）。
        tlp, rho, _ = RLOSS.get_logprob_and_rhos(logits_all, f, traj, [None, None, None])
        r = rho.detach().float().reshape(-1)
        out[f] = (int(r.numel()), float(r.min()), float(r.max()), float(r.mean()),
                  float((r == 1).float().mean()))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")

    d = torch.load(path, map_location="cpu")
    traj, sd = d["traj"], d["state_dict"]
    T = len(traj.state)
    print("载入 %s：T=%d 步，%d 个参数张量，device=%s"
          % (path, T, len(sd), device), flush=True)

    agent = build_agent(sd, device)

    print("\n=== [A] rl_unroll 能否消费这条真实轨迹 ===", flush=True)
    logits_all = agent.rl_unroll([traj], use_opponent_state=False)
    ok_shape = True
    for f in FIELDS:
        lg = getattr(logits_all, f)
        print("  %-12s target_logits %s" % (f, tuple(lg.shape)), flush=True)
        if lg.shape[0] != T - 1 or lg.shape[1] != 1:
            ok_shape = False
    print("  %s：形状应为 [T-1, 1, ...]（rl_unroll 用前 T-1 步的 state 预测后一步）"
          % ("PASS" if ok_shape else "FAIL"), flush=True)

    print("\n=== [B] 同权重（= 落盘时的权重）-> rho 必须恒为 1 ===", flush=True)
    base = rho_stats(traj, agent)
    ok_b = True
    for f, (n, mn, mx, mean, frac) in base.items():
        good = abs(mn - 1.0) < 1e-6 and abs(mx - 1.0) < 1e-6
        print("  %-12s n=%-4d rho∈[%.9f, %.9f] mean=%.9f  %%==1: %.1f%%  %s"
              % (f, n, mn, mx, mean, frac * 100, "PASS" if good else "FAIL"), flush=True)
        ok_b = ok_b and good

    print("\n=== [C] 扰动权重（×1.001）-> rho 必须偏离 1，且全部 <= 1 ===", flush=True)
    with torch.no_grad():
        for p in agent.agent_nn.model.parameters():
            p.mul_(1.001)
    pert = rho_stats(traj, agent)
    ok_c = True
    for f, (n, mn, mx, mean, frac) in pert.items():
        changed = abs(mean - 1.0) > 1e-7
        le1 = mx <= 1.0 + 1e-7
        print("  %-12s rho∈[%.9f, %.9f] mean=%.9f（偏离 %s）<=1:%s  %s"
              % (f, mn, mx, mean, "是" if changed else "**否**",
                 "是" if le1 else "**否**",
                 "PASS" if (changed and le1) else "FAIL"), flush=True)
        ok_c = ok_c and changed and le1

    print("\n=== 结论 ===", flush=True)
    print("[A] 轨迹可消费: %s" % ok_shape, flush=True)
    print("[B] 真实轨迹 rho == 1: %s" % ok_b, flush=True)
    print("[C] rho 对权重敏感且被 clamp: %s" % ok_c, flush=True)
    return 0 if (ok_shape and ok_b and ok_c) else 2


if __name__ == "__main__":
    sys.exit(main())
