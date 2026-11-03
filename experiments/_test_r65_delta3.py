#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 65 轮：真实轨迹上**三个已存 head** 的未截断 log-prob 差（Kimi-04 的可行子集）。

## 为什么不用 `rl_loss.get_logprob_and_rhos`

上一轮试了四次都失败，根因是它把 `AHP.sequence_length × AHP.batch_size` 写死，
只对"一整个 learner batch"成立（见 DS11 §64.13）。本轮绕开它，
直接用底层的 `RA.log_prob` 对齐两侧 —— 只依赖轨迹本身，不依赖任何全局常量。

## 能覆盖与不能覆盖

- ✅ `action_type / delay / queue`：这三个 head 的 behavior_logits 一定有值
  （`_store_logits` 写了 6 个头，但只有这 3 个是"存进 Trajectory 供 rho 用"的）；
- ❌ `units / target_unit / target_location`：它们的行为侧 log_prob 需要掩码语义，
  而掩码的组装依赖 `get_useful_masks` 里的 AHP 常量 —— 这条留到下一轮，
  **不假装覆盖**。

## 判据

| 判据 | 期望 |
|---|---|
| [B] 同权重下 Δ = target_lp − behavior_lp | ≈ 0（三个 head 都要） |
| [C] 把 behavior_logits 沿时间轴错一位 | Δ 必须显著变大（否则观测量对错位不敏感） |
| [D] 权重 ×1.001 | Δ 必须偏离 0 |

用法：
    python _test_r65_delta3.py log/traj_eval30.pt
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
from alphastarmini.core.rl import rl_algo as RA
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

HEADS = ("action_type", "delay", "queue")


def traj_T(traj):
    return len(traj.state)


def build_agent(sd):
    a = AlphaStarAgent(name="r65d", race=Race.protoss, initial_weights=None)
    a.agent_nn.model.load_state_dict(sd)
    a.agent_nn.model.eval()          # 必须 eval：BN/Dropout 会重采样
    return a


def two_sided(agent, traj, behavior, n_steps):
    """返回 {head: (target_lp, behavior_lp)}，形状 (T-1,)。

    只取前 T-1 步：与 `get_logprob_and_rhos` 的 `[:-1]` 口径一致
    （最后一步的 state 没有"下一步动作"可对）。
    """
    logits = agent.rl_unroll([traj], use_opponent_state=False)[0]
    out = {}
    for f in HEADS:
        lg = getattr(logits, f)
        # ⚠️ `rl_unroll` 单条轨迹返回的是 **[B, T, V]**（B=1），
        #    而 `rl_loss` 里会先 `transpose_target_logits` 成 [T, B, V]。
        #    第一版直接按 [T, ...][:-1] 切，切到了 batch 维（size 1），
        #    报 `shape '[7, -1]' is invalid for input of size 4512`（=1×8×564）。
        if lg.dim() == 3 and lg.shape[1] == traj_T(traj):
            lg = lg.transpose(0, 1)          # [B,T,V] -> [T,B,V]
        t_lg = lg[:n_steps].reshape(n_steps, -1)
        b_lg = torch.stack([getattr(b, f) for b in behavior[:n_steps]]).reshape(n_steps, -1)
        act = torch.stack([getattr(a, f) for a in traj.action[:n_steps]]).reshape(n_steps, -1)
        t_lp = RA.log_prob(t_lg, act, [None, None, None], AHP.max_selected + 1)
        b_lp = RA.log_prob(b_lg, act, [None, None, None], AHP.max_selected + 1)
        out[f] = (t_lp.detach().reshape(-1), b_lp.detach().reshape(-1))
    return out


def stats(pair):
    d = (pair[0] - pair[1]).float()
    return dict(n=int(d.numel()), mad=float(d.abs().mean()), mx=float(d.abs().max()),
                lo=float(d.min()), hi=float(d.max()))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = torch.load(sys.argv[1], map_location="cpu")
    traj, sd = d["traj"], d["state_dict"]
    T = len(traj.state)
    n = T - 1
    print("载入 %s：T=%d，版本=%s，actor=%s"
          % (sys.argv[1], T, d.get("policy_version", "n/a"), d.get("actor", "n/a")), flush=True)

    agent = build_agent(sd)
    beh = traj.behavior_logits            # T 个 ArgsActionLogits 组成的 tuple

    print("\n=== [B] 同权重下三个 head 的 Δ = target_lp − behavior_lp ===", flush=True)
    base = {f: stats(p) for f, p in two_sided(agent, traj, beh, n).items()}
    ok_b = True
    for f in HEADS:
        v = base[f]
        good = v["mx"] < 1e-5
        print("  %-12s n=%-2d |Δ|mean=%.3e max=%.3e  rho_unclipped∈[%.6f, %.6f]  %s"
              % (f, v["n"], v["mad"], v["mx"],
                 float(torch.exp(torch.tensor(v["lo"]))), float(torch.exp(torch.tensor(v["hi"]))),
                 "PASS" if good else "FAIL"), flush=True)
        ok_b = ok_b and good

    print("\n=== [C] 变异：behavior_logits 沿时间轴错一位 ===", flush=True)
    shifted = tuple(beh[(i + 1) % len(beh)] for i in range(len(beh)))
    pert = {f: stats(p) for f, p in two_sided(agent, traj, shifted, n).items()}
    ok_c = True
    for f in HEADS:
        bigger = pert[f]["mad"] > 10 * max(base[f]["mad"], 1e-12)
        print("  %-12s |Δ|mean %.3e -> %.3e  %s"
              % (f, base[f]["mad"], pert[f]["mad"], "PASS（被检出）" if bigger else "FAIL（不敏感）"),
              flush=True)
        ok_c = ok_c and bigger

    print("\n=== [D] 权重 ×1.001 后 Δ 必须偏离 0 ===", flush=True)
    with torch.no_grad():
        for p in agent.agent_nn.model.parameters():
            p.mul_(1.001)
    wt = {f: stats(p) for f, p in two_sided(agent, traj, beh, n).items()}
    ok_d = True
    for f in HEADS:
        ch = wt[f]["mad"] > 1e-9
        print("  %-12s |Δ|mean %.3e -> %.3e  %s"
              % (f, base[f]["mad"], wt[f]["mad"], "PASS" if ch else "FAIL（没变）"), flush=True)
        ok_d = ok_d and ch

    print("\n=== 结论 ===", flush=True)
    print("[B] 同权重 Δ≈0：%s" % ok_b, flush=True)
    print("[C] 时间错位可检出：%s" % ok_c, flush=True)
    print("[D] 权重扰动可检出：%s" % ok_d, flush=True)
    print("（units/target_unit/target_location 本轮**未覆盖**，需要 AHP.batch_size 条轨迹）", flush=True)
    return 0 if (ok_b and ok_c and ok_d) else 2


if __name__ == "__main__":
    sys.exit(main())
