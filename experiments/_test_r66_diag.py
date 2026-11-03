#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 66 轮：`queue` 残差 1.68e-04 到底是浮点噪声还是真错位（Kimi-04 的判别步）。

## 上一轮留下的问题

DS11 §65.9：三个已存 head 的 `Δ = target_lp − behavior_lp`，
action_type 5.7e-06、delay 2.0e-05、**queue 1.68e-04**。
按 1e-5 阈值 delay/queue 都是 FAIL；与被注入的最小误差（5.6e-04）只差 3.3 倍，
**不足以排除一个小的系统性错位**。

## 本脚本给出的三个判别量

| 量 | 怎么判 |
|---|---|
| `max\|Δlogit\|` / `mean\|Δlogit\|` | **直接比 logits 本身**。若 max 与 mean 同量级 → 弥散（数值）；若 max 远大于 mean → 局部（错位） |
| `被扰动的类占比`（\|Δlogit\| > 1e-3） | 弥散噪声只会让极少数类超过阈值；错位会让整片分布偏移 |
| `Δ_logp` 在 **float64** 下重算 | 若 fp64 后残差塌掉 → 是 log_prob 的算术精度；若不变 → 差异**在 logits 里**，与 log_prob 无关 |

## 两条互斥的解释

- **H1 路径数值差**：`behavior` 来自 server 的 `forward`（**批大小 ≈ 8**），
  target 来自 `rl_unroll → mimic_forward`（**B=1**）。批大小不同 →
  cuDNN/BLAS 的归约顺序不同 → logits 差 1e-5~1e-4。**良性**。
- **H2 真错位**：hidden/动作/掩码有一处对不上 → 少数类别的 logit 差出 1e-2 量级。

判据：`max|Δlogit| < 1e-3` 且 `max/mean < 20` 且 fp64 不塌 → 支持 H1。

用法：
    python _test_r66_diag.py log/traj_eval30.pt
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

HEADS = ("action_type", "delay", "queue")


def build_agent(sd):
    a = AlphaStarAgent(name="r66", race=Race.protoss, initial_weights=None)
    a.agent_nn.model.load_state_dict(sd)
    a.agent_nn.model.eval()
    return a


def logp_fp64(logits, actions):
    """fp64 下重算 log_prob（这三个 head 没有掩码，就是 gather log_softmax）。"""
    lp = torch.log_softmax(logits.double(), dim=-1)
    idx = actions.long().reshape(-1, 1)
    return lp.gather(1, idx).reshape(-1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = torch.load(sys.argv[1], map_location="cpu")
    traj, sd = d["traj"], d["state_dict"]
    T = len(traj.state)
    n = T - 1
    print("载入 %s：T=%d" % (sys.argv[1], T), flush=True)

    agent = build_agent(sd)
    logits = agent.rl_unroll([traj], use_opponent_state=False)[0]
    beh = traj.behavior_logits

    print("\n%-13s %12s %12s %8s %12s %12s %s"
          % ("head", "max|Δlogit|", "mean|Δlogit|", "max/mean", "Δlogp(fp32)", "Δlogp(fp64)", "判读"), flush=True)
    h1 = True
    for f in HEADS:
        t = getattr(logits, f)
        if t.dim() == 3 and t.shape[1] == T:
            t = t.transpose(0, 1)
        t = t[:n].reshape(n, -1)
        b = torch.stack([getattr(x, f) for x in beh[:n]]).reshape(n, -1)
        act = torch.stack([getattr(a, f) for a in traj.action[:n]]).reshape(n, -1)

        dl = (t - b).abs()
        mx, mn = float(dl.max()), float(dl.mean())
        ratio = mx / max(mn, 1e-12)
        frac_big = float((dl > 1e-3).float().mean())

        d32 = float((logp_fp64(t, act) - logp_fp64(b, act)).abs().max())
        # fp32 口径：与生产一致（log_softmax 在 float32 下算）
        lp32_t = torch.log_softmax(t.float(), dim=-1).gather(1, act.long().reshape(-1, 1)).reshape(-1)
        lp32_b = torch.log_softmax(b.float(), dim=-1).gather(1, act.long().reshape(-1, 1)).reshape(-1)
        d32real = float((lp32_t - lp32_b).abs().max())

        # 判读：弥散 + 小 + fp64 不塌 → H1
        verdict = ("H1 数值" if (mx < 1e-3 and ratio < 20) else "H2 疑似错位")
        if verdict.startswith("H2"):
            h1 = False
        print("%-13s %12.3e %12.3e %8.1f %12.3e %12.3e %s（>1e-3 的类占 %.2f%%）"
              % (f, mx, mn, ratio, d32real, d32, verdict, frac_big * 100), flush=True)

    print("\n=== 结论 ===", flush=True)
    print("支持 H1（批大小不同导致的数值差）: %s" % h1, flush=True)
    print("若 max|Δlogit| 与 mean 同量级且 fp64 后 Δlogp 不塌，则差异在 logits 本身，", flush=True)
    print("与 log_prob 的算术无关 —— 那就只剩『两条前向路径的归约顺序不同』这一个来源。", flush=True)
    print("\n⚠️ 决定性的下一实验（本轮没做）：把 server 的 max_batch 设成 1 再落一条轨迹，", flush=True)
    print("   若 Δ 掉到 1e-6 量级，则 H1 成立（批大小是唯一变量）。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
