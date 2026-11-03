#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 67 轮：整批真实轨迹 + 完整 `loss_function`，对比 legacy / renorm 的六头 KL。

## 为什么这一步值得做

前面几轮的所有 KL 结论都来自**小合成分布**（§65.4 的手算参照）。
本脚本把 §66 落盘的**一整批真实轨迹**（`AHP.batch_size` 条）直接喂给
`loss_function` —— 也就是训练真正走的那条路径 —— 然后**同一个 batch**、
**同一份权重**下分别用 legacy 与 renorm 算一遍，直接对比六头的 KL。

这正是 Kimi-04 要的「真实轨迹进入完整 loss，而不是只调用
`rl_unroll + get_logprob_and_rhos`」。

## 怎么用

`MAS_KL_MODE` 在 `rl_algo` **import 时**读取，所以两种模式要**分两个进程**跑：

    MAS_KL_MODE=legacy python _test_r67_loss.py log/traj_batch8.pt > log/kl_legacy.log
    MAS_KL_MODE=renorm python _test_r67_loss.py log/traj_batch8.pt > log/kl_renorm.log

## 判据

| 项 | 期望 |
|---|---|
| loss 有限 | 不是 nan/inf |
| renorm 的六个 KL | 全部 ≥ 0 |
| legacy 的六个 KL | **可能有负值**（这就是要修的东西） |
| 两者之差 | 能看出量级差异（尤其 units / target_unit） |
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
from alphastarmini.core.rl import rl_loss as RLOSS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

SIX = ("action_type", "delay", "queue", "units", "target_unit", "target_location")


def build_agent(sd):
    a = AlphaStarAgent(name="r67", race=Race.protoss, initial_weights=None)
    a.agent_nn.model.load_state_dict(sd)
    a.agent_nn.model.eval()
    return a


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = torch.load(sys.argv[1], map_location="cpu")
    trajs = d.get("trajs") or [d["traj"]]
    sd = d["state_dict"]
    print("载入 %s：%d 条轨迹（每条 T=%d），pv=%s，MAS_KL_MODE=%s"
          % (sys.argv[1], len(trajs), len(trajs[0].state),
             d.get("policy_version", "n/a"), RA.KL_MODE), flush=True)

    agent = build_agent(sd)

    with torch.no_grad():
        loss, ld = RLOSS.loss_function(
            agent, trajs, use_opponent_state=False,
            no_replay_learn=True, only_update_baseline=False,
            learner_baseline_weight=1)

    print("\n total loss = %.6f   finite=%s" % (float(loss), bool(torch.isfinite(loss))), flush=True)
    print("\n %-12s %s" % ("KL head", "值"), flush=True)
    neg = 0
    for f in SIX:
        k = "all_kl_loss_" + f + ":"
        if k in ld:
            v = float(ld[k])
            print("   %-12s %+.6e  %s" % (f, v, "**负值**" if v < -1e-12 else ""), flush=True)
            if v < -1e-12:
                neg += 1
        else:
            print("   %-12s （loss_dict 里没有这个键）" % f, flush=True)

    print("\n 其余 key：", flush=True)
    for k in sorted(ld):
        if k.startswith("all_kl_loss_") or k in ("loss_upgo:", "loss_ent:", "loss_kl:"):
            continue
        print("   %-32s %+.6e" % (k, float(ld[k])), flush=True)

    print("\n=== 结论（MAS_KL_MODE=%s）===" % RA.KL_MODE, flush=True)
    print(" loss 有限：%s" % bool(torch.isfinite(loss)), flush=True)
    print(" 负值 KL 个数：%d / %d" % (neg, len(SIX)), flush=True)
    return 0 if (torch.isfinite(loss) and (RA.KL_MODE != "renorm" or neg == 0)) else 2


if __name__ == "__main__":
    sys.exit(main())
