# -*- coding: utf-8 -*-
"""第 75 轮（GLM，Kimi-08）：mask 语义审查 —— 可证伪的部分。

**审计发现（读代码得出，测试证实）**

1. `selected_units_head` 对 logits **预先**填 -1e9（:191/:277）——
   所以采样时 softmax 只在合法集合上归一化，这部分是对的。

2. 但 `rl_algo.log_prob` 的 mask 是**乘在 log_softmax 之后**：
   ```
   x = -one_hot(actions) * log_softmax(logits)
   x = x * entity_mask            # ← 事后乘
   loss = -sum(x)
   ```
   后果：**如果被选中的动作恰好被 mask**，它的贡献变成 0，
   于是 `log_prob = -0 = 0`，即**概率 1** —— 非法动作拿到最高似然，
   而不是 -inf。方向完全反了。
   （VTrace 里 `rho = exp(lp_t - lp_b)`，两侧同函数 → rho=1，静默中和。）

3. `get_useful_masks` 的不对称：selected 多一个槽位（`+1`，"不选"），entity **没有** +1。
   这就是历史"12→13 列"问题的来源，必须显式写出来而不是靠记忆。

**为什么不直接修**：`log_prob` 是 VTrace/loss 的路径，改语义（softmax 前重归一化）
会改变所有历史数字；且生产中 actor 只采合法动作，只有"采样侧与 loss 侧 mask 不一致"
时才会触发。本轮把它**证实 + 定级（潜伏）**，修复方案（softmax 前重归一化 /
保证两侧 mask 同源）写进 ERRATA B17，与 Kimi-03 的消融一起做。

跑法：python experiments/_test_r74_mask.py
"""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphastarmini.core.rl import rl_algo as RA
from alphastarmini.core.rl.rl_loss import get_useful_masks
# ⚠️ 必须与 rl_loss 用**同一个** AHP：仓库里有 Arch_Hyper_Parameters 与
#    AlphaStar_Arch_Hyper_Parameters 两个类，batch/seq 值不同（64 vs 8）。
#    第一版导错对象，get_useful_masks 的行数对不上直接 shape 报错。
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP


def main():
    res = [True]

    def chk(name, cond, detail):
        res[0] = res[0] and bool(cond)
        print("  %-50s %s   %s" % (name, "PASS" if cond else "FAIL", detail), flush=True)

    print("=== [A] 预 mask 的 logits：合法 vs 非法动作的 log_prob ===", flush=True)
    V = 6
    lg = torch.tensor([[0.1, 0.2, 0.3, -1e9, 0.5, 0.6]])      # 类别 3 非法
    ent = torch.ones(1, V)
    ent[0, 3] = 0.0                                            # loss 侧同样 mask 它
    lp_legal = RA.log_prob(lg, torch.tensor([[1]]), [None, ent, None], AHP.max_selected + 1)
    lp_illegal = RA.log_prob(lg, torch.tensor([[3]]), [None, ent, None], AHP.max_selected + 1)
    # ⚠️ 参照值 = log_softmax 本身（log-prob 是负数）。
    #    第一版写成了 -log_softmax（那是交叉熵，正数），把正确实现判成 FAIL。
    ref_legal = torch.log_softmax(lg, dim=-1)[0, 1]
    chk("合法动作的 log_prob == log_softmax（数值一致）",
        torch.allclose(lp_legal, ref_legal, atol=1e-5),
        "lp=%.4f ref=%.4f" % (float(lp_legal), float(ref_legal)))
    chk("非法动作被选中时 log_prob == 0（== 概率 1，病理确认）",
        float(lp_illegal) == 0.0, "lp=%.4g（正确语义应为 -inf）" % float(lp_illegal))

    print("\n=== [B] 预 mask 的 logits：合法集合上的概率质量 ===", flush=True)
    p = torch.softmax(lg, dim=-1)
    chk("非法类概率 ≈ 0", float(p[0, 3]) < 1e-12, "p=%.3g" % float(p[0, 3]))
    chk("合法集合质量 == 1", abs(float(p[0].sum()) - 1.0) < 1e-6,
        "%.9f" % float(p[0].sum()))

    print("\n=== [C] 无预 mask、只有事后乘：病理一样存在 ===", flush=True)
    lg_raw = torch.tensor([[0.1, 0.2, 0.3, 0.9, 0.5, 0.6]])   # 类别 3 没被预 mask
    lp_raw = RA.log_prob(lg_raw, torch.tensor([[3]]), [None, ent, None], AHP.max_selected + 1)
    chk("事后乘 mask 同样把非法动作变成 log_prob=0", float(lp_raw) == 0.0,
        "lp=%.4g" % float(lp_raw))

    print("\n=== [D] get_useful_masks 的 +1 不对称与边界 ===", flush=True)
    # ⚠️ get_useful_masks 把行数硬编码为 AHP.sequence_length × AHP.batch_size（=512），
    #    传入任意其它行数会直接 shape 报错（这正是 Kimi-04 说"离线复算困难"的原因之一）。
    #    所以按它的口径造 512 行：第 0 行 = select0/entity0，第 1 行 = select2/entity3。
    _n = AHP.sequence_length * AHP.batch_size
    sun = torch.zeros(_n, 1); sun[1, 0] = 2.0
    en = torch.zeros(_n, 1);  en[1, 0] = 3.0
    sel, entm = get_useful_masks(sun, en, torch.device("cpu"))
    sel = sel.reshape(AHP.sequence_length, AHP.batch_size, -1)
    entm = entm.reshape(AHP.sequence_length, AHP.batch_size, -1)
    s00, s01 = sel[0, 0], sel[0, 1]
    e00, e01 = entm[0, 0], entm[0, 1]
    chk("selected 行宽 == max_selected+1（多一个'不选'槽）",
        sel.shape[-1] == AHP.max_selected + 1, "%d" % sel.shape[-1])
    chk("entity 行宽 == max_entities（无 +1）",
        entm.shape[-1] == AHP.max_entities, "%d" % entm.shape[-1])
    chk("select=0 时 selected 有 1 个 True（'不选'槽）",
        int(s00.sum()) == 1, "sum=%d" % int(s00.sum()))
    chk("entity=0 时 entity 全 False（空支撑）", int(e00.sum()) == 0,
        "sum=%d" % int(e00.sum()))
    chk("select=2 -> 3 个 True；entity=3 -> 3 个 True",
        int(s01.sum()) == 3 and int(e01.sum()) == 3,
        "sel=%d ent=%d" % (int(s01.sum()), int(e01.sum())))

    print("\n=== 结论 ===", flush=True)
    if res[0]:
        print(" 审计完成：采样侧（预 mask）正确；loss 侧（事后乘）存在'非法动作=概率1'的病理，\n"
              " 已定级为潜伏问题（见 ERRATA B17），修复需与 Kimi-03 消融一起做。", flush=True)
        return 0
    print(" 审计的预期本身没成立 —— 重新读代码。", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
