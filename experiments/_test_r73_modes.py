# -*- coding: utf-8 -*-
"""第 73 轮（GLM，Kimi-06）：模型模式矩阵 —— actor(local/batch) x learner(train/eval) x bs(1/N)。

**要证明什么**

Kimi-06 要求"打印实际生效的 dropout 概率、BN 模式及 running stats 前后变化，再归因"，
并建立模式矩阵。这条测试就是那个矩阵的可执行版：

| 模式    | BN running_mean | 同输入两次 forward |
|---------|-----------------|--------------------|
| eval    | **不变**         | **逐位相同**        |
| train   | **被改写**       | 不同（dropout 重采样）|

由此推出两条工程约束（这正是第 64 轮修复的依据）：
- **eval 下没有跨 actor 的 BN 串扰**：running stats 不被采样过程改写，
  行为策略只是权重的函数 —— 这是 `behavior_logits` 有意义的前提；
- **train 下不能用 batch=1 采样**：BN 用单样本统计量，等于"自己归一化自己"。

跑法：python experiments/_test_r73_modes.py
"""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class TinyBNNet(nn.Module):
    """带 BatchNorm2d + Dropout 的最小网络，结构上与真实模型同构（BN+Dropout 组合）。"""

    def __init__(self):
        super(TinyBNNet, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.do = nn.Dropout(p=0.5)
        self.out = nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x):
        return self.out(self.do(self.bn(torch.relu(self.conv(x)))))


def stats(model):
    return float(model.bn.running_mean.abs().mean()), \
           float(model.bn.running_var.abs().mean())


def main():
    res = [True]

    def chk(name, cond, detail):
        res[0] = res[0] and bool(cond)
        print("  %-46s %s   %s" % (name, "PASS" if cond else "FAIL", detail), flush=True)

    net = TinyBNNet()
    x1 = torch.randn(1, 3, 16, 16)
    x3 = torch.randn(3, 3, 16, 16)

    print("=== [A] eval 模式：running stats 不变，输出确定 ===", flush=True)
    net.eval()
    rm0, rv0 = stats(net)
    with torch.no_grad():
        o1 = net(x1)
        o2 = net(x1)
    rm1, rv1 = stats(net)
    chk("eval: running_mean 不被 forward 改写", rm0 == rm1 and rv0 == rv1,
        "%.4f -> %.4f" % (rm0, rm1))
    chk("eval: 同输入两次输出逐位相同（无 dropout 噪声）",
        torch.equal(o1, o2), "max|Δ|=%.3g" % float((o1 - o2).abs().max()))
    chk("eval: dropout 层仍在（p=0.5）", float(net.do.p) == 0.5, "p=%.2f" % net.do.p)

    print("\n=== [B] train 模式 + batch=1：BN 用单样本统计量 ===", flush=True)
    net.train()
    rm0, rv0 = stats(net)
    with torch.no_grad():
        o1 = net(x1)
    rm1, rv1 = stats(net)
    chk("train: running_mean 被 forward 改写", (rm0, rv0) != (rm1, rv1),
        "%.4f -> %.4f" % (rm0, rm1))
    with torch.no_grad():
        o2 = net(x1)
    chk("train: 同输入两次输出不同（dropout）", not torch.equal(o1, o2),
        "max|Δ|=%.3g" % float((o1 - o2).abs().max()))

    print("\n=== [C] train 模式 + batch=N：batch 统计量随同伴变化（串扰）===", flush=True)
    net.train()
    with torch.no_grad():
        a = net(torch.cat([x1, x1, x1], dim=0))[0:1]     # 同一个样本，但和"别人"同批
        b = net(torch.cat([x1, x3, x3], dim=0))[0:1]     # 同一个样本，换了同伴
    chk("train: 同一样本的输出随同批 actor 改变（BN 串扰）",
        not torch.allclose(a, b, atol=1e-5),
        "max|Δ|=%.3g" % float((a - b).abs().max()))

    print("\n=== [D] eval 模式 + batch=N：不随同伴变化（隔离）===", flush=True)
    net.eval()
    with torch.no_grad():
        a = net(torch.cat([x1, x1, x1], dim=0))[0:1]
        b = net(torch.cat([x1, x3, x3], dim=0))[0:1]
    chk("eval: 同一样本输出与同批 actor 无关", torch.allclose(a, b, atol=1e-6),
        "max|Δ|=%.3g" % float((a - b).abs().max()))

    print("\n=== 结论（模式矩阵）===", flush=True)
    print("  eval  x bs=1/N : 输出确定、BN 不串扰   -> acting 的唯一安全模式", flush=True)
    print("  train x bs=1   : 单样本统计量 + dropout -> 采样分布不是策略", flush=True)
    print("  train x bs=N   : 输出随同伴变化        -> 跨 actor 串扰", flush=True)
    if res[0]:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
