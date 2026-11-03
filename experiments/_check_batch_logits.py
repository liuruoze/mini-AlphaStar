"""Kimi 补充实测 [2]：批推理的数值一致性 —— 所有计划共同依赖、却从未验证的前提。

三份计划都写了「比 logits 不比动作」，但都没指出**该用哪个接口比**。
这里有个坑：
  action_logits_by_state -> model.forward(return_logits=True) 是**自回归采样**，
    action_type 采样结果会进入 delay/queue/units/target 的 autoregressive_embedding。
    batch 内采样序列不同 -> 后续 head 的 logits **必然不同**，即使实现完全正确。
    所以拿它做全量 logits 比对会**误报 bug**。

正确的三层校验：
  L2（本脚本实现）：forward 只比 **action_type_logits**（采样前的第一个 head）。
      它已覆盖 三编码器 + LSTM + available_actions mask + action_type head，
      能抓住 batch_size 传错 / hidden 沿错维度拼 / mask 错位 这三类致命 bug。
  L1（更严，需 action_gt）：用 mimic_forward（teacher forcing）比全部 6 个 head。
  L3（统计）：多次采样比动作分布，验证自回归链路行为等价。

本脚本做 L2，并额外校验 LSTM 输出 hidden 是否逐样本对齐。
通过标准：max|Δ| < 1e-3
"""
import os
import sys
import glob

sys.path.insert(0, os.getcwd())
import torch

import param as P
P.on_server = True

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl.feature import Feature


def main():
    torch.manual_seed(1234)
    dev = torch.device("cuda:0")
    net = ArchModel().to(dev).eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    batch_feat = feats[idx].float().to(dev)

    # 每个样本**不同**的非零 hidden，才能验出 hidden 是否被正确按样本使用
    h = torch.randn(1, N, 128, device=dev)
    c = torch.randn(1, N, 128, device=dev)

    singles_at, singles_h, singles_c = [], [], []
    with torch.no_grad():
        for i in range(N):
            st = Feature.feature2state(batch_feat[i:i + 1])
            hi = (h[:, i:i + 1].contiguous(), c[:, i:i + 1].contiguous())
            out = net.forward(st, batch_size=1, sequence_length=1, hidden_state=hi,
                              return_logits=True, obs_list=None)
            singles_at.append(out[0].action_type.detach().reshape(-1).clone())
            nh = out[2]
            singles_h.append(nh[0].detach().reshape(-1).clone())
            singles_c.append(nh[1].detach().reshape(-1).clone())

        st = Feature.feature2state(batch_feat)
        out = net.forward(st, batch_size=N, sequence_length=1, hidden_state=(h, c),
                          return_logits=True, obs_list=None)
        batch_at = out[0].action_type.detach()
        nh = out[2]

    print("action_type_logits shape (batch):", tuple(batch_at.shape))
    print("hidden shape:", tuple(nh[0].shape))
    print("%-4s %-14s %-14s %-14s" % ("i", "max|dLogit|", "max|dHidden_h|", "max|dHidden_c|"))
    worst = 0.0
    for i in range(N):
        d1 = (singles_at[i] - batch_at[i].reshape(-1)).abs().max().item()
        d2 = (singles_h[i] - nh[0][:, i].reshape(-1)).abs().max().item()
        d3 = (singles_c[i] - nh[1][:, i].reshape(-1)).abs().max().item()
        worst = max(worst, d1, d2, d3)
        print("%-4d %-14.3e %-14.3e %-14.3e" % (i, d1, d2, d3))

    print("\n==> 最大偏差 %.3e  %s" % (worst, "PASS (<1e-3)" if worst < 1e-3 else "FAIL"))


if __name__ == "__main__":
    main()
