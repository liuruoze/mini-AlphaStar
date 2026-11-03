"""Kimi 诊断：L1 校验失败的根因定位。

已知（_check_batch_mimic.py）：
  entity_nums 一致、baseline 一致（-> lstm_output 一致）
  action_type/delay/queue/target_location 偏差 0.2~2.6
  units/target_unit 偏差 1e9（mask 位置不同）
而 L2（forward 的 action_type_logits + hidden）是 0 偏差。

本脚本绕开整个 forward，直接逐组件对比 batch 与单样本：
  feature2state 的三段 state、entity_encoder 输出、scalar_encoder 输出。
哪一段出现偏差，根因就在哪一段。
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


def cmp(name, a, b):
    d = (a.detach().reshape(-1) - b.detach().reshape(-1)).abs().max().item()
    print("    %-26s max|d| = %.3e %s" % (name, d, "" if d < 1e-6 else "  <-- 不一致"))
    return d


def main():
    torch.manual_seed(1234)
    dev = torch.device("cuda:0")
    net = ArchModel().to(dev).eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 4
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    bf = feats[idx].float().to(dev)

    with torch.no_grad():
        st_b = Feature.feature2state(bf)
        ee_b = net.entity_encoder(st_b.entity_state, return_unit_types=True)
        sc_b = net.scalar_encoder(st_b.statistical_state)
        sp_b = net.spatial_encoder(st_b.map_state, ee_b[0])
        h = torch.randn(1, N, 128, device=dev)
        c = torch.randn(1, N, 128, device=dev)
        lo_b, nh_b = net.core(sc_b[0], ee_b[1], sp_b[1], N, 1, (h, c))

        for i in range(N):
            st_i = Feature.feature2state(bf[i:i + 1])
            print("sample %d" % i)
            cmp("entity_state", st_b.entity_state[i:i + 1], st_i.entity_state)
            cmp("map_state", st_b.map_state[i:i + 1], st_i.map_state)
            for j, (a, b) in enumerate(zip(st_b.statistical_state, st_i.statistical_state)):
                aa = a[i:i + 1] if a.shape[0] == N else a
                cmp("statistical[%d]" % j, aa, b)

            ee_i = net.entity_encoder(st_i.entity_state, return_unit_types=True)
            cmp("entity_embeddings", ee_b[0][i:i + 1], ee_i[0])
            cmp("embedded_entity", ee_b[1][i:i + 1], ee_i[1])
            cmp("unit_types_one", ee_b[3][i:i + 1], ee_i[3])

            sc_i = net.scalar_encoder(st_i.statistical_state)
            cmp("embedded_scalar", sc_b[0][i:i + 1], sc_i[0])
            cmp("scalar_context", sc_b[1][i:i + 1], sc_i[1])

            sp_i = net.spatial_encoder(st_i.map_state, ee_i[0])
            cmp("embedded_spatial", sp_b[1][i:i + 1], sp_i[1])
            lo_i, nh_i = net.core(sc_i[0], ee_i[1], sp_i[1], 1, 1,
                                  (h[:, i:i + 1].contiguous(), c[:, i:i + 1].contiguous()))
            cmp("lstm_output", lo_b[i:i + 1], lo_i)
            cmp("new_hidden_h", nh_b[0][:, i:i + 1], nh_i[0])


if __name__ == "__main__":
    main()
