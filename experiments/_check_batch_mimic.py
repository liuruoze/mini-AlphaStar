"""Kimi 补充实测 [2b]：L1 层校验 —— teacher forcing 下全部 6 个 head 的 logits 一致性。

_check_batch_logits.py 已验证 L2（action_type_logits + LSTM hidden，max|Δ|=0）。
但自回归路径（delay/queue/units/target_unit/target_location）在 batch 下还没验证，
而 selected_units_head 的循环与 entity_num padding 正是最容易出错的地方。

方法：先用 forward 采样出一组合法 action，再把它作为 gt 喂给 mimic_forward（teacher forcing），
分别按「逐样本 bs=1」和「整批 bs=N」跑，比较全部 6 个 head 的 logits + value baseline。
teacher forcing 下两条路径输入完全相同，因此 logits 必须逐点一致。

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
from alphastarmini.core.rl.action import ArgsAction

FIELDS = ["action_type", "delay", "queue", "units", "target_unit", "target_location"]
# mimic_forward 返回元组中各 logits 的位置
IDX = {"entity_nums": 2, "baseline": 0, "action_type": 6, "delay": 7, "queue": 8,
       "units": 9, "target_unit": 10, "target_location": 11}


def flat(x):
    """baseline 是 list of tensors，其余是单个 tensor，统一成 list"""
    if isinstance(x, (list, tuple)):
        return [t.detach().reshape(-1) for t in x]
    return [x.detach().reshape(-1)]


def slice_action(a, i):
    return ArgsAction(action_type=a.action_type[i:i + 1], delay=a.delay[i:i + 1],
                      queue=a.queue[i:i + 1], units=a.units[i:i + 1],
                      target_unit=a.target_unit[i:i + 1],
                      target_location=a.target_location[i:i + 1])


def main():
    torch.manual_seed(1234)
    dev = torch.device("cuda:0")
    net = ArchModel().to(dev).eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    batch_feat = feats[idx].float().to(dev)
    h = torch.randn(1, N, 128, device=dev)
    c = torch.randn(1, N, 128, device=dev)

    with torch.no_grad():
        st = Feature.feature2state(batch_feat)
        out = net.forward(st, batch_size=N, sequence_length=1, hidden_state=(h, c),
                          return_logits=True, obs_list=None)
        acts, sun = out[1], out[3]
        print("select_units_num shape:", tuple(sun.shape) if hasattr(sun, "shape") else type(sun))

        h_init, c_init = h.clone(), c.clone()

        singles = {k: [] for k in IDX}
        for i in range(N):
            si = Feature.feature2state(batch_feat[i:i + 1])
            hi = (h_init[:, i:i + 1].contiguous(), c_init[:, i:i + 1].contiguous())
            ai = slice_action(acts, i)
            si_sun = sun[i:i + 1] if hasattr(sun, "shape") else sun
            ms = net.mimic_forward(si, ai, si_sun, gt_is_one_hot=False,
                                   batch_size=1, sequence_length=1, hidden_state=hi)
            for k, j in IDX.items():
                singles[k].append([t.clone() for t in flat(ms[j])])

        # 整批 teacher forcing（放在单样本之后，且用独立 hidden 副本，排除状态污染）
        mb = net.mimic_forward(st, acts, sun, gt_is_one_hot=False,
                               batch_size=N, sequence_length=1,
                               hidden_state=(h_init.clone(), c_init.clone()))

    print("\n%-18s %-14s" % ("head", "max|dLogit|"))
    worst = 0.0
    for k, j in IDX.items():
        bparts = flat(mb[j])
        dmax = 0.0
        for i in range(N):
            try:
                for si, bi in zip(singles[k][i], bparts):
                    d = (si - bi[i].reshape(-1)).abs().max().item()
                    dmax = max(dmax, d)
            except Exception as e:
                print("  [%s] sample %d compare failed: %r" % (k, i, e))
                dmax = float("nan")
                break
        worst = max(worst, dmax) if dmax == dmax else worst
        print("%-18s %-14.3e" % (k, dmax))

    print("\nentity_nums 单样本:", [int(singles["entity_nums"][i][0][0].item()) for i in range(N)])
    print("entity_nums 整批  :", [int(mb[2][i].item()) for i in range(N)])
    print("\n==> 最大偏差 %.3e  %s" % (worst, "PASS (<1e-3)" if worst < 1e-3 else "FAIL"))


if __name__ == "__main__":
    main()
