"""探测 behavior_logits 各字段的真实 shape 与体积。

目的：behavior_logits 回传需要在 fork 前预分配共享内存，所以必须事先知道
每个 head 的 logits 维度。这里用真实 forward 测出来，而不是猜常量。

同时验证：`al`（ArgsActionLogits）在 return_logits=True 时是否含全部 6 个 head。
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

import numpy as np
import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP


def main():
    import glob
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    print("用数据集:", f0, "帧数:", feats.shape[0], flush=True)

    idx = torch.linspace(0, feats.shape[0] - 1, 4).long()
    st = Feature.feature2state(feats[idx].float())

    dev = torch.device("cuda:0")
    agent = Agent()
    agent.to(dev)
    agent.model.eval()

    st_gpu = type(st)(entity_state=st.entity_state.float().to(dev),
                      statistical_state=[x.float().to(dev) for x in st.statistical_state],
                      map_state=st.map_state.float().to(dev))
    h = torch.zeros(1, 4, 128, device=dev)
    c = torch.zeros(1, 4, 128, device=dev)

    with torch.no_grad():
        al, acts, nh, sun, en = agent.batch_action_logits_by_state(
            st_gpu, hidden_state=(h, c), obs_list=None)

    print("\n=== al (ArgsActionLogits) 各字段 ===", flush=True)
    total = 0
    for name in ("action_type", "delay", "queue", "units", "target_unit", "target_location"):
        v = getattr(al, name, None)
        if v is None:
            print("  %-16s None" % name, flush=True)
            continue
        per = v[0].numel() if v.numel() % v.shape[0] == 0 else -1
        total += per
        print("  %-16s shape=%-22s dtype=%-14s 每样本 %d 个 float32 (%.1f KB)"
              % (name, tuple(v.shape), v.dtype, per, per * 4 / 1024), flush=True)

    print("\n=== 每样本合计 %d floats = %.1f KB ===" % (total, total * 4 / 1024), flush=True)
    print("  24 actor 每轮 = %.2f MB" % (total * 4 * 24 / 1024 / 1024), flush=True)

    print("\n=== 从常量推导的对照 ===", flush=True)
    print("  Label_Size.action_type_encoding =", LS.action_type_encoding, flush=True)
    print("  Label_Size.delay_encoding       =", LS.delay_encoding, flush=True)
    print("  Label_Size.queue_encoding       =", LS.queue_encoding, flush=True)
    print("  AHP.max_selected / max_entities =", AHP.max_selected, "/", AHP.max_entities, flush=True)
    print("  world_size^2                    =", SCHP.world_size ** 2, flush=True)
    print("  推导 units = (max_selected+1) x max_entities =",
          (AHP.max_selected + 1) * AHP.max_entities, flush=True)

    print("\n=== acts（采样的动作）===", flush=True)
    for name in ("action_type", "delay", "queue", "units", "target_unit", "target_location"):
        v = getattr(acts, name, None)
        print("  %-16s %s" % (name, tuple(v.shape) if v is not None else None), flush=True)


if __name__ == "__main__":
    main()
