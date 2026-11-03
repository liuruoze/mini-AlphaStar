"""T0 + T1 验收脚本（DeepSeek 执行记录）

T0: pick_cuda_device 单测 —— 确认 rank 0..14 都落在真实存在的卡上，且 MAS_GPU_LIST 生效
T1: Agent.batch_action_logits_by_state 与单样本 forward 的数值一致性
    比 action_type_logits + LSTM new_hidden，要求 max|Δ| < 1e-3
    注意：两条路径都传 obs_list=None，避免 get_batch_unit_type_mask 在无真实 obs 时崩
"""
import os
import sys
import glob

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()


def test_t0():
    print("== T0: pick_cuda_device ==", flush=True)
    from alphastarmini.core.rl.rl_vs_inner_bot_mp import pick_cuda_device
    n_gpu = torch.cuda.device_count()
    print("可见 GPU 数: %d" % n_gpu, flush=True)

    ok = True
    for rank in range(16):
        d = pick_cuda_device(rank, True)
        idx = int(d.split(":")[1]) if d.startswith("cuda:") else -1
        bad = d != "cpu" and (idx < 0 or idx >= n_gpu)
        ok = ok and not bad
        print("  rank %2d -> %-8s %s" % (rank, d, "  <-- 越界!" if bad else ""), flush=True)

    os.environ["MAS_GPU_LIST"] = "1,2,3"
    print("  MAS_GPU_LIST=1,2,3 时:", flush=True)
    got = [pick_cuda_device(r, True) for r in range(6)]
    for r, g in enumerate(got):
        print("  rank %2d -> %s" % (r, g), flush=True)
    whitelist_ok = got == ["cuda:1", "cuda:2", "cuda:3", "cuda:1", "cuda:2", "cuda:3"]
    del os.environ["MAS_GPU_LIST"]
    print("  ==> T0 %s\n" % ("PASS" if (ok and whitelist_ok) else "FAIL"), flush=True)
    return ok and whitelist_ok


def test_t1():
    print("== T1: batch_action_logits_by_state 数值一致性 ==", flush=True)
    from alphastarmini.core.arch.agent import Agent
    from alphastarmini.core.sl.feature import Feature

    dev = torch.device("cuda:0")
    agent = Agent()
    agent.to(dev)
    net = agent.model.eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    bf = feats[idx].float().to(dev)

    torch.manual_seed(1234)
    h = torch.randn(1, N, 128, device=dev)
    c = torch.randn(1, N, 128, device=dev)

    with torch.no_grad():
        single_at, single_h, single_c = [], [], []
        for i in range(N):
            st = Feature.feature2state(bf[i:i + 1])
            hi = (h[:, i:i + 1].contiguous(), c[:, i:i + 1].contiguous())
            out = net.forward(st, batch_size=1, sequence_length=1, hidden_state=hi,
                              return_logits=True, obs_list=None)
            single_at.append(out[0].action_type.detach().reshape(-1).clone())
            single_h.append(out[2][0].detach().reshape(-1).clone())
            single_c.append(out[2][1].detach().reshape(-1).clone())

        st_b = Feature.feature2state(bf)
        al, acts, nh, sun, en = agent.batch_action_logits_by_state(
            st_b, hidden_state=(h, c), obs_list=None)

    print("  action_type_logits: %s | hidden: %s | select_units_num: %s"
          % (tuple(al.action_type.shape), tuple(nh[0].shape), tuple(sun.shape)), flush=True)
    worst = 0.0
    for i in range(N):
        d1 = (single_at[i] - al.action_type[i].reshape(-1)).abs().max().item()
        d2 = (single_h[i] - nh[0][:, i].reshape(-1)).abs().max().item()
        d3 = (single_c[i] - nh[1][:, i].reshape(-1)).abs().max().item()
        worst = max(worst, d1, d2, d3)
        print("  sample %d: dLogit=%.3e dH_h=%.3e dH_c=%.3e" % (i, d1, d2, d3), flush=True)

    passed = worst < 1e-3
    print("  ==> T1 %s (最大偏差 %.3e)\n" % ("PASS" if passed else "FAIL", worst), flush=True)
    return passed


if __name__ == "__main__":
    a = test_t0()
    b = test_t1()
    print("== 汇总: T0 %s | T1 %s ==" % ("PASS" if a else "FAIL", "PASS" if b else "FAIL"))
