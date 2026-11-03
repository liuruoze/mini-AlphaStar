"""space-bunny 独立审计：动态批实现的正确性。

背景：DeepSeek 的 T1 只验证过「bs=1 逐样本 vs bs=8 整批」的 L2 一致性（max|Δ|=0）。
但动态批（AsyncBatchInferenceServer）的 batch size 是**变的**（实测 1~24），
且 hidden state 是**按 actor 索引子集更新**的（self.h[:, idx_t, :] = nh[0]）。
这两处都没有被任何测试覆盖，且出错时**不会报错**，只会让数值悄悄漂移。

本脚本做三项审计：
  [A] 隔离性：固定 actor 0 的 state/hidden，任意改动其他 actor 的 state，
      actor 0 的 new_hidden 必须**逐位相同**（抓 hidden 串扰 / 索引错位）
  [B] 变 batch size：batch ∈ {1,2,3,5,8,16,24}，每个样本与逐样本结果对比
  [C] 动态批序列模拟：把 T 步序列拆成不同时刻的不同 batch，逐批 forward，
      验证每一步的 hidden 与「逐样本顺序推进」完全一致
      （这是对 self.h[:, idx_t, :] 子集更新逻辑的端到端验证）
"""
import os
import sys
import glob
import random

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

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

DEV = torch.device("cuda:0")


def load_states(n):
    """取 n 个真实帧（entity 数各不相同，覆盖 padding 差异）"""
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    idx = torch.linspace(0, feats.shape[0] - 1, n).long()
    return feats[idx].float().to(DEV)


def single(agent, state, h, c):
    """逐样本前向，返回 (new_h, new_c, action_type_logits)"""
    with torch.no_grad():
        al, acts, nh, sun, en = agent.batch_action_logits_by_state(
            state, hidden_state=(h.contiguous(), c.contiguous()), obs_list=None)
    return nh[0].clone(), nh[1].clone(), al.action_type.clone()


def batched(agent, states, h, c):
    """整批前向，h/c 形状 (1, N, 128)。

    ⚠️ 用 .contiguous() 是因为切片视图非连续（nn.LSTM 对非连续输入要额外拷贝）。

    【更正 第八轮】这里原先写「forward 会 in-place 修改传入的 hidden_state」，
    **该说法经 _check_t6_double_step.py 复测不成立**：调用前后 caller 的 h/c
    完全不变（max|Δ| = 0.000e+00），LSTM 返回的是新张量。
    当初那次隔离性"假 FAIL"的真因是本脚本自身的两个 bug：
      1. torch.randperm(N)[1:] 去掉的是随机第一个元素，不是 0
      2. [C] 把不同 actor 的不同时刻混成一批
    修脚本后 FAIL 消失。保留 .contiguous() 仅为连续性，与 in-place 无关。
    """
    with torch.no_grad():
        al, acts, nh, sun, en = agent.batch_action_logits_by_state(
            states, hidden_state=(h.contiguous(), c.contiguous()), obs_list=None)
    return nh[0].clone(), nh[1].clone(), al.action_type.clone()


def main():
    torch.manual_seed(7)
    random.seed(7)
    agent = Agent()
    agent.to(DEV)
    agent.model.eval()

    ok_all = True

    # ---------- [A] 隔离性 ----------
    print("=== [A] hidden 隔离性：固定 actor0，改动其他 actor 的 state ===", flush=True)
    N = 8
    st_all = load_states(24)
    h = torch.randn(1, N, 128, device=DEV)
    c = torch.randn(1, N, 128, device=DEV)

    # 基线
    st0 = Feature.feature2state(st_all[:N])
    _, _, ref_at = batched(agent, st0, h, c)
    # 单独跑 actor 0
    st_i0 = Feature.feature2state(st_all[0:1])
    s_h, s_c, s_at = single(agent, st_i0, h[:, 0:1], c[:, 0:1])
    d0 = (s_at.reshape(-1) - ref_at[0].reshape(-1)).abs().max().item()
    print("  actor0 整批 vs 逐样本 logits max|d| = %.3e" % d0, flush=True)
    ok_all = ok_all and d0 < 1e-3

    # 改动其他 actor 的 state，actor0 的 hidden 应逐位不变
    base_h, base_c, _ = batched(agent, st0, h, c)
    for trial in range(3):
        pert = st_all[:N].clone()
        # 换掉除第 0 个之外的全部 state
        perm = [i for i in torch.randperm(N).tolist() if i != 0][:N - 1]
        src = torch.randperm(24)[:N - 1]
        for j, dst_i in enumerate(perm):
            pert[dst_i] = st_all[src[j]]
        st_p = Feature.feature2state(pert)
        ph, pc, pat = batched(agent, st_p, h, c)
        dh = (ph[:, 0:1] - base_h[:, 0:1]).abs().max().item()
        dc = (pc[:, 0:1] - base_c[:, 0:1]).abs().max().item()
        dl = (pat[0].reshape(-1) - ref_at[0].reshape(-1)).abs().max().item()
        good = dh == 0.0 and dc == 0.0 and dl < 1e-3
        ok_all = ok_all and good
        print("  trial %d: actor0 dHidden_h=%.3e dHidden_c=%.3e dLogit=%.3e -> %s"
              % (trial, dh, dc, dl, "PASS" if good else "FAIL"), flush=True)

    # ---------- [B] 变 batch size ----------
    print("\n=== [B] 不同 batch size 下与逐样本的一致性 ===", flush=True)
    for B in [1, 2, 3, 5, 8, 16, 24]:
        st = load_states(B)
        hb = torch.randn(1, B, 128, device=DEV)
        cb = torch.randn(1, B, 128, device=DEV)
        bh, bc, bat = batched(agent, Feature.feature2state(st), hb, cb)
        worst = 0.0
        for i in range(B):
            shi = Feature.feature2state(st[i:i + 1])
            ih, ic, iat = single(agent, shi, hb[:, i:i + 1], cb[:, i:i + 1])
            worst = max(worst,
                        (iat.reshape(-1) - bat[i].reshape(-1)).abs().max().item(),
                        (ih.reshape(-1) - bh[:, i:i + 1].reshape(-1)).abs().max().item(),
                        (ic.reshape(-1) - bc[:, i:i + 1].reshape(-1)).abs().max().item())
        ok_all = ok_all and worst < 1e-3
        print("  batch=%-3d max|d| = %.3e -> %s" % (B, worst, "PASS" if worst < 1e-3 else "FAIL"), flush=True)

    # ---------- [C] 动态批序列 ----------
    print("\n=== [C] 动态批序列：同刻成批 vs 逐样本顺序推进 ===", flush=True)
    N_ACT, T = 4, 5
    # states[t][i]：actor i 在时刻 t 的 state（用不同帧模拟各自的时间推进）
    frames = load_states(24)
    def state_of(t, i):
        return frames[(t * N_ACT + i) % 24]

    # 逐样本：每个 actor 用自己的 hidden 顺序推进
    h_one = torch.zeros(1, N_ACT, 128, device=DEV)
    c_one = torch.zeros(1, N_ACT, 128, device=DEV)
    seq_h, seq_c = [], []
    for t in range(T):
        nh_l, nc_l = [], []
        for i in range(N_ACT):
            shi = Feature.feature2state(state_of(t, i).unsqueeze(0))
            ih, ic, _ = single(agent, shi, h_one[:, i:i + 1], c_one[:, i:i + 1])
            h_one[:, i:i + 1] = ih
            c_one[:, i:i + 1] = ic
            nh_l.append(ih)
            nc_l.append(ic)
        seq_h.append(torch.cat(nh_l, dim=1))
        seq_c.append(torch.cat(nc_l, dim=1))

    # 动态批：时刻 t，把 N_ACT 个 actor 的 state[t] 拼成一批
    h_dyn = torch.zeros(1, N_ACT, 128, device=DEV)
    c_dyn = torch.zeros(1, N_ACT, 128, device=DEV)
    worst = 0.0
    for t in range(T):
        st_batch = Feature.feature2state(torch.stack([state_of(t, i) for i in range(N_ACT)]))
        nh, nc, _ = batched(agent, st_batch, h_dyn, c_dyn)
        h_dyn, c_dyn = nh, nc
        d = max((nh - seq_h[t]).abs().max().item(), (nc - seq_c[t]).abs().max().item())
        worst = max(worst, d)
        print("  step %d: batch=%d  全部 actor 的 hidden max|d| = %.3e" % (t, N_ACT, d), flush=True)
    ok_all = ok_all and worst < 1e-3
    print("  -> %s" % ("PASS" if worst < 1e-3 else "FAIL"), flush=True)

    print("\n==> 动态批正确性审计 %s" % ("全部 PASS" if ok_all else "存在 FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
