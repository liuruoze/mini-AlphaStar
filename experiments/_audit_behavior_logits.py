"""T4 验收：behavior_logits 回传的正确性。

为什么需要这个测试
------------------
接 RL 训练时 VTrace 要算 rho = pi_target(a|s) / pi_behavior(a|s)。
rl_loss.py:377 `get_logprob_and_rhos` 直接读 `trajectories.behavior_logits`，
把它当 ArgsActionLogits 用（6 个 head 的原始 logits，loss 端自己算 log_prob）。
所以回传的东西必须是**产生该 action 的那一次前向的 logits**。

一旦错位（比如拿到上一个 actor 的、或上一步的 logits），程序**不会报错**，
只会让 rho 静默算错 -> VTrace 的 vs 与 pg_advantages 全错 -> 训练悄悄走偏。
这是最难查的一类 bug，所以必须有严格的验收。

四层验收
--------
[A] 逐位一致：read_behavior_logits() 与直接 forward 的 al 完全相同（max|Δ| = 0）
[B] **rho == 1.0（黄金标准）**：behavior 与 target 是同一份权重、同一 state，
    用同一 action 算 log_prob，比值必须恰好 1。任何错位都会让 rho 明显偏离 1。
[C] 变 batch size：动态批的 batch 是变的，验证子集（k=3、k=5）下仍正确
[D] 乱序：actor 索引乱序提交（模拟真实调度），验证 logits 不会张冠李戴

不依赖 SC2 env，纯推理层面，秒级完成。
"""
import os
import sys
import glob
import contextlib

sys.path.insert(0, os.getcwd())

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
from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes, LOGIT_FIELDS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")


@contextlib.contextmanager
def deterministic_sampling():
    """把 torch.multinomial 替换成 argmax，消除自回归采样的随机性。

    为什么必须这么做（这是本测试最重要的一处设计）：
      `action_logits_by_state` / `forward(return_logits=True)` 是**自回归采样**——
      action_type 的采样结果会进入 delay/queue/units/target 的 autoregressive_embedding。
      两次独立 forward 即使权重与输入完全相同，action_type 采样也可能不同，
      于是后续 head 的 logits **必然不同**（实测 delay 差 0.028、units 差 1e9 的 mask 位置）。

      第一版测试直接比"server 传回的 logits vs 另一次 forward 的 logits"，
      6 个 head 里只有 action_type 是 0 偏差 —— 这不是 bug，是**验收方法错了**。
      （Kimi 那轮在批推理校验上踩过同一个坑，这里又出现一次。）

      修法：把采样固定成 argmax。此时两次 forward 走**完全相同的自回归路径**，
      全部 6 个 head 的 logits 必须逐位相等，任何偏差都是真的传输错误。
      argmax(probs) == argmax(logits)，所以替换不改变语义。
    """
    orig = torch.multinomial

    def _argmax(input, num_samples=1, replacement=False):
        idx = input.argmax(dim=-1, keepdim=True)
        if int(num_samples) > 1:
            idx = idx.expand(-1, int(num_samples))
        return idx

    torch.multinomial = _argmax
    try:
        yield
    finally:
        torch.multinomial = orig



def load_frames(n):
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    idx = torch.linspace(0, feats.shape[0] - 1, n).long()
    return feats[idx].float()


def make_server(agent, n, return_logits=True):
    shapes = StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])
    return AsyncBatchInferenceServer(agent, n, DEV, shapes,
                                     max_units=128, max_selected=AHP.max_selected,
                                     max_batch=n, return_logits=return_logits)


def feed(srv, feats, idxs):
    """把第 i 个 state 写到 actor i 的槽位。"""
    for slot, fi in enumerate(idxs):
        st = Feature.feature2state(feats[fi:fi + 1])
        ut = np.zeros(8, dtype=np.int64)
        srv.write_state(slot, st, ut)


def logprob_of(logits, action):
    """所选动作的 log 概率（不处理 mask，仅用于一致性校验）。"""
    lp = torch.log_softmax(logits.reshape(1, -1).float(), dim=-1)
    a = int(action.reshape(-1)[0])
    return float(lp[0, a])


def main_body():
    torch.manual_seed(7)
    feats = load_frames(10)
    N = 8

    agent = Agent()
    agent.to(DEV)
    agent.model.eval()

    srv = make_server(agent, N, return_logits=True)
    print("server 建好：return_logits=%s，logit_shapes=%s"
          % (srv.return_logits, {k: v for k, v in srv.logit_shapes.items()}), flush=True)

    feed(srv, feats, list(range(N)))
    h0 = srv.h.clone()
    c0 = srv.c.clone()

    # 参考：用完全相同的输入直接 forward（用 hidden 副本，见下方 device 说明）
    # 注意 srv.h / srv.c 是 CPU 共享内存张量，_serve_batch 内部会 .to(device)，
    # 这里也必须显式搬 —— 否则报 "Input and hidden tensors are not at the same device"。
    st_ref = srv._build_state_subset(torch.arange(N), N)
    with torch.no_grad():
        al_ref, acts_ref, nh_ref, sun_ref, en_ref = agent.batch_action_logits_by_state(
            st_ref, hidden_state=(h0.to(DEV).clone(), c0.to(DEV).clone()),
            obs_list=srv._build_obs_list())

    # 正式：走 server 的批前向
    srv._serve_batch(list(range(N)))

    ok = True

    # ---------- [A] 逐位一致 ----------
    print("\n=== [A] behavior_logits 与直接 forward 逐位一致 ===", flush=True)
    for i in range(N):
        bl = srv.read_behavior_logits(i)
        per = {}
        for name in LOGIT_FIELDS:
            got = getattr(bl, name)
            ref = getattr(al_ref, name)[i:i + 1]
            if tuple(got.shape) != tuple(ref.shape):
                per[name] = "形状不符 %s vs %s" % (tuple(got.shape), tuple(ref.shape))
                ok = False
                continue
            per[name] = float((got - ref.cpu()).abs().max())
            if per[name] != 0.0:
                ok = False
        if i == 0:
            for name in LOGIT_FIELDS:
                print("    %-16s max|Δ| = %s" % (name, per[name]), flush=True)
        bad = [k for k, v in per.items() if v != 0.0]
        print("  actor %d  %s" % (i, "PASS" if not bad else "FAIL -> %s" % bad), flush=True)

    # ---------- [B] rho == 1.0（黄金标准）----------
    # ⚠️ 必须用**同一个 action** 分别对 behavior/target logits 求 log_prob。
    # 第一版拿 acts_ref（另一次独立采样）当 action，而 behavior 侧用的是
    # _serve_batch 那次采样的 action —— 两次采样不同，rho 必然偏离 1，测的是采样噪声。
    print("\n=== [B] rho = pi_target/pi_behavior == 1.0（同权重同 state 同 action）===", flush=True)
    for i in range(N):
        bl = srv.read_behavior_logits(i)
        act = srv.read_action(i)
        a_type = torch.tensor([act[0]])
        a_delay = torch.tensor([act[1]])
        lp_b = logprob_of(bl.action_type, a_type)
        lp_t = logprob_of(al_ref.action_type[i:i + 1], a_type)
        rho = np.exp(lp_t - lp_b)
        lp_b_d = logprob_of(bl.delay, a_delay)
        lp_t_d = logprob_of(al_ref.delay[i:i + 1], a_delay)
        rho_d = np.exp(lp_t_d - lp_b_d)
        verdict = "PASS" if abs(rho - 1.0) < 1e-6 and abs(rho_d - 1.0) < 1e-6 else "FAIL"
        print("  actor %d  rho(action_type)=%.9f  rho(delay)=%.9f  %s"
              % (i, rho, rho_d, verdict), flush=True)
        if verdict == "FAIL":
            ok = False

    # ---------- [C] 变 batch size（动态批的真实形态）----------
    print("\n=== [C] 变 batch size：子集 k=3 / k=5 ===", flush=True)
    from alphastarmini.core.rl.batch_server import FakeRawUnits
    for subset in ([0, 3, 7], [1, 2, 4, 5, 6]):
        for i in range(N):
            srv.h[:, i, :] = 0
            srv.c[:, i, :] = 0
        k = len(subset)
        st_sub = srv._build_state_subset(torch.as_tensor(subset), k)
        with torch.no_grad():
            al_s = agent.batch_action_logits_by_state(
                st_sub, hidden_state=(torch.zeros(1, k, 128, device=DEV),
                                      torch.zeros(1, k, 128, device=DEV)),
                obs_list=[{"raw_units": FakeRawUnits(srv.obs_ut[i].numpy())} for i in subset])[0]
        srv._serve_batch(subset)
        worst, bad_head = 0.0, None
        for j, i in enumerate(subset):
            bl = srv.read_behavior_logits(i)
            for name in LOGIT_FIELDS:
                d = float((getattr(bl, name) -
                           getattr(al_s, name)[j:j + 1].cpu()).abs().max())
                if d > worst:
                    worst, bad_head = d, name
        print("  subset=%s (k=%d)  max|Δ| = %.3e %s  %s"
              % (subset, k, worst, "" if bad_head is None else "(%s)" % bad_head,
                 "PASS" if worst == 0.0 else "FAIL"), flush=True)
        if worst != 0.0:
            ok = False

    # ---------- [D] 乱序提交 ----------
    print("\n=== [D] 乱序提交（模拟真实调度）===", flush=True)
    for i in range(N):
        srv.h[:, i, :] = 0
        srv.c[:, i, :] = 0
    order = [5, 2, 7, 0, 3, 6, 1, 4]
    st_sh = srv._build_state_subset(torch.as_tensor(order), len(order))
    with torch.no_grad():
        al_sh = agent.batch_action_logits_by_state(
            st_sh, hidden_state=(torch.zeros(1, len(order), 128, device=DEV),
                                 torch.zeros(1, len(order), 128, device=DEV)),
            obs_list=[{"raw_units": FakeRawUnits(srv.obs_ut[i].numpy())} for i in order])[0]
    srv._serve_batch(order)
    worst, bad_head = 0.0, None
    for j, i in enumerate(order):
        bl = srv.read_behavior_logits(i)
        for name in LOGIT_FIELDS:
            d = float((getattr(bl, name) - getattr(al_sh, name)[j:j + 1].cpu()).abs().max())
            if d > worst:
                worst, bad_head = d, name
    print("  order=%s  max|Δ| = %.3e %s  %s"
          % (order, worst, "" if bad_head is None else "(%s)" % bad_head,
             "PASS" if worst == 0.0 else "FAIL"), flush=True)
    if worst != 0.0:
        ok = False

    # ---------- 附带：不开 logits 时应报错而不是静默返回 ----------
    print("\n=== [E] return_logits=False 时必须显式报错（防静默错误）===", flush=True)
    srv2 = make_server(agent, 4, return_logits=False)
    feed(srv2, feats, list(range(4)))
    srv2._serve_batch(list(range(4)))
    try:
        srv2.read_behavior_logits(0)
        print("  FAIL：没有报错，会静默返回 None/垃圾值", flush=True)
        ok = False
    except RuntimeError as e:
        print("  PASS：抛出 RuntimeError（%s...）" % str(e)[:40], flush=True)

    print("\n==> behavior_logits 回传审计：%s" % ("全部 PASS" if ok else "有 FAIL"), flush=True)
    return 0 if ok else 1


def main():
    # 全流程固定采样（argmax），否则自回归的随机性会让"逐位比对"必然 FAIL，
    # 详见 deterministic_sampling 的 docstring。
    with deterministic_sampling():
        return main_body()


if __name__ == "__main__":
    sys.exit(main())
