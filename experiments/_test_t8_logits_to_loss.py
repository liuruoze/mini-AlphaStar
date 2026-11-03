"""T8 验收：behavior_logits 能不能被 RL loss 真正消费。

前一轮（_audit_behavior_logits.py）证明了"server 传回的 logits 与直接 forward 逐位一致"，
但那只是**传输正确**。接进训练还差一步：**loss 端能不能吃它、算出的 rho 对不对**。

本脚本走通这条链路：
    batch_server.submit_and_get() -> ArgsActionLogits
      -> 组织成 trajectories.behavior_logits 的结构（list of list）
      -> RA.log_prob()          （rl_algo.py:438，loss 里真正用的那个函数）
      -> RA.compute_cliped_importance_weights()  （rl_algo.py:416）
      -> rho

三个验收点
----------
[A] shape/dtype 兼容：loss 端的 log_prob 能直接吃我们回传的 logits，不需要任何转换
[B] **rho 恒等于 1**：同权重、同 state、同 action（用 argmax 固定采样路径）
[C] **rho 会随权重变化**：把权重微扰后重算 target，rho 必须偏离 1，
    且**全部 ≤ 1**（因为 compute_cliped_importance_weights 是 clamp(max=1.0)，
    VTrace 的 rho 是不对称的 —— 这点很容易误判，单独列一条）

⚠️ 为什么必须用 argmax 固定采样：
    behavior 与 target 是两次独立前向。若用 multinomial，两次的 action_type 采样不同，
    后续 head 的 logits 必然不同（自回归），rho 就不等于 1 —— 测的会是采样噪声而不是链路正确性。
    这是本项目被不同模型踩过三次的坑（见 BATCH_ACTING_EXECUTION_LOG_DS3.md §2.4）。
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
from alphastarmini.core.rl.batch_server import (
    AsyncBatchInferenceServer, StateShapes, FakeRawUnits, LOGIT_FIELDS)
from alphastarmini.core.rl import rl_algo as RA
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")
T = 8          # 轨迹长度
MAX_SEL = AHP.max_selected


@contextlib.contextmanager
def deterministic_sampling():
    """把 torch.multinomial 换成 argmax，固定自回归采样路径。"""
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


def load_feats(n):
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    return torch.load(f0)[0][:n].float()


def make_server(agent, n=1):
    shapes = StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])
    srv = AsyncBatchInferenceServer(agent, n, DEV, shapes,
                                    max_units=128, max_selected=MAX_SEL,
                                    max_batch=n, return_logits=True)
    srv.agent = agent
    return srv


def collect(srv, feats):
    """走 server 的批前向，收 T 步的 (action, behavior_logits)。

    返回 behaviors: list，每项是 ArgsActionLogits（形状 [1, ...]，CPU）。
    同时返回 actions: list of dict（拍平后的整数，便于后面组 tensor）。
    """
    behaviors, actions = [], []
    n = 1
    for t in range(feats.shape[0]):
        st = Feature.feature2state(feats[t:t + 1])
        ut = np.zeros(8, dtype=np.int64)
        srv.write_state(0, st, ut)
        srv._serve_batch([0])
        a = srv.read_action(0)
        bl = srv.read_behavior_logits(0)
        behaviors.append(bl)
        actions.append(a)
    return behaviors, actions


def flatten_actions(actions):
    """把 read_action 的 7 元组转成 loss 端要的形状：[T, 1] 或 [T, 1, k]。

    直接放到 DEV：真实 loss 里 `filter_by_for_lists` 也是 `getattr(b, field).to(device)`，
    两边都在 GPU 上才能算 log_prob。
    """
    out = {}
    out["action_type"] = torch.tensor([[a[0]] for a in actions], dtype=torch.int64, device=DEV)
    out["delay"] = torch.tensor([[a[1]] for a in actions], dtype=torch.int64, device=DEV)
    out["queue"] = torch.tensor([[a[2]] for a in actions], dtype=torch.int64, device=DEV)
    out["target_unit"] = torch.tensor([[a[4]] for a in actions], dtype=torch.int64, device=DEV)
    # units: [T, 1, max_sel]，每步的有效长度是 a[6]
    units = torch.zeros(len(actions), 1, MAX_SEL, dtype=torch.int64, device=DEV)
    for i, a in enumerate(actions):
        k = min(a[6], MAX_SEL)
        units[i, 0, :k] = a[3][:k]
    out["units"] = units
    out["target_location"] = torch.tensor([[int(a[5][0]), int(a[5][1])] for a in actions],
                                          dtype=torch.int64, device=DEV)
    return out


def cat_field(logits_list, field):
    """复刻 rl_loss.py:53 filter_by_for_lists 的语义：
    torch.cat([getattr(b, field).to(device) for a in target_list for b in a], dim=0)
    —— 即把每步的 [1, ...] 沿 dim=0 拼成 [T, ...]。"""
    return torch.cat([getattr(b, field).to(DEV) for b in logits_list], dim=0)


def logprob(logits, actions, field):
    """调 RL loss 里真正用的 log_prob（无 mask 的 head）。"""
    return RA.log_prob(logits, actions, [None, None, None], MAX_SEL)


def main():
    torch.manual_seed(17)
    feats = load_feats(T)
    agent = Agent()
    agent.to(DEV)
    agent.model.eval()
    srv = make_server(agent, n=1)

    ok = True
    with deterministic_sampling():
        # ---------- [A] 收轨迹，验证 loss 端能直接吃 ----------
        print("=== [A] loss 端能否直接消费 server 回传的 logits ===", flush=True)
        behaviors, actions = collect(srv, feats)
        acts = flatten_actions(actions)

        shape_ok = True
        for f in LOGIT_FIELDS:
            lg = cat_field(behaviors, f)
            act_raw = acts[f]
            print("  %-16s logits %-18s action %s" % (f, tuple(lg.shape), tuple(act_raw.shape)),
                  flush=True)
            if lg.shape[0] != T:
                shape_ok = False
        if not shape_ok:
            print("  FAIL：拼接后的时间维不是 %d" % T, flush=True)
            ok = False
        else:
            print("  PASS：cat 后时间维 = %d，与 filter_by_for_lists 的语义一致" % T, flush=True)

        # ---------- [B] 同权重同 action -> rho == 1 ----------
        print("\n=== [B] 同权重、同 state、同 action -> rho 必须恒为 1 ===", flush=True)
        behavior_lp = {}
        for f in ("action_type", "delay", "queue"):
            behavior_lp[f] = logprob(cat_field(behaviors, f), acts[f], f)

        # 用同一份权重再走一遍（argmax 固定，所以 action 与 logits 都应完全一致）
        agent2 = Agent()
        agent2.to(DEV)
        agent2.model.load_state_dict(agent.model.state_dict())
        agent2.model.eval()
        srv2 = make_server(agent2, n=1)
        behaviors2, actions2 = collect(srv2, feats)
        acts2 = flatten_actions(actions2)

        same_action = all(acts[k].tolist() == acts2[k].tolist() for k in acts)
        print("  两次收集的 action 完全一致：%s" % same_action, flush=True)

        for f in ("action_type", "delay", "queue"):
            target_lp = logprob(cat_field(behaviors2, f), acts[f], f)
            rho = RA.compute_cliped_importance_weights(target_lp, behavior_lp[f])
            mx = float(rho.max())
            mn = float(rho.min())
            verdict = "PASS" if abs(mx - 1.0) < 1e-6 and abs(mn - 1.0) < 1e-6 else "FAIL"
            print("  [%s] log_prob %s，rho ∈ [%.9f, %.9f]  %s"
                  % (f, tuple(target_lp.shape), mn, mx, verdict), flush=True)
            if verdict == "FAIL":
                ok = False

        # ---------- [C] 改权重 -> rho 必须变，且全部 <= 1 ----------
        # ⚠️ 设计要点：单一种子 + 单一扰动方向可能**恰好**让所选 action 的概率全部升高，
        # 此时 exp(diff) 全 > 1、被 clamp(max=1) 全压成 1，看起来"没变" —— 第一版就踩了这个。
        # 所以这里换 3 个种子各扰动一次，任一次出现 rho < 1 即证明链路通。
        print("\n=== [C] 改权重后 rho 必须偏离 1，且全部 <= 1（clamp 不对称）===", flush=True)
        print("  说明：compute_cliped_importance_weights = clamp(exp(target-behavior), max=1.)，",
              flush=True)
        print("        所以 rho 只能 <= 1，不会 > 1。看到 '全 1' 不代表没变，", flush=True)
        print("        可能是 target 概率恰好全部更高被截断了 —— 必须看多个种子。", flush=True)

        base = {f: cat_field(behaviors, f) for f in ("action_type", "delay", "queue")}
        base_lp = {f: logprob(base[f], acts[f], f) for f in base}

        hit_below = {f: 0 for f in base}     # 出现 rho < 1 的种子数
        hit_any = {f: 0 for f in base}       # 出现任何变化的种子数
        for seed in (101, 202, 303):
            torch.manual_seed(seed)
            # 复原到原始权重再扰动，避免三次扰动叠加
            srv2.agent.model.load_state_dict(agent.model.state_dict())
            with torch.no_grad():
                for name, prm in srv2.agent.model.named_parameters():
                    if "action_type" in name or "delay" in name or "queue" in name \
                            or "resblock" in name:
                        prm.add_(torch.randn_like(prm) * 0.05 * prm.abs().mean().clamp(min=1e-6))
            beh_seed, _ = collect(srv2, feats)
            for f in base:
                tgt_lp = logprob(cat_field(beh_seed, f), acts[f], f)
                rho = RA.compute_cliped_importance_weights(tgt_lp, base_lp[f])
                mx, mn = float(rho.max()), float(rho.min())
                if mn < 1.0 - 1e-6:
                    hit_below[f] += 1
                if abs(mx - 1.0) > 1e-6 or abs(mn - 1.0) > 1e-6:
                    hit_any[f] += 1
                if mx > 1.0 + 1e-6:
                    print("  [%s][seed %d] ⚠️ rho > 1（%.6f），clamp 失效？" % (f, seed, mx),
                          flush=True)
                    ok = False
            print("  seed %d 完成" % seed, flush=True)

        for f in base:
            good = hit_below[f] >= 1
            print("  [%s] 3 个种子里：rho 出现过 <1 的有 %d 次，出现过变化的有 %d 次  %s"
                  % (f, hit_below[f], hit_any[f], "PASS" if good else "FAIL"), flush=True)
            if not good:
                ok = False

        # 复原权重（后面 [D] 用 base 的 logits，不受影响；但保持干净）
        srv2.agent.model.load_state_dict(agent.model.state_dict())

        # ---------- [D] units head 的 mask 是**硬要求**，不是可选项 ----------
        print("\n=== [D] units head 必须传 mask（不传会直接炸）===", flush=True)
        units_lg = cat_field(behaviors, "units")   # [T, max_sel, max_entities]
        print("  units logits shape = %s（三维，log_prob 内部会 view 成二维再按 select_size 展开 mask）"
              % (tuple(units_lg.shape),), flush=True)
        try:
            RA.log_prob(units_lg, acts["units"], [None, None, None], MAX_SEL)
            print("  FAIL：不传 mask 竟然没炸，需要人工确认 units 是否算对了", flush=True)
            ok = False
        except Exception as e:
            print("  PASS：抛出 %s —— 证实 units 必须有 mask" % type(e).__name__, flush=True)
            print("       真实训练里由 get_useful_masks(select_units_num, entity_num) 提供，", flush=True)
            print("       所以 Trajectory 里 player_select_units_num 与 entity_num **两个字段都不能缺**。",
                  flush=True)

    print("\n==> T8 结论：%s" % ("behavior_logits 可被 loss 直接消费，链路打通"
                                if ok else "有问题，见上面 FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
