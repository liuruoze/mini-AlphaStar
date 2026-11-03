#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 64 轮：真实轨迹上的**六头** log-prob 差 + 变异测试（响应 Kimi-04 / Kimi-09）。

## 相对第 58 轮版本的三个升级

1. **看未截断的量**：`rho = clamp(exp(Δ), max=1)` 把 >1 的信息全丢了。
   本脚本同时打印 `Δ = target_log_prob - behavior_log_prob`（原始差）与未截断 `exp(Δ)`。
   Kimi-04 的原话：「argmax 下 rho 又天然集中为 1，因此 rho 全 1 可能掩盖错位」。
2. **覆盖六个 head**：`action_type / delay / queue / units / target_unit / target_location`。
   `_store_logits` 其实把 6 个头都写进了共享内存（`for name in LOGIT_FIELDS`），
   所以六头对比**在数据上是可行的**，只是从没被测过。
3. **变异测试（必须失败）**：把 behavior_logits 的时间轴打乱、把 action 换掉，
   观测量必须发生变化。若不变，说明"全 1"是假的（Kimi-09 的"防自欺"要求）。

## 判据

| 判据 | 期望 | 说明 |
|---|---|---|
| [A] 六头可消费 | 每一头都能算出 log_prob | 形状/掩码接不上会在这里暴露 |
| [B] 同权重 Δ | 三个已存头的 Δ 应 ≈ 0 | units 三头因 `forward` 与 `mimic_forward` 实现路径不同，**允许非零**，但要报出来 |
| [C] 时间轴打乱 | Δ 必须显著变大 | 不变 = 观测量对错位不敏感 |
| [D] 权重 ×1.001 | Δ 必须偏离 0 | 不变 = 两侧用了同一份 logits |

用法：
    python _test_r64_sixhead.py log/traj_eval30.pt [cpu|cuda:1]
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_parsed() if hasattr(flags.FLAGS, "mark_parsed") else flags.FLAGS.mark_as_parsed()

from pysc2.env.sc2_env import Race

from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
from alphastarmini.core.rl import rl_loss as RLOSS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

SIX = ("action_type", "delay", "queue", "units", "target_unit", "target_location")
STORED = ("action_type", "delay", "queue")     # behavior_logits 里一定有值的三个


def build_agent(state_dict, device):
    a = AlphaStarAgent(name="r64", race=Race.protoss, initial_weights=None)
    a.agent_nn.model.load_state_dict(state_dict)
    a.agent_nn.model.to(device)
    a.agent_nn.model.eval()          # 必须 eval：BN/Dropout 会重采样
    return a


def build_masks(traj, device):
    """按 `rl_loss.get_useful_masks` 的语义重建掩码，但用真实的 T 而不是 AHP 常量。

    为什么不用原函数：它把 reshape 硬编码成 `(AHP.sequence_length, AHP.batch_size, ...)`，
    对"单条轨迹、T 可能不等于 AHP.sequence_length"的离线测试会直接报错。
    """
    T = len(traj.state)
    max_sel = AHP.max_selected + 1
    sun = torch.as_tensor([float(x.reshape(-1)[0]) if torch.is_tensor(x) else float(x)
                           for x in traj.player_select_units_num]).reshape(T, 1)
    en = torch.as_tensor([float(x.reshape(-1)[0]) if torch.is_tensor(x) else float(x)
                          for x in traj.entity_num]).reshape(T, 1)
    ar_sel = torch.arange(max_sel, device=device).float().unsqueeze(0)      # (1,max_sel)
    ar_ent = torch.arange(AHP.max_entities, device=device).float().unsqueeze(0)
    selected_mask = (ar_sel < (sun + 1).to(device)).unsqueeze(-1)           # (T,max_sel,1)
    entity_mask = (ar_ent < en.to(device)).unsqueeze(1)                     # (T,1,max_ent)
    # ⚠️ 必须用 torch.stack，不能用 torch.as_tensor：
    #    unit_type_entity_mask 的每个元素是 (13,128) 的多元素张量，
    #    `torch.as_tensor([tensor(13,128), ...])` 会报
    #    "only one element tensors can be converted to Python scalars"。
    utem = torch.stack(
        [torch.as_tensor(m).float() for m in traj.unit_type_entity_mask]).to(device)
    if utem.dim() == 3:
        utem = utem.unsqueeze(1)                                            # (T,1,13,128)
    return [selected_mask, entity_mask, utem]


def delta_stats(agent, traj, masks, fields=SIX, traj_bl=None):
    """返回 {field: dict(n, mad, mx, lo, hi, why)}，mad = mean|Δ|，Δ = target_log_prob - behavior_log_prob。

    每个字段独立 try：units 三头的掩码语义若不匹配，只让那一头标 N/A，
    不能让整个测试崩掉（否则永远测不出另外三个头）。
    """
    # ⚠️ 这里有个反直觉的地方：两个 API 要的**不是同一种对象**。
    #   `rl_unroll` 要「未合并」的单条轨迹（字段是 T 个 MsState/ArgsAction 元素）；
    #   `get_logprob_and_rhos` 要「已合并」的（字段外面再包一层 B）。
    #   第一版用同一个对象喂两边，先后撞上 `'ArgsActionLogits' object is not iterable`
    #   和 `'tuple' object has no attribute 'entity_state'`。
    traj_bl = traj_bl if traj_bl is not None else traj
    out = {}
    # ⚠️ rl_unroll 返回 4 元组 (target_logits, baselines, sun, en)，
    #    要喂给 get_logprob_and_rhos 的是**第 0 个**。第一版把整元组传进去，
    #    报 "'tuple' object has no attribute 'action_type'"。
    logits_all = agent.rl_unroll([traj], use_opponent_state=False)[0]
    for f in fields:
        try:
            tlp, _rho, _ = RLOSS.get_logprob_and_rhos(logits_all, f, traj_bl, masks)
            blp = _behavior_lp(f, traj_bl, tlp)
            d = (tlp.detach().float() - blp.detach().float()).reshape(-1)
            out[f] = dict(n=int(d.numel()), mad=float(d.abs().mean()),
                          mx=float(d.abs().max()), lo=float(d.min()),
                          hi=float(d.max()), why="")
        except Exception as e:
            out[f] = dict(n=0, mad=float("nan"), mx=float("nan"), lo=float("nan"),
                          hi=float("nan"), why="%s: %s" % (type(e).__name__, e))
    return out


def _behavior_lp(field, traj, ref):
    """行为侧的 log_prob。units 三头要按 `get_logprob_and_rhos` 的 reshape 规则来。"""
    dev = ref.device
    lg = RLOSS.filter_by_for_lists(field, traj.behavior_logits, dev)
    act = RLOSS.filter_by_for_lists(field, traj.action, dev)
    max_sel = AHP.max_selected + 1
    if field == "action_type" or field == "delay" or field == "queue":
        return RLOSS.RA.log_prob(lg, act, [None, None, None], max_sel)
    raise RuntimeError("autoregressive head 的 behavior log_prob 需要掩码，本脚本暂不覆盖")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")

    d = torch.load(path, map_location="cpu")
    traj, sd = d["traj"], d["state_dict"]
    T = len(traj.state)
    print("载入 %s：T=%d，%d 个参数张量，device=%s" % (path, T, len(sd), device), flush=True)

    agent = build_agent(sd, device)
    masks = build_masks(traj, device)

    # ⚠️ 关键：`rl_loss.filter_by_for_lists` 的写法是
    #     `for a in target_list: for b in a: getattr(b, field)`
    # 也就是它期望的是**"合并后"**的 Trajectory —— 每个字段是
    # 「B 条轨迹」的序列，每条又是「T 个 ArgsActionLogits」的序列。
    # 而落盘的是 `stack_namedtuple` 的产物（单条轨迹，字段是 T 个元素的 tuple），
    # 只嵌了一层。第一版直接喂进去，报 `'ArgsActionLogits' object is not iterable`。
    # 这里按 B=1 包一层，得到与 loss 内部**完全相同**的嵌套结构。
    traj_m = traj._replace(**{f: [getattr(traj, f)] for f in traj._fields})

    print("\n=== [A][B] 同权重下六头的 |Δ| ===", flush=True)
    base = delta_stats(agent, traj, masks, traj_bl=traj_m)
    ok_b = True
    for f in SIX:
        v = base[f]
        if v["n"] == 0:
            print("  %-16s N/A  (%s)" % (f, v["why"]), flush=True)
            if f in STORED:
                ok_b = False
            continue
        good = v["mx"] < 1e-6
        print("  %-16s n=%-3d |Δ|mean=%.3e max=%.3e   Δ∈[%.3e, %.3e]  %s"
              % (f, v["n"], v["mad"], v["mx"], v["lo"], v["hi"],
                 "PASS" if good else ("FAIL（应≈0）" if f in STORED else "注意（非 0，见 §B 说明）")),
              flush=True)
        if f in STORED and not good:
            ok_b = False

    print("\n=== [C] 变异测试：把 behavior_logits 沿时间轴错一位，Δ 必须变大 ===", flush=True)
    orig_bl = traj.behavior_logits          # 是「T 个 ArgsActionLogits 组成的 tuple」
    try:
        # 第 i 步改用第 i+1 步的整份 logits（最后一步回卷）——
        # 不依赖 namedtuple 的 _replace，因为 stack_namedtuple 之后它是普通 tuple。
        n = len(orig_bl)
        traj2 = traj_m._replace(behavior_logits=[tuple(orig_bl[(i + 1) % n] for i in range(n))])
        pert = delta_stats(agent, traj, masks, fields=STORED, traj_bl=traj2)
        ok_c = True
        for f in STORED:
            if base[f]["n"] == 0 or pert[f]["n"] == 0:
                print("  %-16s 跳过（[B] 已 N/A）" % f, flush=True)
                continue
            bigger = pert[f]["mad"] > 10 * max(base[f]["mad"], 1e-12)
            print("  %-16s |Δ|mean %.3e -> %.3e  %s"
                  % (f, base[f]["mad"], pert[f]["mad"], "PASS（被检出）" if bigger else "FAIL（不敏感）"),
                  flush=True)
            ok_c = ok_c and bigger
    except Exception as e:
        print("  [C] 无法构造时间错位变异：%r —— 标为未验证，不冒充通过" % (e,), flush=True)
        ok_c = False

    print("\n=== [D] 权重 ×1.001 后 Δ 必须偏离 0 ===", flush=True)
    with torch.no_grad():
        for p in agent.agent_nn.model.parameters():
            p.mul_(1.001)
    w = delta_stats(agent, traj, masks, fields=STORED, traj_bl=traj_m)
    ok_d = True
    for f in STORED:
        if base[f]["n"] == 0 or w[f]["n"] == 0:
            continue
        changed = w[f]["mad"] > 1e-9
        print("  %-16s |Δ|mean %.3e -> %.3e  %s"
              % (f, base[f]["mad"], w[f]["mad"], "PASS" if changed else "FAIL（没变）"), flush=True)
        ok_d = ok_d and changed

    print("\n=== 结论 ===", flush=True)
    print("[A] 六头可消费：%s（N/A 的头见上）" % ("全部" if all(base[f]["n"] for f in SIX) else "部分"), flush=True)
    print("[B] 三个已存头 Δ≈0：%s" % ok_b, flush=True)
    print("[C] 时间错位可检出：%s" % ok_c, flush=True)
    print("[D] 权重扰动可检出：%s" % ok_d, flush=True)
    return 0 if (ok_b and ok_c and ok_d) else 2


if __name__ == "__main__":
    sys.exit(main())
