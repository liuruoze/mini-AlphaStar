"""T6 验证：rl_eval_sl.py:218 那次多余的 step_from_state 到底有没有副作用。

背景
----
rl_eval_sl.py:218 与 :221 连续调用了两次 `step_from_state`，第一次的返回值
（`player_step`）从未被读取 —— 看起来是纯浪费，删掉能省一半推理时间。

但第七轮已经证实：**`forward` 会 in-place 修改传入的 hidden_state**
（`_audit_batched_hidden.py` 第一版就因此报了假 FAIL）。
如果单样本的 `step_from_state` 也一样，那第一次调用会把 `player_memory`
**in-place 推进一个时间步**，第二次调用就站在错误的 LSTM 状态上跑。

后果比"慢一倍"严重得多：
  - 评估时每帧 LSTM 推进 2 次而不是 1 次 -> 策略行为与训练不一致
  - 这种错误不会报错，只会让评估胜率莫名其妙地偏低

本脚本做三件事：
  [A] 检查 `action_logits_by_state(single_inference=True)` 是否 in-place 改 hidden
  [B] 连续两次 step_from_state，比较第一次前后的 hidden 与两次的 action 是否相同
  [C] 若确认有副作用，验证"只调一次"与"调两次取第二次"的结果差异
       （差异即 bug 的影响幅度）
"""
import os
import sys
import glob

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

DEV = torch.device("cuda:0")


def load_state(i=0):
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    return Feature.feature2state(feats[i:i + 1].float())


def main():
    torch.manual_seed(3)
    agent = Agent()
    agent.to(DEV)
    agent.model.eval()

    st = load_state(0)
    st = type(st)(entity_state=st.entity_state.float().to(DEV),
                  statistical_state=[x.float().to(DEV) for x in st.statistical_state],
                  map_state=st.map_state.float().to(DEV))

    h0 = torch.zeros(1, 1, 128, device=DEV)
    c0 = torch.zeros(1, 1, 128, device=DEV)

    ok = True

    # ---------- [A] 单样本前向是否 in-place 改 hidden ----------
    print("=== [A] action_logits_by_state 是否 in-place 修改 hidden ===", flush=True)
    h = h0.clone()
    c = c0.clone()
    with torch.no_grad():
        out = agent.batch_action_logits_by_state if False else None
        al, acts, nh, sun, en = agent.model.forward(
            st, batch_size=1, sequence_length=1, hidden_state=(h, c),
            return_logits=True, obs_list=None)
    d_h = float((h - h0).abs().max())
    d_c = float((c - c0).abs().max())
    print("  调用前 h 全 0；调用后 h 的 max|Δ| = %.3e, c 的 max|Δ| = %.3e" % (d_h, d_c), flush=True)
    if d_h == 0.0 and d_c == 0.0:
        print("  -> PASS：forward **没有** in-place 改 caller 的 hidden（内部做了 clone）", flush=True)
        print("     那么 rl_eval_sl.py:218 的重复调用只是性能浪费，不影响正确性", flush=True)
    else:
        print("  -> ⚠️ 确认：forward **in-place 修改**了 caller 的 hidden", flush=True)
        print("     这意味着 rl_eval_sl.py:218 的那次多余调用会推进 LSTM 状态 -> 真 bug", flush=True)
        ok = False

    # ---------- [B] 连续两次 step_from_state ----------
    print("\n=== [B] 连续两次 step_from_state（复现 rl_eval_sl.py:218/221 的调用序列）===", flush=True)
    h = h0.clone()
    c = c0.clone()
    mem = (h, c)
    with torch.no_grad():
        r1 = agent.step_from_state(st, mem)
        mem_after_1 = (mem[0].clone(), mem[1].clone()) if not isinstance(mem, tuple) else (mem[0].clone(), mem[1].clone())
        r2 = agent.step_from_state(st, mem)

    a1, a2 = r1[1], r2[1]
    same_at = int(a1.action_type.reshape(-1)[0]) == int(a2.action_type.reshape(-1)[0])
    print("  第 1 次 action_type = %d" % int(a1.action_type.reshape(-1)[0]), flush=True)
    print("  第 2 次 action_type = %d" % int(a2.action_type.reshape(-1)[0]), flush=True)
    print("  两次 action_type 相同: %s（采样有随机性，不同不代表有问题）" % same_at, flush=True)

    # 关键：把 mem 传进去之后，mem 里的张量有没有变
    print("  第 1 次调用后 mem[0] 的 max|Δ| = %.3e"
          % float((mem[0] - h0).abs().max()), flush=True)

    # ---------- [C] 用相同 hidden 直接比 logits ----------
    print("\n=== [C] 同一 state、同一 hidden 下：单次 vs 连续两次的 logits 对比 ===", flush=True)
    # 单次
    hh = h0.clone(); cc = c0.clone()
    with torch.no_grad():
        al_once = agent.model.forward(st, batch_size=1, sequence_length=1,
                                      hidden_state=(hh.clone(), cc.clone()),
                                      return_logits=True, obs_list=None)[0]
    # 连续两次（第一次用 h0，第二次用被污染的 hidden）
    hh2 = h0.clone(); cc2 = c0.clone()
    with torch.no_grad():
        _ = agent.model.forward(st, batch_size=1, sequence_length=1,
                               hidden_state=(hh2, cc2), return_logits=True, obs_list=None)
        al_twice = agent.model.forward(st, batch_size=1, sequence_length=1,
                                       hidden_state=(hh2, cc2), return_logits=True, obs_list=None)[0]
    d_at = float((al_once.action_type - al_twice.action_type).abs().max())
    print("  单次 vs 连续两次：action_type logits 的 max|Δ| = %.3e" % d_at, flush=True)
    if d_at == 0.0:
        print("  -> 完全一致：说明 hidden 未被污染（或污染不影响结果）", flush=True)
    else:
        print("  -> ⚠️ 有差异：连续两次调用会让第二次的输入 hidden 与单次不同", flush=True)

    print("\n==> T6 结论：%s" % ("重复调用只是性能问题" if ok and d_at == 0.0
                                else "重复调用同时是**正确性**问题，必须修"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
