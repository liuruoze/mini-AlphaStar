"""第 15 轮验证：ActorInferenceClient 的「本地 forward」与「中心批推理」必须等价。

接线层的价值完全取决于一件事：**换成批推理后，actor 产出的东西一模一样**。
只要这里不等价，后面接 Worker 就是错的。

七个检查项：

[A] 单步动作一致：6 个 head 的采样值（action_type/delay/queue/units/target_unit/target_location）
[B] 采样前的量一致：action_type_logits + select_units_num + entity_num
[C] 多步 hidden 演化一致：4 步，每步的「本步输入 hidden」必须逐位相等
    （这是本轮最容易错的地方 —— batch_server 的 self.h 会被就地推进，
      不另存 h_prev 的话 actor 读到的是**超前一步**的 hidden，不崩不报错）
[D] `last_trajectory_memory` 语义：local 模式必须 == 调用时传入的 memory
[E] `reset_hidden()` 生效（清零后 hidden 全 0）
[F] 防御：server 没开 return_logits 时构造 client 应直接报错，而不是运行时才炸
[G] 多 actor 隔离：两个 actor 共用一个 server，actor0 的结果不受 actor1 影响

设计说明：
- 用**线程**跑 server 循环（同进程内 Event / 共享内存都能用），避免多进程调试复杂度；
- 把 `Agent.action_to_func_call` 临时替换成桩函数：它需要真实 pysc2 `action_spec`，
  而我们只测「推理结果是否一致」。**这是合理的**——func_call 是 action 的纯函数，
  action 一致则 func_call 必然一致（两者共用同一个 `action_to_func_call`）。
- 采样固定为 argmax（`deterministic_sampling`），否则自回归的随机性会让比对必然失败。

用法：python _test_r15_client.py
"""
import glob
import os
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

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
from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
from alphastarmini.core.rl.actor_inference import ActorInferenceClient, from_env
from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes
from alphastarmini.core.sl.feature import Feature
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")
MAX_UNITS = 128
MAX_ENT = AHP.max_entities
MAX_SEL = AHP.max_selected

ACTION_FIELDS = ("action_type", "delay", "queue", "units", "target_unit", "target_location")


# --------------------------------------------------------------------- 工具
def deterministic_sampling():
    """把 torch.multinomial 换成 argmax，消除自回归采样的随机性。"""
    class _Ctx(object):
        def __enter__(self):
            self.orig = torch.multinomial
            def _argmax(inp, num_samples, replacement=False, **kw):
                if inp.dim() == 1:
                    return torch.argmax(inp).reshape(1)
                return torch.argmax(inp, dim=-1, keepdim=True)
            torch.multinomial = _argmax
            return self

        def __exit__(self, *a):
            torch.multinomial = self.orig
            return False
    return _Ctx()


class FakeObs(object):
    """只提供 client 需要的 `.raw_units`（用于抽 unit_type 列）。

    ⚠️ **第 17 轮修正：不能全填 0。**
    原来 `np.zeros((n, 21))` 让每一列都是 0，于是"取第 0 列（unit_type）"与
    "取第 1 列（alliance）"结果完全相同 —— 这个测试数据把
    `actor_inference.py` 里 `[:, 1]` 的**取错列 bug 完全掩盖了**。

    现在故意让各列取不同值：
      col 0 = unit_type  -> 1..200（合法单位类型）
      col 1 = alliance   -> 0..3（完全不同的量级）
    这样只要有一方取错列，mask 就会不同 -> logits 不同 -> 测试必然 FAIL。
    """
    def __init__(self, n=64):
        self.raw_units = np.zeros((n, 21), dtype=np.int64)
        idx = np.arange(n)
        self.raw_units[:, 0] = idx % 200 + 1      # FeatureUnit.unit_type（真实值 0）
        self.raw_units[:, 1] = idx % 4            # FeatureUnit.alliance（真实值 1）


def load_states(n):
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    out = []
    for i in range(n):
        out.append(Feature.feature2state(feats[i:i + 1]))
    return out


def make_shapes():
    return StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])


def server_loop(srv, stop_ev):
    """同进程内模拟 server 的调度循环（生产环境是独立进程）。"""
    while not stop_ev.is_set():
        idxs = srv._collect()
        if idxs:
            srv._serve_batch(idxs[:srv.max_batch])
        else:
            time.sleep(0.0005)


def same_tensor(a, b):
    # 统一搬到 CPU 比较：两种模式的 device 语义已在 client 里对齐，
    # 但比较本身不该对 device 敏感（否则测的是部署细节而不是数值）。
    a = torch.as_tensor(a).detach().cpu()
    b = torch.as_tensor(b).detach().cpu()
    return tuple(a.shape) == tuple(b.shape) and float((a - b).abs().max()) == 0.0


def maxdiff(a, b):
    a = torch.as_tensor(a).detach().cpu()
    b = torch.as_tensor(b).detach().cpu()
    return float((a - b).abs().max())


def main():
    torch.manual_seed(20261102)
    ok = True
    results = []

    def record(name, passed, detail=""):
        nonlocal ok
        results.append((name, passed, detail))
        if not passed:
            ok = False
        print("  %-6s %s%s" % ("PASS" if passed else "FAIL", name,
                               ("  " + detail) if detail else ""), flush=True)

    # 桩掉 action_to_func_call（需要真实 action_spec，且它只是 action 的纯函数）
    _orig_atfc = Agent.action_to_func_call
    Agent.action_to_func_call = staticmethod(
        lambda action, select_units_num, action_spec, use_random_args=False:
        ("STUB_FC", int(action.action_type.reshape(-1)[0])))
    print("已桩掉 Agent.action_to_func_call（func_call 是 action 的纯函数，不参与比对）", flush=True)

    states = load_states(6)
    feats = None

    as_agent = AlphaStarAgent(name="r15")
    as_agent.agent_nn.to(DEV)
    # ⚠️ 必须 eval()：`model.train()` 会打开 dropout，每次 forward 的 dropout mask 都随机，
    #    同一份输入跑两次结果都不同（见下面的 [I] 对照实验），根本没法比"实现是否等价"。
    #    第 10 轮已经取消注释了 learner.py:218 的 model.train()，所以真实训练时 dropout 是开的
    #    —— 但那属于"批内共享同一次 dropout 采样"的正常 batch 语义，与 batch acting 无关。
    as_agent.agent_nn.model.eval()
    as_agent.set_rl_training(False)

    srv = AsyncBatchInferenceServer(None, 2, DEV, make_shapes(),
                                    max_units=MAX_UNITS, max_selected=MAX_SEL,
                                    max_batch=2, return_logits=True)
    srv.agent = as_agent.agent_nn
    stop_ev = threading.Event()
    th = threading.Thread(target=server_loop, args=(srv, stop_ev), daemon=True)
    th.start()

    local = ActorInferenceClient(agent=as_agent, action_spec=None)
    cli0 = ActorInferenceClient(agent=as_agent, action_spec=None, srv=srv, idx=0)
    cli1 = ActorInferenceClient(agent=as_agent, action_spec=None, srv=srv, idx=1)
    print("client: %r / %r / %r" % (local, cli0, cli1), flush=True)

    obs = FakeObs()

    # ---------- [H] 先诊断 dtype：batch_server.write_state 用 .float() 写共享内存，
    # 而 local 路径直接用 preprocess_state_all 产出的原始 dtype。若两者不同，
    # 差异**不是**实现的 bug，而是"精度不同" —— 必须先量出来再决定怎么处理。
    print("\n=== [H] 诊断：state 各段的真实 dtype ===", flush=True)
    _st = states[0]
    print("  entity_state   :", _st.entity_state.dtype, tuple(_st.entity_state.shape), flush=True)
    _dt = {}
    for _i, _x in enumerate(_st.statistical_state):
        _dt[str(_x.dtype)] = _dt.get(str(_x.dtype), 0) + 1
    print("  statistical    :", _dt, flush=True)
    print("  map_state      :", _st.map_state.dtype, tuple(_st.map_state.shape), flush=True)

    from alphastarmini.core.rl.state import MsState

    def to_f32(st):
        return MsState(entity_state=st.entity_state.float(),
                       statistical_state=[x.float() for x in st.statistical_state],
                       map_state=st.map_state.float())

    with deterministic_sampling():
        # ------------------------------------------------ [A][B][D] 单步
        print("\n=== [A][B][D] 单步：local vs batch（actor0）===", flush=True)
        h0 = as_agent.initial_state()
        st0 = states[0]

        # 先量"原始 dtype 的 local"与 batch 的差（这一项用来判断精度影响有多大）
        _, _, lg_raw, _, _, _ = local.step(st0, h0, obs)

        # 再用 float32 的 state 对比（batch 路径本来就是 float32）
        st0f = to_f32(st0)
        fc_l, act_l, lg_l, mem_l, sun_l, en_l = local.step(st0f, h0, obs)
        fc_b, act_b, lg_b, mem_b, sun_b, en_b = cli0.step(st0f, h0, obs)

        d_raw = maxdiff(lg_raw.action_type, lg_b.action_type)
        print("  [H] local(原始 dtype) vs batch(float32) 的 action_type_logits 差 = %.3e" % d_raw,
              flush=True)
        print("      若这一项明显大于下一项，说明差异来自 dtype 降精度，不是实现错误", flush=True)

        # [A] 6 个 head 的采样值
        bad = []
        for f in ACTION_FIELDS:
            if not same_tensor(getattr(act_l, f), getattr(act_b, f)):
                bad.append(f)
        record("[A] 6 个 head 的 action 逐位一致", not bad, "不一致: %s" % (bad or "无"))

        # [B] 采样前量
        d_lg = maxdiff(lg_l.action_type, lg_b.action_type)
        record("[B] action_type_logits 一致", d_lg == 0.0, "max|Δ| = %.3e" % d_lg)
        record("[B] select_units_num 一致", int(sun_l.reshape(-1)[0]) == int(sun_b.reshape(-1)[0]),
               "local %d vs batch %d" % (int(sun_l.reshape(-1)[0]), int(sun_b.reshape(-1)[0])))
        record("[B] entity_num 一致", int(en_l.reshape(-1)[0]) == int(en_b.reshape(-1)[0]),
               "local %d vs batch %d" % (int(en_l.reshape(-1)[0]), int(en_b.reshape(-1)[0])))

        # [D] local 的 last_trajectory_memory 必须就是传入的 memory
        d_m = max(maxdiff(local.last_trajectory_memory[0], h0[0]),
                  maxdiff(local.last_trajectory_memory[1], h0[1]))
        record("[D] local 的 last_trajectory_memory == 传入的 memory", d_m == 0.0,
               "max|Δ| = %.3e" % d_m)

        # batch 的第一份「本步输入」也应该是 h0（server 初始为 0，不是 h0！）
        # -> 所以这里**不应该**相等；真正该比的是「清零后从 0 开始」的演化，见 [C]
        print("  提示：batch 的 hidden 由 server 从 0 起步，与 local 传 h0 起步不同，故 [C] 统一从 0 开始",
              flush=True)

        # ------------------------------------------------ [C] 多步 hidden 演化
        print("\n=== [C] 4 步 hidden 演化：local vs batch（都从 0 hidden 起步）===", flush=True)
        z = (torch.zeros(1, 1, 128, device=DEV), torch.zeros(1, 1, 128, device=DEV))
        srv.reset_hidden(0)
        h_l = z
        worst_mem = 0.0
        worst_act = 0.0
        for t in range(4):
            st = states[t % len(states)]
            _, a_l, _, nh_l, _, _ = local.step(st, h_l, obs)
            _, a_b, _, _, _, _ = cli0.step(st, h_l, obs)   # 传 h_l 只为接口一致，batch 模式内部忽略

            # 本步输入 hidden 必须一致
            d = max(maxdiff(local.last_trajectory_memory[0], cli0.last_trajectory_memory[0]),
                    maxdiff(local.last_trajectory_memory[1], cli0.last_trajectory_memory[1]))
            worst_mem = max(worst_mem, d)

            # 本步动作也必须一致
            d_act = max(maxdiff(getattr(a_l, f), getattr(a_b, f))
                        for f in ACTION_FIELDS if getattr(a_l, f).numel() > 0)
            worst_act = max(worst_act, d_act)

            print("  step %d: hidden max|Δ| = %.3e | action max|Δ| = %.3e" % (t, d, d_act), flush=True)
            h_l = nh_l                                     # local 滚动
            # batch 的滚动由 server 内部完成，无需动作

        record("[C] 4 步「本步输入 hidden」逐位一致", worst_mem == 0.0, "max|Δ| = %.3e" % worst_mem)
        record("[C] 4 步动作逐位一致", worst_act == 0.0, "max|Δ| = %.3e" % worst_act)

        # local 滚动到第 4 步的 hidden 应与 batch 第 4 步的「本步输入」一致
        # （说明 server 内部的滚动推进与 local 的 new_memory 完全同步）
        d_final = max(maxdiff(h_l[0], cli0.last_trajectory_memory[0]),
                      maxdiff(h_l[1], cli0.last_trajectory_memory[1]))
        # 注意：h_l 是第 4 步之后的 new hidden，而 last_trajectory_memory 是第 4 步输入，
        # 两者相差一步，所以这里**不该相等**；用下一句做正确断言
        record("[C] 步进关系正确（local 的 new != batch 的本步输入，差一步）", d_final > 0.0,
               "差 %.3e（若为 0 反而说明步进没发生）" % d_final)

        # ------------------------------------------------ [E]
        print("\n=== [E] reset_hidden 生效 ===", flush=True)
        srv.reset_hidden(0)
        hh, cc = srv.read_hidden(0)
        record("[E] reset 后 read_hidden 全 0",
               float(hh.abs().max()) == 0.0 and float(cc.abs().max()) == 0.0,
               "h %.3e / c %.3e" % (float(hh.abs().max()), float(cc.abs().max())))

        # ------------------------------------------------ [G]
        print("\n=== [G] 多 actor 隔离 ===", flush=True)
        srv.reset_hidden(0)
        srv.reset_hidden(1)
        # 先单独跑 actor0，记录结果
        _, a0_alone, _, _, _, _ = cli0.step(states[0], z, obs)
        # 再把 actor0/actor1 交错各跑一步（actor1 用不同 state）
        srv.reset_hidden(0)
        srv.reset_hidden(1)
        _, a0_mixed, _, _, _, _ = cli0.step(states[0], z, obs)
        _, a1_mixed, _, _, _, _ = cli1.step(states[3], z, obs)
        d_iso = max(maxdiff(getattr(a0_alone, f), getattr(a0_mixed, f))
                    for f in ACTION_FIELDS if getattr(a0_alone, f).numel() > 0)
        record("[G] actor0 的结果不受 actor1 影响", d_iso == 0.0, "max|Δ| = %.3e" % d_iso)

    # ---------------------------------------------------- [J] 真实训练路径
    # 第 17 轮新增：前面 [A]~[G] 全部在 `set_rl_training(False)` 下跑的，
    # 而**真实训练是 True**（`learner.py` 里会 set_rl_training(True)）。
    # 这个差别不是小事：`get_batch_unit_type_mask` 只在 True 时才被调用，
    # 也就是说 obs_list / unit_type 列只在 True 下才参与计算 ——
    # `actor_inference.py` 里 `[:, 1]`（取成 alliance 列）的 bug 正是靠这组才暴露。
    print("\n=== [J] is_rl_training=True（真实训练路径，obs_list 真正参与计算）===", flush=True)
    as_agent.set_rl_training(True)
    hz = (torch.zeros(1, 1, 128, device=DEV), torch.zeros(1, 1, 128, device=DEV))
    with deterministic_sampling():
        srv.reset_hidden(0)
        _, a_l2, lg_l2, _, _, _ = local.step(states[0], hz, obs)
        srv.reset_hidden(0)
        _, a_b2, lg_b2, _, _, _ = cli0.step(states[0], hz, obs)

    d_j_at = maxdiff(lg_l2.action_type, lg_b2.action_type)
    # ⚠️ 关键：`unit_type_entity_mask` 只传给 **selected_units_head**
    #    （arch_model.py:214-218），对 action_type_logits 没有影响 ——
    #    所以真正该比的是 **units 的 logits**。第一版只比 action_type，
    #    结果"一致"是必然的、什么都证明不了（见下面的灵敏度验证）。
    d_j_un = maxdiff(lg_l2.units, lg_b2.units)
    record("[J] RL 训练路径下 local vs batch 一致（action_type）", d_j_at == 0.0,
           "max|Δ| = %.3e" % d_j_at)
    record("[J] RL 训练路径下 local vs batch 一致（**units**，受 mask 影响）",
           d_j_un == 0.0, "max|Δ| = %.3e" % d_j_un)

    # 灵敏度验证：**故意**把 ut 换成 alliance 列（重现第 17 轮修掉的 bug），
    # 若 units logits 测不出差异，说明这组测试没有灵敏度、等于白测。
    srv.reset_hidden(0)
    srv.write_state(0, states[0], obs.raw_units[:, 1])   # 错列：alliance 而非 unit_type
    srv._serve_batch([0])
    bl_wrong = srv.read_behavior_logits(0)
    d_sens = maxdiff(bl_wrong.units, lg_l2.units)
    record("[J] 灵敏度：喂 alliance 列时 **units logits** 必须变化", d_sens > 0.0,
           "max|Δ| = %.3e（>0 才说明这组测试真的能抓到取错列）" % d_sens)

    # 再补一个更直接的证据：直接调 mask 函数，比对两列的产物
    from alphastarmini.lib import utils as _L
    import torch as _t
    _at = lg_l2.action_type.argmax(dim=-1).reshape(1, 1)
    _m_ok = _L.get_batch_unit_type_mask(_at, [{"raw_units": obs.raw_units}])
    _m_bad = _L.get_batch_unit_type_mask(_at, [{"raw_units": obs.raw_units[:, [1, 1]]}])
    _m_bad2 = _L.get_batch_unit_type_mask(_at, [{"raw_units": obs.raw_units[:, 1:2]}])
    d_mask = float(np.abs(_m_ok - _m_bad2).max())
    print("  [J] 直接比 mask：正确列 sum=%d，alliance 列 sum=%d"
          % (int(_m_ok.sum()), int(_m_bad2.sum())), flush=True)
    record("[J] 直接比 mask：两列产物必须不同", d_mask > 0,
           "max|Δ| = %.0f（mask 是 0/1，差 1 即说明取错列会改变 mask）" % d_mask)

    as_agent.set_rl_training(False)

    # ---------------------------------------------------- [I] 对照实验
    print("\n=== [I] 对照：train() 模式下 local 自己跑两次是否也不一致 ===", flush=True)
    as_agent.agent_nn.model.train()
    with deterministic_sampling():
        _, a_eval1, lg_e1, _, _, _ = local.step(states[0], h0, obs)
        _, a_eval2, lg_e2, _, _, _ = local.step(states[0], h0, obs)
    d_self = maxdiff(lg_e1.action_type, lg_e2.action_type)
    as_agent.agent_nn.model.eval()
    print("  [I] local 连续两次（train 模式）action_type_logits 差 = %.3e" % d_self, flush=True)
    record("[I] train 模式下 local 自身也不可复现（证实差异来自 dropout）", d_self > 0.0,
           "若为 0 说明 dropout 没生效，前面的 FAIL 另有原因")

    # ---------------------------------------------------- [F]（不需要确定性采样）
    print("\n=== [F] 防御：server 未开 return_logits 时构造应报错 ===", flush=True)
    srv2 = AsyncBatchInferenceServer(None, 2, DEV, make_shapes(),
                                     max_units=MAX_UNITS, max_selected=MAX_SEL,
                                     max_batch=2, return_logits=False)
    try:
        ActorInferenceClient(agent=as_agent, action_spec=None, srv=srv2, idx=0)
        record("[F] 未开 return_logits 应报错", False, "竟然没报错")
    except ValueError as e:
        record("[F] 未开 return_logits 应报错", True, str(e)[:60])

    # from_env 的路由
    os.environ["MAS_BATCH_ACTING"] = "0"
    record("[F] from_env 默认走 local", not from_env(as_agent, None, srv, 0).is_batch)
    os.environ["MAS_BATCH_ACTING"] = "1"
    record("[F] from_env=1 且给 srv 走 batch", from_env(as_agent, None, srv, 0).is_batch)
    record("[F] from_env=1 但没给 srv 仍走 local", not from_env(as_agent, None, None, 0).is_batch)
    os.environ["MAS_BATCH_ACTING"] = "0"

    stop_ev.set()
    th.join(timeout=2)
    Agent.action_to_func_call = _orig_atfc

    print("\n" + "=" * 68, flush=True)
    passed = sum(1 for _, p, _ in results if p)
    print("第15轮验证汇总：%d/%d PASS%s" % (passed, len(results), "" if ok else "  <-- 有 FAIL"),
          flush=True)
    for name, p, detail in results:
        if not p:
            print("  FAIL  %s  %s" % (name, detail), flush=True)
    print("==> %s" % ("全部 PASS" if ok else "有 FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
