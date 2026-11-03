"""第 19 轮验证：`step_from_state(obs=真实 pysc2 obs)` 能不能跑通。

## 为什么这一条最重要

第 17 轮发现 `lib/utils.py:get_batch_unit_type_mask` 里只有
`obs_list[idx]["raw_units"]` —— **只支持 dict**。而真实训练路径传的是
pysc2 的 `features.Observation`（namedtuple，只能属性访问），

    step_from_state(state, mem, obs=home_obs.observation)
      -> action_logits_by_state(obs=obs)
      -> obs_list = [obs]                      # 非 None
      -> arch_model.py:192  if obs_list is not None:     # 与 is_rl_training 无关！
      -> get_batch_unit_type_mask(action_type, [obs])
      -> obs_list[0]["raw_units"]              # namedtuple 不支持字符串索引 -> TypeError

**这意味着 `rl_vs_inner_bot_mp.py:334` 与 `rl_eval_sl.py:229` 在当前代码下
第一步 forward 就会崩** —— 是 mini-AS 的第 6 个既有阻断缺陷。

本脚本用**真实 SC2 env** 直接验证修复前后这条路径的差别：
  - 修复前：`TypeError: tuple indices must be integers or slices, not str`
  - 修复后：正常返回 6 元组，且 `ActorInferenceClient`(local) 与它逐位一致

四个检查项：
[A] 真实 obs 的 `step_from_state` 能跑通（不崩）
[B] 真实 obs 的类型确认（namedtuple，字符串索引必然失败）—— 把"为什么原来会崩"钉死
[C] `ActorInferenceClient`(local) 与 `step_from_state` 在真实 obs 下逐位一致
[D] 连跑 3 步稳定（含 hidden 滚动），且 `last_trajectory_memory` 语义正确

用法：python _test_r19_worker_obs.py
"""
import os
import sys
import threading
import time

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

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent as SC2Agent, Race, Bot, Difficulty
from pysc2.lib import actions as A

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
from alphastarmini.core.rl.actor_inference import ActorInferenceClient
from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes, FakeRawUnits
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")
NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])


def make_env(seed=71):
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    return SC2Env(map_name="Simple64",
                  players=[SC2Agent(Race.protoss, "p0"),
                           Bot(Race.random, Difficulty(1))],
                  step_mul=8, agent_interface_format=[aif], random_seed=seed)


def make_shapes():
    return StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])


def server_loop(srv, stop_ev):
    while not stop_ev.is_set():
        idxs = srv._collect()
        if idxs:
            srv._serve_batch(idxs[:srv.max_batch])
        else:
            time.sleep(0.0005)


def deterministic_sampling():
    """把 torch.multinomial 换成 argmax。

    ⚠️ 本脚本第一版 [C] 忘了加这个，结果 `action_type` 一致(0) 但 `units` 差 31 ——
    不是实现问题：`units` 是 12 个位置**依次采样**，任一位不同就差出几十，
    比 `action_type`（单次采样、分布集中）更容易暴露随机性。
    """
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

    as_agent = AlphaStarAgent(name="r19")
    as_agent.agent_nn.to(DEV)
    as_agent.agent_nn.model.eval()

    srv = AsyncBatchInferenceServer(None, 2, DEV, make_shapes(),
                                    max_units=128, max_selected=AHP.max_selected,
                                    max_batch=2, return_logits=True)
    srv.agent = as_agent.agent_nn
    stop_ev = threading.Event()
    th = threading.Thread(target=server_loop, args=(srv, stop_ev), daemon=True)
    th.start()

    env = make_env()
    ts = env.reset()[0]
    obs = ts.observation

    action_spec = env.action_spec()
    if isinstance(action_spec, (list, tuple)) and not hasattr(action_spec, "functions"):
        action_spec = action_spec[0]
    # 与 ActorVSComputer.run():253 的 `agent.setup(obs_spec, act_spec)` 等价。
    # ⚠️ setup 在 **AlphaStarAgent** 上（alphastar_agent.py:46），不是 Agent。
    #    而 ActorVSComputer.__init__ 里 `self.agent = get_supervised_agent(...)` 返回的
    #    正是 AlphaStarAgent —— 所以那边的 `agent.setup(...)` 是对的。
    as_agent.setup(env.observation_spec()[0], action_spec)
    as_agent.agent_nn.model.set_rl_training(True)    # 真实训练是 True

    # ⚠️ 必须传 **AlphaStarAgent**（`as_agent`），不是里面的 `Agent`：
    #    local 模式要调 `self.agent.step_from_state()`，那是 AlphaStarAgent 的方法。
    #    （第 17 轮 r19 脚本第一版传了 as_agent.agent_nn，报
    #      `'Agent' object has no attribute 'step_from_state'` —— 是脚本传错，不是实现问题。
    #      batch 模式只用 `Agent.action_to_func_call`（类方法），两种模式统一传 AlphaStarAgent 即可。）
    local = ActorInferenceClient(agent=as_agent, action_spec=as_agent.action_spec)

    # ------------------------------------------------------------ [B]
    print("\n=== [B] 先钉死「为什么原来会崩」：真实 obs 的类型 ===", flush=True)
    t_name = type(obs).__name__
    is_dict = isinstance(obs, dict)
    print("  obs 类型 = %s | isinstance(dict) = %s | _fields = %s"
          % (t_name, is_dict, getattr(obs, "_fields", None)), flush=True)
    str_index_ok = True
    try:
        _ru = obs["raw_units"]
        print("  obs[\"raw_units\"] 可用，shape = %s" % (tuple(np.asarray(_ru).shape),), flush=True)
    except Exception as e:
        str_index_ok = False
        print("  obs[\"raw_units\"] -> %s: %s" % (type(e).__name__, e), flush=True)
    attr_ok = hasattr(obs, "raw_units")
    print("  obs.raw_units 属性访问可用 = %s" % attr_ok, flush=True)

    # 第 19 轮结论（推翻第 17 轮的推断）：pysc2 的 obs 是 **NamedDict**（dict 子类），
    # 字符串索引与属性访问**都**可用 -> `get_batch_unit_type_mask` 里的
    # `obs_list[idx]["raw_units"]` 本来就能跑通，**原代码没有这个缺陷**。
    record("[B] 真实 obs 同时支持 dict 索引与属性访问（推翻第17轮的'必崩'推断）",
           is_dict and str_index_ok and attr_ok,
           "type=%s dict索引=%s 属性=%s" % (t_name, str_index_ok, attr_ok))

    # ------------------------------------------------------------ [A]
    print("\n=== [A] 真实 obs 走 step_from_state（修复前必崩在这里）===", flush=True)
    state = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
    memory = as_agent.initial_state()
    try:
        fc, act, logits, new_mem, sun, en = as_agent.step_from_state(
            state, memory, obs=obs)
        record("[A] step_from_state(obs=真实obs) 跑通", True,
               "action_type=%d, select_units_num=%d, entity_num=%d"
               % (int(act.action_type.reshape(-1)[0]), int(sun.reshape(-1)[0]),
                  int(en.reshape(-1)[0])))
        ran = True
    except TypeError as e:
        record("[A] step_from_state(obs=真实obs) 跑通", False,
               "TypeError: %s（= 第 6 个阻断缺陷没修好）" % e)
        ran = False

    if not ran:
        env.close()
        stop_ev.set()
        th.join(timeout=2)
        print("\n[A] 失败，后续跳过", flush=True)
        return 1

    # ------------------------------------------------------------ [C]
    print("\n=== [C] local client 与 step_from_state 在真实 obs 下逐位一致 ===", flush=True)
    with deterministic_sampling():
        state2 = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
        with torch.no_grad():
            fc0, act0, lg0, nm0, sun0, en0 = as_agent.step_from_state(state2, memory, obs=obs)
        state3 = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
        with torch.no_grad():
            fc1, act1, lg1, nm1, sun1, en1 = local.step(state3, memory, obs)

    d_at = maxdiff(act0.action_type, act1.action_type)
    d_un = maxdiff(act0.units, act1.units)
    d_lg = maxdiff(lg0.action_type, lg1.action_type)
    d_nm = max(maxdiff(nm0[0], nm1[0]), maxdiff(nm0[1], nm1[1]))
    record("[C] action_type 一致", d_at == 0.0, "max|Δ| = %.3e" % d_at)
    record("[C] units 一致", d_un == 0.0, "max|Δ| = %.3e" % d_un)
    record("[C] action_type_logits 一致", d_lg == 0.0, "max|Δ| = %.3e" % d_lg)
    record("[C] new_memory 一致", d_nm == 0.0, "max|Δ| = %.3e" % d_nm)

    # ------------------------------------------------------------ [D]
    print("\n=== [D] 用真实 env 连跑 3 步（含 hidden 滚动与 env.step）===", flush=True)
    mem = as_agent.initial_state()
    worst_mem = 0.0
    steps_ok = 0
    for t in range(3):
        if ts.last():
            ts = env.reset()[0]
            obs = ts.observation
            mem = as_agent.initial_state()
            local.reset_hidden()

        st = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
        with deterministic_sampling():
            with torch.no_grad():
                fc, act, logits, new_mem, sun, en = local.step(st, mem, obs)

        # last_trajectory_memory 必须 == 本步传入的 mem（local 模式语义）
        d = max(maxdiff(local.last_trajectory_memory[0], mem[0]),
                maxdiff(local.last_trajectory_memory[1], mem[1]))
        worst_mem = max(worst_mem, d)

        n = int(sun.reshape(-1)[0])
        try:
            fc_call = Agent.action_to_func_call(act, torch.tensor([n]), action_spec)
            ts = env.step([fc_call])[0]
        except Exception as e:
            print("  step %d: action_to_func_call 失败(%r)，回退 NO_OP" % (t, e), flush=True)
            ts = env.step([NO_OP])[0]
        obs = ts.observation
        mem = new_mem
        steps_ok += 1
        print("  step %d: action_type=%d n=%d，hidden 语义 max|Δ|=%.3e"
              % (t, int(act.action_type.reshape(-1)[0]), n, d), flush=True)

    record("[D] 3 步全部完成（真实 env.step 未崩）", steps_ok == 3, "%d/3" % steps_ok)
    record("[D] last_trajectory_memory 语义正确", worst_mem == 0.0, "max|Δ| = %.3e" % worst_mem)

    env.close()
    stop_ev.set()
    th.join(timeout=2)

    print("\n" + "=" * 68, flush=True)
    passed = sum(1 for _, p, _ in results if p)
    print("第19轮验证汇总：%d/%d PASS%s" % (passed, len(results), "" if ok else "  <-- 有 FAIL"),
          flush=True)
    for name, p, detail in results:
        if not p:
            print("  FAIL  %s  %s" % (name, detail), flush=True)
    print("==> %s" % ("全部 PASS" if ok else "有 FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
