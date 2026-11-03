#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""actor 侧的推理路由：本地 forward ↔ 中心批推理，同一接口。

## 为什么需要这一层

`rl_vs_inner_bot_mp.py:329-332` 的 Worker 里写死了
    player_function_call, player_action, player_logits, player_new_memory,
        player_select_units_num, entity_num = self.agent.step_from_state(...)
要把 batch acting 接进训练，只有这一处需要换，但**不能**直接把
`batch_server.submit_and_get()` 塞进去，原因有两个：

1. `submit_and_get()` 只返回**采样后的 int 值**（7 元组），而训练需要
   `ArgsAction` 张量（填 `Trajectory.action`）、`ArgsActionLogits`（填 `behavior_logits`）、
   以及**每步输入**的 hidden（填 `memory`）；
2. `action_to_func_call()` 必须留在 actor 侧 —— `arch/agent.py:353` 写死
   `action.action_type.item()`，注释明说 "we assume single inference"，只能单样本调用。

所以这一层的职责就是把「批推理 + 本地组装」这两半拼起来，对外保持与
`step_from_state()` **完全一致**的 6 元组，让 Worker 只改一行。

## 用法

    client = ActorInferenceClient(agent=as_agent, action_spec=spec)          # 本地
    client = ActorInferenceClient(agent=as_agent, srv=srv, idx=rank,
                                  action_spec=spec)                          # 批推理
    fc, act, logits, new_mem, sun, en = client.step(state, memory, obs)

环境变量 `MAS_BATCH_ACTING=1` 时，`from_env()` 会自动走批推理。

## ⚠️ 两个静默错误点（都已处理，改代码时别拆掉）

- `memory` 必须是**本步 forward 之前**的 hidden（`srv.read_hidden`），
  不是 new hidden —— 后者会让 LSTM 时序整体超前一步，不崩不报错；
- episode 结束必须 `client.reset_hidden()`，否则新一局继承上一局的 LSTM 状态。

`__author__ = "DeepSeek (第 15 轮 batch acting 接线)"`
"""

import os

import numpy as np
import torch

from pysc2.lib.features import FeatureUnit

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.rl.action import ArgsAction


class ActorInferenceClient(object):
    """把「本地 single_inference」与「中心批推理」统一成同一个 6 元组接口。"""

    def __init__(self, agent, action_spec, srv=None, idx=None, device=None):
        """
        agent:        AlphaStarAgent（local 模式自己 forward；batch 模式下只用它的
                      action_to_func_call，不再自己算网络）
        action_spec:  来自 env.action_spec()（多 agent 时要先取 [0]）
        srv:          AsyncBatchInferenceServer 实例；None 表示本地模式
        idx:          本 actor 在 server 里的槽位下标（batch 模式必填）
        device:       local 模式下 state 要搬到的设备；None 时用 agent 自己的
        """
        self.agent = agent
        self.action_spec = action_spec
        self.srv = srv
        self.idx = idx
        # batch 模式产出的张量必须与 local 模式**同 device**，否则 Worker 把
        # Trajectory.action / logits / memory 混着 CPU 和 GPU 喂给 rl_unroll 会直接报错。
        # 未显式给定时跟随 agent 所在的卡（Worker 里一定有 CUDA context）。
        if device is None:
            try:
                device = agent.agent_nn.device()
            except Exception:
                device = torch.device("cpu")
        self.device = device

        # 最近一次 step() 时「本步 forward 之前」的 hidden，专供填 Trajectory.memory。
        # 两种模式下语义一致，Worker 用它就对了（见 step() 的说明）。
        self.last_trajectory_memory = None

        if srv is not None:
            if idx is None:
                raise ValueError("batch 模式必须给 idx（该 actor 在 server 里的槽位）")
            if not getattr(srv, "return_logits", False):
                raise ValueError(
                    "server 必须 return_logits=True：训练需要 behavior_logits 填 Trajectory，"
                    "否则 read_behavior_logits 会直接抛错")

    # ------------------------------------------------------------------ 模式
    @property
    def is_batch(self):
        return self.srv is not None

    def __repr__(self):
        if self.is_batch:
            return "<ActorInferenceClient batch idx=%d N=%d>" % (self.idx, self.srv.N)
        return "<ActorInferenceClient local>"

    # ------------------------------------------------------------------ 主接口
    def step(self, state, memory, obs):
        """与 `AlphaStarAgent.step_from_state` **完全同构**的 6 元组。

        返回: (func_call, action, action_logits, new_memory, select_units_num, entity_num)

        ⚠️ **`new_memory` 在两种模式下语义不同，Worker 必须区别对待**：

        | 模式 | `new_memory` 是 | Worker 该怎么做 |
        |---|---|---|
        | local | forward **之后**的 hidden（与 `step_from_state` 一致） | 照旧 `player_memory = player_new_memory` |
        | batch | 本步 forward **之前**的 hidden（server 集中维护滚动） | **不要**用它更新 `player_memory` |

        为什么 batch 模式返回「之前」而不是「之后」：
          `Trajectory.memory` 要的正是「本步输入」（`rl_vs_inner_bot_mp.py:473` 存旧值、
          `:505` 才更新），而 batch 模式下 actor 手里已经**没有** hidden 了（在 server 里），
          所以只能由 server 回传这一份；如果返回「之后」，Worker 还得再倒推一步，
          反而更容易错位。

        填 `Trajectory.memory` 请一律用 `self.last_trajectory_memory`（两种模式都对），
        不要直接用返回值 —— local 模式下返回值是「之后」的，直接填会整体超前一步。
        """
        if not self.is_batch:
            result = self.agent.step_from_state(state, memory, obs=obs)
            # local：本步输入就是调用方传进来的 memory
            self.last_trajectory_memory = memory
            return result

        # ---- 1) 提交 state 并等结果（server 在这一步做批前向） ----
        # ⚠️ 必须用 `FeatureUnit.unit_type` 取列，**不能硬编码下标**。
        #    第 17 轮实测：`FeatureUnit.unit_type` 的值是 **0**（不是 1！），
        #    这里原来写死 `[:, 1]`，实际取的是 `alliance` 列。
        #
        #    为什么测试没抓到：`get_batch_unit_type_mask` 只在
        #    `is_rl_training=True` 时才被调用；而 `_test_r15_client.py` 用的是
        #    `set_rl_training(False)`，obs_list 根本不被使用 —— **bug 潜伏在真实训练路径上**。
        #    一旦接进 Worker（会 set_rl_training(True)），unit_type mask 就会用错数据算，
        #    结果是"不崩、不报错，但策略被喂了错误的可选单位集合"。
        ut = np.asarray(obs.raw_units)[:, int(FeatureUnit.unit_type)]
        self.srv.write_state(self.idx, state, ut)
        action_7, logits = self.srv.submit_and_get(self.idx)

        # ---- 2) 本步**输入**的 hidden（必须在 submit 之后读，server 已存好） ----
        input_hidden = self.srv.read_hidden(self.idx, device=self.device)
        self.last_trajectory_memory = input_hidden

        # ---- 3) 组装 ArgsAction（形状严格照 forward 的产物，见第 9 轮踩坑记录） ----
        a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n = action_7
        action = self._build_action(a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n)

        # ---- 4) func_call 留在 actor 侧（arch/agent.py:353 只支持单样本） ----
        sun = torch.as_tensor([n], dtype=torch.int64, device=self.device)
        func_call = Agent.action_to_func_call(action, sun, self.action_spec)

        en = torch.as_tensor([self.srv.read_entity_num(self.idx)], dtype=torch.int64,
                             device=self.device)
        return func_call, action, logits, input_hidden, sun, en

    def reset_hidden(self):
        """episode 结束时调用（batch 模式必须调，local 模式由 agent 自己 reset）。"""
        if self.is_batch:
            self.srv.reset_hidden(self.idx)

    def close(self):
        """actor 退出时注销槽位（幂等）。

        第 50 轮补：原来 actor 退出后 `req[idx]` 停在 IDLE，
        server 只能靠 `idle_timeout`（`Batch_Server_Worker` 里给的是 120 s）
        兜底退出 —— 一次 30 min 的长跑结束后要**白等 2 min**，
        而且日志里会被记成「idle timeout，仍有 N 个 actor 未退出」，
        看起来像异常，实际是正常收尾。

        为什么不在 `_collect()` 里靠「进程没了」判断：
          fork 出来的子进程退出后，父进程侧的 `mp.Process.is_alive()`
          在 server 进程里拿不到（server 只持有 share_memory 张量，
          没有 Worker 的 Process 句柄）。事件通知是唯一可靠通道，
          与 `mark_exited` 的设计一致。

        `mark_exited` 只写一个 `mp.Array` 元素 + 置 Event，重复调用无副作用。
        """
        if not self.is_batch:
            return
        try:
            self.srv.mark_exited(self.idx)
        except Exception as e:
            print("[ActorInferenceClient] close() 忽略异常: %r" % (e,), flush=True)

    # ------------------------------------------------------------------ 内部
    def _build_action(self, a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n):
        """按 `forward` 的真实形状组装 ArgsAction（batch=1）。

        ⚠️ 形状不能按语义猜 —— 第 9 轮连错 5 次：
            units 是 (1, max_selected, 1)【变长的是有效前缀，宽度固定】，
            不是 (1, 1, n)；
            target_unit 是 (1, 1, 1)，不是 (1, 1)；
            target_location 是 (1, 2)，不是 (1, 1, 2)。
        这些值来自 `_probe_logits.py` 的实测。
        """
        max_sel = int(self.srv.max_selected)
        n = max(0, min(int(n), max_sel))
        d = self.device

        units = torch.zeros(1, max_sel, 1, dtype=torch.int64, device=d)
        if n > 0:
            units[0, :n, 0] = torch.as_tensor(a_units, dtype=torch.int64, device=d)[:n]

        tloc = torch.as_tensor(a_tloc, dtype=torch.int64, device=d).reshape(-1)
        if tloc.numel() < 2:
            tloc = torch.zeros(2, dtype=torch.int64, device=d)
        tloc = tloc[:2].reshape(1, 2)

        return ArgsAction(
            action_type=torch.tensor([[int(a_type)]], dtype=torch.int64, device=d),
            delay=torch.tensor([[int(a_delay)]], dtype=torch.int64, device=d),
            queue=torch.tensor([[int(a_queue)]], dtype=torch.int64, device=d),
            units=units,
            target_unit=torch.tensor([[[int(a_tunit)]]], dtype=torch.int64, device=d),
            target_location=tloc)


def from_env(agent, action_spec, srv=None, idx=None):
    """按环境变量决定走哪条路。`MAS_BATCH_ACTING=1` 且给了 srv 才走批推理。"""
    use_batch = os.environ.get("MAS_BATCH_ACTING", "0") == "1"
    if use_batch and srv is not None:
        return ActorInferenceClient(agent=agent, action_spec=action_spec, srv=srv, idx=idx)
    return ActorInferenceClient(agent=agent, action_spec=action_spec)
