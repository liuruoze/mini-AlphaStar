#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""批推理服务（T2）—— 多 actor 进程 + 单中心批推理。

设计依据（四份计划的共识 + 本轮实测修正）：
  1. actor 必须进程级隔离（GIL 扩展性实测 0.08x）
  2. 每个 Worker 进程内一个 BatchInferenceServer，服务本 Worker 的 N 个 actor 子进程
  3. 中心只做前向，返回 action 字段；action_to_func_call 留在 actor 侧
     （arch/agent.py:353 硬编码 .item() + "we assume single inference"）
  4. 权重同步只在批与批之间（批内换权重会让 behavior logits 与权重快照不一致）
  5. USE_BATCH_ACTING=0 时整条路径不启用，可一键回退

⚠️ 本轮修正（见 doc/BATCH_ACTING_EXECUTION_LOG.md）：
  1. 双缓冲"流水"在同步批下**无效**。原因：actor 的 preprocess 与 env.step 都发生在
     拿到 action 之后，无法与 GPU 前向重叠——之前 bench 用 time.sleep 模拟 CPU 是错的。
     → 同步批版本保留作对照；生产用 AsyncBatchInferenceServer（动态批）。
  2. **线程超订是最大的坑**：nproc=128 时每个进程的 BLAS 默认开 128 线程，
     8 个 actor = 1024 线程抢 128 核，实测 preprocess 从 10 ms 变 200 ms、吞吐 85→14 step/s。
     → 调用方必须在入口脚本 import numpy 之前设
       OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS = 1
       （rl_vs_inner_bot_mp.py 顶部与 _test_t3b_async.py 已内置）。
     → actor 子进程里再调 torch.set_num_threads(1)。

MsState 各段体积（实测，N=1，_probe_state_shape.py）：
  entity_state (1,128,1856) f32  928.0 KB
  map_state    (1,24,64,64) f32  384.0 KB
  statistical_state 16 项         62.4 KB
  合计                          1374.4 KB / 帧
"""

from time import time, sleep
import multiprocessing as mp

import numpy as np
import torch

from alphastarmini.core.rl.state import MsState

__author__ = "DeepSeek (batch acting T2)"

debug = False


def logit_dims():
    """每个 head 的 logits 形状（per-sample）。

    为什么需要它：behavior_logits 回传要在 fork **之前** 预分配共享内存
    （fork 之后分配的 buffer 子进程看不到）。所以维度必须能离线推导。

    推导值已用真实 forward 校验（_probe_logits.py，N=4）：
      action_type (4,564)  delay (4,128)  queue (4,2)
      units (4,12,128)     target_unit (4,1,128)  target_location (4,64,64)
      -> 每样本 6454 floats = 25.2 KB；24 actor 每轮 0.59 MB

    ⚠️ 注意 units 用的是 `AHP.max_selected`（=12）而不是 `max_selected + 1`：
     loss 端 rl_loss.py:398 的 `max_selected = AHP.max_selected + 1` 是它自己
     补一列 padding **之后** 的量，不是 head 的原始输出宽度。别混。
    """
    from alphastarmini.lib.hyper_parameters import Label_Size as LS
    from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
    from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
    return {
        "action_type": (int(LS.action_type_encoding),),
        "delay": (int(LS.delay_encoding),),
        "queue": (int(LS.queue_encoding),),
        "units": (int(AHP.max_selected), int(AHP.max_entities)),
        "target_unit": (1, int(AHP.max_entities)),
        "target_location": (int(SCHP.world_size), int(SCHP.world_size)),
    }


LOGIT_FIELDS = ("action_type", "delay", "queue", "units", "target_unit", "target_location")


class FakeRawUnits(object):
    """给 lib/utils.py:get_batch_unit_type_mask 用的伪 raw_units。

    该函数只访问 `obs_list[idx]["raw_units"][:, FeatureUnit.unit_type]` 一列，
    且 obs_list 里的 obs 是 dict（不是 pysc2 的 TimeStep.observation）。
    这里对任何索引都返回预先传进来的 unit_type 一维数组，避免跨进程传整个 obs
    （screen/minimap 等 ~390 KB 全用不到）。
    """

    __slots__ = ('_ut',)

    def __init__(self, unit_types):
        self._ut = unit_types

    def __getitem__(self, key):
        return self._ut


class StateShapes(object):
    """MsState 三段的形状描述，用于分配共享内存。

    由调用方用一次真实 preprocess 探测得到（_probe_state_shape.py 打印的就是这些）。
    """

    def __init__(self, entity_shape, map_shape, scalar_shapes):
        self.entity_shape = tuple(entity_shape)   # (128, 1856)
        self.map_shape = tuple(map_shape)         # (24, 64, 64)
        self.scalar_shapes = [tuple(s) for s in scalar_shapes]  # [(10,), (5,), ..., (20, 259), ...]
        self.scalar_numel = [int(np.prod(s)) if len(s) else 1 for s in self.scalar_shapes]

    def bytes_per_frame(self):
        n = int(np.prod(self.entity_shape)) * 4
        n += int(np.prod(self.map_shape)) * 4
        n += sum(self.scalar_numel) * 4
        return n


class BatchInferenceServer(object):
    """同步批推理中心。

    用法（父进程 / Worker）：
        srv = BatchInferenceServer(agent, N, device, shapes, max_units=128, max_selected=48)
        procs = srv.spawn_actors(target=actor_main, args_tail=(...))
        for step in range(steps):
            srv.serve_once()
        srv.join()

    子进程（actor）侧只需要 srv 对象（fork 继承）：
        srv.write_state(i, state, obs)
        srv.b_ready.wait()
        srv.b_done.wait()
        action, sun = srv.read_action(i)
    """

    def __init__(self, agent, num_actors, device, shapes, max_units=128, max_selected=48,
                 sync_weights_each_round=False, return_logits=False):
        self.agent = agent
        self.N = int(num_actors)
        self.device = device
        self.shapes = shapes
        self.max_units = int(max_units)
        self.max_selected = int(max_selected)
        self.sync_weights_each_round = sync_weights_each_round
        self.pending_weights = None
        self.rounds = 0

        E = shapes.entity_shape
        M = shapes.map_shape
        self.entity = torch.zeros(self.N, *E).share_memory_()
        self.map = torch.zeros(self.N, *M).share_memory_()
        self.scalars = [torch.zeros(self.N, n).share_memory_() for n in shapes.scalar_numel]
        self.obs_ut = torch.zeros(self.N, self.max_units, dtype=torch.int64).share_memory_()

        # LSTM hidden：只由 server 维护，actor 不碰
        self.h = torch.zeros(1, self.N, 128).share_memory_()
        self.c = torch.zeros(1, self.N, 128).share_memory_()

        # 本步**输入**的 hidden（第 15 轮新增，接 RL 训练必需）。
        # 为什么需要：`Trajectory.memory` 存的是「该步 forward **之前**的 hidden」——
        # 看 rl_vs_inner_bot_mp.py:473（memory=player_memory 用的是旧的）与 :505（之后才更新），
        # 而 rl_unroll（alphastar_agent.py:306-311）把它当作 mimic_forward 的初始 hidden。
        # server 的 self.h/self.c 在 _serve_batch 里被就地推进成「下一步的输入」，
        # 所以必须在 forward **之前**另存一份，否则 actor 读到的是超前一步的 hidden：
        # 不崩、不报错，但 LSTM 的时序完全错位 —— 典型的静默错误。
        self.h_prev = torch.zeros(1, self.N, 128).share_memory_()
        self.c_prev = torch.zeros(1, self.N, 128).share_memory_()

        # 结果槽（全整数，体积很小）
        self.a_type = torch.zeros(self.N, dtype=torch.int64).share_memory_()
        self.a_delay = torch.zeros(self.N, dtype=torch.int64).share_memory_()
        self.a_queue = torch.zeros(self.N, dtype=torch.int64).share_memory_()
        self.a_units = torch.zeros(self.N, self.max_selected, dtype=torch.int64).share_memory_()
        self.a_tunit = torch.zeros(self.N, dtype=torch.int64).share_memory_()
        self.a_tloc = torch.zeros(self.N, 2, dtype=torch.int64).share_memory_()
        self.a_sun = torch.zeros(self.N, dtype=torch.int64).share_memory_()
        # entity_num：**训练必需**，不是可选。
        # rl_loss.py 的 get_useful_masks(select_units_num, entity_num) 用它生成 units 的
        # entity_mask / unit_type_entity_mask；实测不给这两个字段，units 的 log_prob 直接抛
        # AttributeError（见 _test_t8_logits_to_loss.py [D]）。
        self.a_en = torch.zeros(self.N, dtype=torch.int64).share_memory_()

        self.b_ready = mp.Barrier(self.N + 1)
        self.b_done = mp.Barrier(self.N + 1)
        self.stop_ev = mp.Event()
        self.done_ev = mp.Event()

        # ---- behavior_logits 回传（接 RL 训练的前提）----
        # VTrace 需要 rho = pi_target(a|s) / pi_behavior(a|s)，
        # 而 rl_loss.py:390 的 get_logprob_and_rhos 直接读 trajectories.behavior_logits
        # 并把它当 ArgsActionLogits 用（6 个 head 的原始 logits，loss 端自己算 log_prob）。
        # 所以这里**必须回传完整 logits，不能只回传 log_prob**：
        #   - units head 在 loss 端要经过 change_units_and_logits() 按 entity_num 重排 mask，
        #     只存 log_prob 就无法重做这一步；
        #   - target_location 要做 x/y -> index 的变换，同样需要原始 logits。
        # 默认关闭（省内存与每轮 ~0.6MB 拷贝）；接 RL 训练时置 True。
        self.return_logits = bool(return_logits)
        self.logit_shapes = logit_dims() if self.return_logits else {}
        self.lg = {}
        if self.return_logits:
            for k, shp in self.logit_shapes.items():
                n = int(np.prod(shp))
                self.lg[k] = torch.zeros(self.N, n, dtype=torch.float32).share_memory_()

        self._procs = []
        self.timing = {"build": 0.0, "forward": 0.0, "write": 0.0, "wait": 0.0, "logits": 0.0, "sync": 0.0}

    # ---------------- 供 actor 子进程调用 ----------------

    def write_state(self, idx, state, unit_types):
        """actor 侧：把 MsState 与 unit_type 写进共享内存（统一 float32）。

        unit_types: 一维 numpy 整数数组，来自 obs.raw_units[:, FeatureUnit.unit_type]
        """
        self.entity[idx].copy_(state.entity_state[0].float())
        self.map[idx].copy_(state.map_state[0].float())
        for j, s in enumerate(state.statistical_state):
            self.scalars[j][idx].copy_(s.reshape(-1).float())

        ut = np.asarray(unit_types, dtype=np.int64).reshape(-1)
        n = min(ut.shape[0], self.max_units)
        self.obs_ut[idx, :n] = torch.from_numpy(ut[:n])
        if n < self.max_units:
            self.obs_ut[idx, n:] = 0

    def read_action(self, idx):
        """actor 侧：取回该 actor 的动作字段。"""
        n = int(self.a_sun[idx].item())
        n = max(0, min(n, self.max_selected))
        return (int(self.a_type[idx].item()),
                int(self.a_delay[idx].item()),
                int(self.a_queue[idx].item()),
                self.a_units[idx, :n].clone(),
                int(self.a_tunit[idx].item()),
                self.a_tloc[idx].clone(),
                n)

    def read_entity_num(self, idx):
        """actor 侧：取回该 actor 当前的 entity_num。

        为什么单独一个方法而不是塞进 read_action 的 7 元组：
        read_action 的返回值已被所有测试脚本解包，加一项会破坏兼容。
        而 entity_num 直接决定 units 的 mask，接 RL 训练时必须拿它填 Trajectory.entity_num。
        """
        return int(self.a_en[idx].item())

    def read_hidden(self, idx, device=None):
        """actor 侧：取回该 actor **本步 forward 之前**的 LSTM hidden `(h, c)`。

        形状 `(1, 1, 128)`，与 `AlphaStarAgent.initial_state()` / `step_from_state()`
        返回的 hidden 完全同构，可直接塞进 `Trajectory.memory`。

        device: 默认 None → 保持 **CPU**（共享内存的本色）。
          为什么默认 CPU 而不是自动跟 server 的卡：actor 进程在多进程部署里
          **不应该**为了读一个 hidden 就建 CUDA context（每个上下文几百 MB）。
          需要 GPU 张量时显式传 `device=`，由调用方决定。

        ⚠️ 为什么不能直接用 new hidden：`Trajectory.memory` 的语义是
        「该步输入 LSTM 的状态」（rl_vs_inner_bot_mp.py:473 用旧值、:505 才更新），
        `rl_unroll` 把它当 `mimic_forward` 的初始 hidden。给成 new hidden 会整体超前一步。

        必须在 `submit_and_wait()` 之后调用（server 已在 _serve_batch 里存好）。
        """
        h = self.h_prev[:, idx:idx + 1, :].clone()
        c = self.c_prev[:, idx:idx + 1, :].clone()
        if device is not None:
            h = h.to(device)
            c = c.to(device)
        return (h, c)

    def reset_hidden(self, idx):
        """actor 侧：episode 结束时清零自己的 hidden。

        真实训练里每个 episode 开始都要 `agent.initial_state()`；
        走批推理时 actor 不再持有 hidden，必须显式通知 server 清零，
        否则新 episode 会继承上一局的 LSTM 状态（不崩不报错的静默错误）。
        """
        self.h[:, idx, :] = 0
        self.c[:, idx, :] = 0
        self.h_prev[:, idx, :] = 0
        self.c_prev[:, idx, :] = 0

    def read_behavior_logits(self, idx):
        """actor 侧：取回该 actor 的 behavior_logits（ArgsActionLogits）。

        ⚠️ 这是 PPO/VTrace 里 `rho = pi_target / pi_behavior` 的 **分母** 来源。
        必须与产生 action 的那一次前向是**同一份 logits**，否则 rho 恒错。
        本实现中 logits 与 action 在同一次 `_serve_batch` 里由同一个 `al` 写出，
        且权重同步只在批边界发生 -> 二者严格对应。

        返回的张量在 **CPU** 上（`[1, ...]` 形状，与单样本 forward 的输出一致）。
        后续 `rl_loss.py:filter_by_for_lists` 会把它们搬到 device 再算 log_prob。
        """
        if not self.return_logits:
            raise RuntimeError(
                "构造本 server 时 return_logits=False，没有 behavior_logits 可读。"
                "接 RL 训练时必须用 return_logits=True。")
        from alphastarmini.core.rl.action import ArgsActionLogits
        d = {}
        for k, shp in self.logit_shapes.items():
            d[k] = self.lg[k][idx].reshape(1, *shp).clone()
        return ArgsActionLogits(**d)

    def _store_logits(self, al, idx_t, k):
        """server 侧：把本次前向的 6 个 head logits 写进共享内存。

        精度：源是 float32、buffer 是 float32，`.float()` 无损。
        用 `reshape(k, -1)` 平坦化以兼容 2D/3D/4D 输出，但**断言 numel 严格相等**——
        宁可报错也不能静默截断（截断会让 rho 静默算错，是那种最难查的 bug）。
        一次 `.cpu()` 搬完所有字段，避免 6 次 GPU 同步。
        """
        moved = {}
        for name in LOGIT_FIELDS:
            v = getattr(al, name)
            want = int(np.prod(self.logit_shapes[name]))
            got = int(v[0].numel())
            if got != want:
                raise RuntimeError(
                    "behavior_logits 维度不符：%s 实际每样本 %d，推导值 %d（%s）。"
                    "请更新 logit_dims()。"
                    % (name, got, want, self.logit_shapes[name]))
            moved[name] = v.detach().reshape(k, -1).float().cpu()
        for name in LOGIT_FIELDS:
            self.lg[name][idx_t] = moved[name]

    # ---------------- 供 server 主进程调用 ----------------

    def set_weights(self, state_dict):
        """参数服务器推来新权重时调用；只在批边界生效。"""
        self.pending_weights = state_dict

    def _build_state(self):
        E = self.shapes.entity_shape
        M = self.shapes.map_shape
        scalars = []
        for j, s in enumerate(self.shapes.scalar_shapes):
            t = self.scalars[j][:, :int(np.prod(s))]
            scalars.append(t.reshape(self.N, *s).to(self.device))
        return MsState(entity_state=self.entity.to(self.device),
                       statistical_state=scalars,
                       map_state=self.map.to(self.device))

    def _build_obs_list(self):
        return [{"raw_units": FakeRawUnits(self.obs_ut[i].numpy())} for i in range(self.N)]

    def serve_once(self):
        """跑一轮：等齐所有人 -> 批前向 -> 回填结果 -> 放行。"""
        t0 = time()
        self.b_ready.wait()
        self.timing["wait"] += time() - t0

        # 权重同步只允许发生在批边界（正确性硬约束）
        if self.pending_weights is not None:
            self.agent.model.load_state_dict(self.pending_weights)
            self.pending_weights = None
        elif self.sync_weights_each_round:
            pass

        t0 = time()
        st = self._build_state()
        obs_list = self._build_obs_list()
        self.timing["build"] += time() - t0

        t0 = time()
        with torch.no_grad():
            al, acts, nh, sun, en = self.agent.batch_action_logits_by_state(
                st, hidden_state=(self.h.to(self.device), self.c.to(self.device)),
                obs_list=obs_list)
        self.timing["forward"] += time() - t0

        self.h.copy_(nh[0].cpu())
        self.c.copy_(nh[1].cpu())

        t0 = time()
        self.a_type.copy_(acts.action_type.reshape(-1).cpu().long())
        self.a_delay.copy_(acts.delay.reshape(-1).cpu().long())
        self.a_queue.copy_(acts.queue.reshape(-1).cpu().long())
        self.a_units.copy_(acts.units.reshape(self.N, -1)[:, :self.max_selected].cpu().long())
        self.a_tunit.copy_(acts.target_unit.reshape(-1).cpu().long())
        self.a_tloc.copy_(acts.target_location.reshape(self.N, -1)[:, :2].cpu().long())
        self.a_sun.copy_(sun.reshape(-1).cpu().long())
        self.a_en.copy_(en.reshape(-1).cpu().long())
        self.timing["write"] += time() - t0

        self.rounds += 1
        self.b_done.wait()

    def reset_all_hidden(self):
        """把所有 actor 的 hidden 一起清零（同步版用；异步版请用 `reset_hidden(idx)`）。

        ⚠️ 名字必须是 `reset_all_hidden` 而不是 `reset_hidden`：后者已被
        第 15 轮加的**单 actor** 版本 `reset_hidden(idx)` 占用（定义在类的前部）。
        Python 类体里同名方法后定义者覆盖先定义者 —— 第一版就是因为这个名字
        撞车，`srv.reset_hidden(0)` 报 `takes 1 positional argument but 2 were given`。
        """
        self.h.zero_()
        self.c.zero_()
        if hasattr(self, "h_prev"):
            self.h_prev.zero_()
            self.c_prev.zero_()

    # ---------------- 进程管理 ----------------

    def spawn_actors(self, target, args_list):
        """启动 N 个 actor 子进程。args_list[i] 是第 i 个 actor 的额外参数 tuple。

        ⚠️ CUDA 必须在 fork 之后初始化：调用本方法前 agent 不应已经建好 CUDA context
        （或者 agent 在父进程建好、子进程只用 CPU 侧的数据搬运，本项目即如此）。

        参数会被记住，用于 T5 的 actor 崩溃重启（_restart_dead）。
        """
        self._actor_target = target
        self._actor_args = [tuple(a) for a in args_list]
        self._actor_restarts = mp.Array('i', self.N)
        self._procs = []
        for i in range(self.N):
            self._spawn_one(i)
        return self._procs

    def _spawn_one(self, i):
        p = mp.Process(target=self._actor_target, args=(self, i) + self._actor_args[i],
                       daemon=True)
        p.start()
        while len(self._procs) <= i:
            self._procs.append(None)
        self._procs[i] = p
        return p

    def restart_count(self):
        return int(sum(self._actor_restarts)) if getattr(self, "_actor_restarts", None) else 0

    def join(self, timeout=30):
        for p in self._procs:
            p.join(timeout=timeout)

    def alive(self):
        return sum(1 for p in self._procs if p.is_alive())

    def summary(self, rounds):
        t = self.timing
        tot = max(rounds, 1)
        return ("[batch_server] N=%d rounds=%d | 平均每轮: wait %.2f ms, build %.2f ms, "
                "forward %.2f ms, write %.2f ms"
                % (self.N, rounds, t["wait"] / tot * 1000, t["build"] / tot * 1000,
                   t["forward"] / tot * 1000, t["write"] / tot * 1000))


class AsyncBatchInferenceServer(BatchInferenceServer):
    """动态批（异步调度）—— 修正 T2 的核心缺陷。

    T2 用同步 barrier 的实测结论（见 doc/BATCH_ACTING_EXECUTION_LOG.md §4.5）：
      N=4 时轮次 258.7 ms，而 actor 自身只用 101.9 ms -> **利用率仅 39%**，
      其余 61% 在等最慢的 actor + 等 forward。吞吐 15~19 step/s，比现状还低。

    本类改为「谁先到谁先进批」：不再有全局 barrier，actor 提交后立即返回等待自己的完成信号，
    server 每轮只取当前已就绪的那部分 actor 做一次 forward。
    这正是 AlphaStar 原文 "requests for agent steps are batched together dynamically" 的含义。

    与父类的接口差异：
      actor 侧用  submit_and_wait(idx)  取代  b_ready.wait() + b_done.wait()
      server 侧用 run(max_rounds)       取代  for: serve_once()
      两代接口可以共存，方便 A/B 对比。
    """

    # req 状态
    IDLE, SUBMITTED, DONE, EXITED = 0, 1, 2, 3

    def __init__(self, agent, num_actors, device, shapes, max_units=128, max_selected=48,
                 max_batch=None, poll_s=0.0005, return_logits=False):
        super(AsyncBatchInferenceServer, self).__init__(
            agent, num_actors, device, shapes, max_units=max_units,
            max_selected=max_selected, return_logits=return_logits)
        self.max_batch = int(max_batch) if max_batch else self.N
        self.poll_s = poll_s
        self.applied_version = -1      # 第 65 轮：server 已应用的权重版本
        self.req = mp.Array('i', self.N)
        self.ev_done = [mp.Event() for _ in range(self.N)]
        self.stats = mp.Array('i', 6)   # [rounds, steps, max_batch_seen, sum_batch, 0, 0]

    # ---------- actor 侧 ----------

    def submit_and_wait(self, idx):
        """提交本轮 state 并阻塞到自己的结果就绪，返回动作字段。"""
        self.req[idx] = self.SUBMITTED
        self.ev_done[idx].wait()
        self.ev_done[idx].clear()
        return self.read_action(idx)

    def submit_and_get(self, idx):
        """提交并同时取回 (动作 7 元组, behavior_logits)。

        接 RL 训练时用这个而非 submit_and_wait —— VTrace 需要 behavior_logits 算 rho。
        两者是**同一次**前向的产物，所以分开调用也不会错位：
        `_serve_batch` 在放出 `ev_done` 之前已经把 action 与 logits 都写好了。
        """
        self.req[idx] = self.SUBMITTED
        self.ev_done[idx].wait()
        self.ev_done[idx].clear()
        return self.read_action(idx), self.read_behavior_logits(idx)

    def mark_exited(self, idx):
        self.req[idx] = self.EXITED

    def read_policy_version(self, idx=None):
        """server **当前已应用**的权重版本（-1 表示还没同步过）。

        第 65 轮（Kimi-05）：让 actor 能把"这一步的行为策略是哪一版"记下来。
        跨进程只能走 `mp.Value`，所以这里读的是 server 进程写进共享变量的副本
        （`applied_version` 由 `_serve_batch` 在本进程内维护，通过 `pending_version`
        的共享来源间接可见 —— 见 `Batch_Server_Worker` 里的 `srv.policy_version_ref`）。
        """
        ref = getattr(self, "policy_version_ref", None)
        if ref is not None:
            return int(ref.value)
        return int(getattr(self, "applied_version", -1))

    # ---------- server 侧 ----------

    def _collect(self):
        return [i for i in range(self.N) if self.req[i] == self.SUBMITTED]

    def _restart_dead(self, max_restarts=20):
        """T5：actor 崩溃重启。

        背景（第四轮列的盲区 4）：actor 进程可能被 SC2 崩溃 / OOM / 外部 kill 干掉，
        此时它来不及 `mark_exited`，`req[i]` 永久停在 SUBMITTED/IDLE。
        之前只靠 `idle_timeout` 兜底（整轮强制退出），那是**放弃**而不是恢复。
        长跑训练里一个 actor 挂掉就损失 1/32 吞吐并且轨迹出现空洞，必须能自愈。

        语义：重启的 actor 会重新 `env.reset()`，等于换一个全新的 actor 接上。
        它之前采样到一半的轨迹由调用方决定丢弃（本层不管轨迹）。

        返回本次重启的 actor 列表。
        """
        if getattr(self, "_actor_target", None) is None:
            return []
        restarted = []
        for i in range(self.N):
            if self.req[i] == self.EXITED:
                continue
            p = self._procs[i] if i < len(self._procs) else None
            if p is not None and p.is_alive():
                continue

            code = None if p is None else p.exitcode
            if code == 0:
                # 正常退出却没 mark_exited（调用方漏调），按完成处理，不重启
                self.req[i] = self.EXITED
                continue

            if int(self._actor_restarts[i]) >= max_restarts:
                print("[supervisor] actor %d 已重启 %d 次，放弃（exitcode=%r）"
                      % (i, int(self._actor_restarts[i]), code), flush=True)
                self.req[i] = self.EXITED
                continue

            print("[supervisor] actor %d 死亡（exitcode=%r），重启第 %d 次"
                  % (i, code, int(self._actor_restarts[i]) + 1), flush=True)
            self._actor_restarts[i] = int(self._actor_restarts[i]) + 1
            # 重置该 actor 的共享状态：新 env -> hidden 必须归零，否则会接着上一个
            # 轨迹的 LSTM 状态跑，产生与观测不匹配的 hidden（静默错误）。
            self.req[i] = self.IDLE
            self.ev_done[i].clear()
            self.h[:, i, :] = 0
            self.c[:, i, :] = 0
            # prev buffer 同步清零：新 actor 第一步 submit 时 read_hidden 拿到的是
            # 「本步输入的 hidden」，若不清零会读到上一个 actor 的残留。
            self.h_prev[:, i, :] = 0
            self.c_prev[:, i, :] = 0
            self._spawn_one(i)
            restarted.append(i)
        return restarted

    def run(self, max_rounds=None, idle_timeout=60.0, supervise=False, max_restarts=20):
        """动态批主循环。

        idle_timeout: 连续这么多秒没有任何 actor 提交请求就退出。
          ⚠️ 必须有：actor 进程可能被外部杀死（SC2 崩溃 / OOM / 用户 kill），
          这时它来不及调 mark_exited，req 会永久停在 SUBMITTED/IDLE，
          原来 `alive == 0` 的退出条件永远不成立 -> 主循环死等。
          （T4 首测就踩到了：24 个 SC2 实例跑了 4 分半不退出。）

        supervise: 开启 actor 崩溃重启（T5）。默认关，避免影响既有基准的复现性。
        """
        rounds = 0
        last_progress = time()
        # 稳态吞吐的测量点（见 async_summary 的说明）。
        # 为什么需要：墙钟吞吐 = 总步数/run() 耗时，而 run() 的前几十轮里 actor 还在
        # env.reset / 首次 preprocess，batch 凑不满 -> 前段显著偏慢。
        # 于是"跑 30 步"和"跑 200 步"会给出差很多的数字（实测 323 vs 277），
        # 这就是前几轮反复被短测误导的根因。稳态口径跳过前 50 轮，只测后半段。
        self._t_warm = None      # 第 50 轮结束的时刻
        self._s_warm = 0         # 那一刻的累计步数
        self._t_end = None
        self._s_end = 0
        while True:
            if max_rounds is not None and rounds >= max_rounds:
                break

            if supervise:
                self._restart_dead(max_restarts=max_restarts)

            idxs = self._collect()
            if not idxs:
                alive = sum(1 for i in range(self.N) if self.req[i] != self.EXITED)
                if alive == 0:
                    break
                if time() - last_progress > idle_timeout:
                    print("[async] idle timeout %.0fs，仍有 %d 个 actor 未退出，强制结束"
                          % (idle_timeout, alive), flush=True)
                    break
                sleep(self.poll_s)
                continue
            last_progress = time()
            idxs = idxs[:self.max_batch]

            try:
                self._serve_batch(idxs)
            except Exception:
                import traceback
                traceback.print_exc()
                # 出错也必须放行，否则 actor 永久阻塞
                for i in idxs:
                    self.req[i] = self.DONE
                    self.ev_done[i].set()
                continue

            rounds += 1
            self.stats[0] = rounds
            self.stats[1] += len(idxs)
            self.stats[2] = max(self.stats[2], len(idxs))
            self.stats[3] += len(idxs)
            if rounds == 50:
                self._t_warm = time()
                self._s_warm = int(self.stats[1])
            self._t_end = time()
            self._s_end = int(self.stats[1])
        return rounds

    def _serve_batch(self, idxs):
        """对给定的 actor 子集做一次批前向（batch size 可变）。"""
        k = len(idxs)
        idx_t = torch.as_tensor(idxs, dtype=torch.long)

        # ---- 第 55 轮：把「一轮 71 ms」拆开 ----
        # 之前 `timing["forward"] += time() - t0` 的 t0 取在**函数开头**，
        # 所以那个 "forward 每轮" 其实是 **build + forward 的合计**。
        # 拿它跟模型基准（A=8 时约 31 ms）对比会得出"服务器慢 2 倍"的错误结论。
        # 分段后 4 个数才能分别对上：sync(权重) / build(共享内存->device) / forward(网络) / write(回写)。
        t_s = time()
        # 权重同步只允许在批边界
        if self.pending_weights is not None:
            _pw = self.pending_weights
            # 第 65 轮（Kimi-05）：快照自带版本号，应用时一并记下，
            # 这样"server 当前在跑哪一版"是可查的，而不是只能靠时间猜。
            _pver = getattr(self, "pending_version", -1)
            self.pending_weights = None
            self.pending_version = None
            try:
                self.agent.model.load_state_dict(_pw)
            except Exception as e:
                # ⚠️ 第 64 轮（Kimi-05）：加载失败绝不能静默。
                # 原来这个异常会被 run() 的 try/except 吃掉（"出错也必须放行"），
                # 结果是 server 用**随机初始化**的权重一直服务下去 ——
                # actor 拿到噪声动作、behavior_logits 与策略无关，而主进程 rc=0。
                self.timing["load_fail"] = self.timing.get("load_fail", 0) + 1
                if self.timing["load_fail"] <= 3:
                    print("[batch_server] ⚠️ 权重加载失败（继续用旧权重）: %r" % (e,), flush=True)
            else:
                # 只做一次自检：keys 是否全覆盖、首键是否真的落进去了、
                # BN 的 running_mean 是否从随机初值变成了 SL 的值。
                if not getattr(self, "_wcheck_done", False):
                    self._wcheck_done = True
                    try:
                        sd = self.agent.model.state_dict()
                        k0 = next(iter(_pw))
                        # ⚠️ 必须搬到同一设备再相减：server 的模型在 cuda:N，
                        # 而 `_pw` 来自 cuda:0 上的 `model_learner.state_dict()`。
                        # `load_state_dict` 自己会处理跨设备拷贝，但我这里手写的减法不会 ——
                        # 第一版就撞上 `Expected all tensors to be on the same device`。
                        _ref = sd[k0]
                        _got = _pw[k0].to(_ref.device)
                        d0 = float((_ref.float() - _got.float()).abs().max())
                        hit = sum(1 for k in _pw if k in sd)
                        bn = [m for m in self.agent.model.modules()
                              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
                        rm = float(bn[0].running_mean.abs().mean()) if bn and bn[0].running_mean is not None else -1
                        print("[batch_server] 权重自检：%d/%d 个键命中，首键 max|Δ|=%.3g，"
                              "首个 BN |running_mean|=%.3g（随机初值是 0）"
                              % (hit, len(_pw), d0, rm), flush=True)
                    except Exception as e:
                        print("[batch_server] 权重自检失败: %r" % (e,), flush=True)
            self.applied_version = int(_pver) if _pver is not None else -1
            del _pw
        self.timing["sync"] += time() - t_s

        t_b = time()
        st = self._build_state_subset(idx_t, k)
        obs_list = [{"raw_units": FakeRawUnits(self.obs_ut[i].numpy())} for i in idxs]

        # 先把「本步输入」的 hidden 存档，供 actor 填 Trajectory.memory（见 __init__ 里的说明）。
        # 必须在 forward 之前，且必须在 self.h 被写回 new hidden 之前。
        self.h_prev[:, idx_t, :] = self.h[:, idx_t, :]
        self.c_prev[:, idx_t, :] = self.c[:, idx_t, :]
        self.timing["build"] += time() - t_b

        t_f = time()
        with torch.no_grad():
            # `.contiguous()` 必需：self.h[:, idx_t, :] 是**非连续视图**（沿 dim=1 切片），
            # nn.LSTM 对非连续输入要额外拷贝，且容易在后续 reshape 里出错。
            #
            # ⚠️ 更正（第八轮复测，_check_t6_double_step.py）：
            #    这里原先写「forward 会 in-place 修改传入的 hidden_state」——**该说法不成立**。
            #    实测调用前后 caller 的 h/c 完全不变（max|Δ| = 0），LSTM 返回的是新张量。
            #    当初那次"隔离性假 FAIL"的真因是审计脚本自身的两个 bug
            #    （randperm[1:] 去掉的是随机元素不是 0；把不同 actor 的不同时刻混批），
            #    修脚本后 FAIL 消失，与 in-place 无关。
            #    保留 .contiguous() 是因为切片视图非连续，不是因为 in-place。
            al, acts, nh, sun, en = self.agent.batch_action_logits_by_state(
                st,
                hidden_state=(self.h[:, idx_t, :].contiguous().to(self.device),
                              self.c[:, idx_t, :].contiguous().to(self.device)),
                obs_list=obs_list)
        self.timing["forward"] += time() - t_f

        t_w = time()
        self.h[:, idx_t, :] = nh[0].cpu()
        self.c[:, idx_t, :] = nh[1].cpu()

        # behavior_logits 必须与 action 来自**同一次**前向的同一个 `al`（见 read_behavior_logits 说明）
        if self.return_logits:
            t_lg = time()
            self._store_logits(al, idx_t, k)
            self.timing["logits"] += time() - t_lg

        self.a_type[idx_t] = acts.action_type.reshape(-1).cpu().long()
        self.a_delay[idx_t] = acts.delay.reshape(-1).cpu().long()
        self.a_queue[idx_t] = acts.queue.reshape(-1).cpu().long()
        self.a_units[idx_t] = acts.units.reshape(k, -1)[:, :self.max_selected].cpu().long()
        self.a_tunit[idx_t] = acts.target_unit.reshape(-1).cpu().long()
        self.a_tloc[idx_t] = acts.target_location.reshape(k, -1)[:, :2].cpu().long()
        self.a_sun[idx_t] = sun.reshape(-1).cpu().long()
        self.a_en[idx_t] = en.reshape(-1).cpu().long()

        for i in idxs:
            self.req[i] = self.DONE
            self.ev_done[i].set()
        self.timing["write"] += time() - t_w

    def _build_state_subset(self, idx_t, k):
        E = self.shapes.entity_shape
        M = self.shapes.map_shape
        scalars = []
        for j, sh in enumerate(self.shapes.scalar_shapes):
            n = int(np.prod(sh))
            t = self.scalars[j][idx_t, :n]
            scalars.append(t.reshape(k, *sh).to(self.device))
        return MsState(entity_state=self.entity[idx_t].to(self.device),
                       statistical_state=scalars,
                       map_state=self.map[idx_t].to(self.device))

    def steady_throughput(self):
        """稳态吞吐（跳过前 50 轮的暖机期）。返回 None 表示轮数不足。

        ⚠️ 比较不同配置时必须用这个数，不要用"总步数/墙钟"：
        墙钟包含了 actor 的 env.reset 与首次 preprocess，短测下这部分占比极高。
        实测同一配置：30 步 -> 323、200 步 -> 277，差的 15% 全是暖机。
        """
        if getattr(self, "_t_warm", None) is None or getattr(self, "_t_end", None) is None:
            return None
        dt = self._t_end - self._t_warm
        if dt <= 0:
            return None
        return (self._s_end - self._s_warm) / dt

    def async_summary(self):
        rounds = max(int(self.stats[0]), 1)
        steps = int(self.stats[1])
        t = self.timing
        s = ("[async] rounds=%d steps=%d | 平均 batch %.2f（峰值 %d / 上限 %d）"
             "| 每轮 sync %.2f + build %.2f + forward %.2f + write %.2f = %.2f ms"
             % (rounds, steps, steps / rounds, int(self.stats[2]), self.max_batch,
                t["sync"] / rounds * 1000, t["build"] / rounds * 1000,
                t["forward"] / rounds * 1000, t["write"] / rounds * 1000,
                (t["sync"] + t["build"] + t["forward"] + t["write"]) / rounds * 1000))
        st = self.steady_throughput()
        if st is not None:
            s += " | 稳态 %.0f step/s" % st
        if self.return_logits:
            s += " | logits 每轮 %.2f ms（含在 write 里）" % (self.timing["logits"] / rounds * 1000)
        rc = self.restart_count()
        if rc:
            s += " | actor 重启 %d 次" % rc
        return s
