#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the learner in the actor-learner mode in the IMPALA architecture"

# modified from AlphaStar pseudo-code

# import objgraph

from time import time, sleep, strftime, localtime

import gc
import os
import traceback
import threading
import itertools
import datetime
import random
import copy

import numpy as np

import torch
from torch.optim import Adam, RMSprop

from tensorboardX import SummaryWriter

from alphastarmini.core.rl.rl_loss import loss_function
from alphastarmini.core.rl import rl_utils as RU
from alphastarmini.core.rl import shared_adam as SA

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import RL_Training_Hyper_Parameters as THP

__author__ = "Ruo-Ze Liu"

debug = False

# 第 86 轮（DS，#93）：H4 的**运行期判据**。
# H4 的静态证据是「构造 Learner 时没传 pub_lock/policy_version，pv 恒 0」（DS14 #68），
# 修完只有静态证据不够 —— 必须看到 pv 真的随 optimizer step 增长。
# 默认关（每次 step 一行会淹没日志），验证时用 MAS_PV_LOG=1 打开。
_PV_LOG = os.environ.get("MAS_PV_LOG") == "1"
_PV_N = [0]
# 2026-11-19：NaN 梯度守卫与 KL 周期日志
#   NaN_GUARD 默认开：一旦梯度含 NaN/Inf，Adam 会把整个模型写成 NaN，之后所有更新都无意义。
#   守卫的做法是**跳过这一步**（清零梯度）并计数，而不是静默继续。
#   KL_LOG 每 N 次更新打一行「当前 KL + 历史极值」，用于观察策略相对 SL teacher 的漂移。
_NAN_GUARD = os.environ.get("MAS_RL_NAN_GUARD", "1") == "1"
_KL_LOG = os.environ.get("MAS_RL_KL_LOG") == "1"
_KL_EVERY = int(os.environ.get("MAS_RL_KL_EVERY", "20"))

# ---- 2026-11-16 技术债 D14：梯度裁剪开关（默认关闭，算法行为改变需 A/B）----
_GRAD_CLIP = os.environ.get("MAS_RL_CLIP", "0") == "1"
_GRAD_CLIP_NORM = float(os.environ.get("MAS_RL_CLIP_NORM", str(THP.clip)))

# model path
MODEL = "rl"
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
# ---- 2026-11-16 RL 技术债 D13：存盘路径按进程区分、且可环境变量覆盖 ----
# 原实现 SAVE_PATH 在 import 时由 strftime 固定一次（秒级），同一秒内启动的多个训练
# 会写到同一个 .pth 互相覆盖；worker 之间虽只有 rank0 存盘，但多次 run 仍会撞名。
# 现在允许 MAS_RL_SAVE_PATH 显式指定；未指定时追加 pid 保证唯一。
SAVE_PATH = os.environ.get("MAS_RL_SAVE_PATH", "").strip() or os.path.join(
    MODEL_PATH, MODEL + "_" + strftime("%y-%m-%d_%H-%M-%S", localtime()) + "_p%d" % os.getpid())


class Learner:
    """Learner worker that updates agent parameters based on trajectories."""

    def __init__(self, player, rank, v_steps, cuda_device, optimizer, global_model, 
                 max_time_for_training=60 * 3, lr=THP.learning_rate, 
                 weight_decay=THP.weight_decay, baseline_weight=1,
                 is_training=True, buffer_lock=None, writer=None,
                 use_opponent_state=True, no_replay_learn=False, 
                 num_epochs=THP.num_epochs, count_of_batches=1,
                 buffer_size=10, use_random_sample=False,
                 only_update_baseline=False,
                 need_save_result=True, process_lock=None,
                 update_params_interval=10, pub_lock=None, policy_version=None):
        self.player = player
        self.player.set_learner(self)

        self.rank = rank
        self.v_steps = v_steps

        # 第 64 轮（Kimi-02）：可观测的更新计数。
        # 为什么需要：之前只有"loss 有限、backward 不报错、364 个参数有梯度"这类
        # **一次性**证据，无法回答"Learner 有没有持续更新"。
        # 长跑里如果 Learner 线程静默退出，主进程仍然 rc=0 —— 这一点被 Kimi 点出来了。
        # 现在每次 optimizer.step() 记一次，Worker 退出时打印。
        self.update_count = 0
        self.last_loss = None

        # ---- 第 73 轮（Kimi-02）：把"更新是否健康"变成可观测量 ----
        # Kimi 的原话：finite loss / 部分参数有梯度 / backward 成功，只能证明计算图连通，
        # 不能证明"每次更新都在正常幅度内"。这里补三个量：
        #   grad_norm   : backward 后、step 前的全局梯度范数（爆炸/消失会立刻显形）
        #   update_norm : step 前后参数差的 L2 范数（= 这一步到底改了多少）
        #   traj_recv   : 收到的轨迹数（按 normal/final/win 分开）
        # 只记极值与最后一次，避免每步刷日志。
        self.grad_norm_last = None
        self.grad_norm_max = 0.0
        self.update_norm_last = None
        self.update_norm_max = 0.0
        self.nan_grad_steps = 0
        self.nan_guard_skips = 0
        self.skip_update_count = 0
        self._last_skip_print = 0.0
        self.kl_last = {}
        self.traj_recv = {"normal": 0, "final": 0, "win": 0}

        # ---- 第 65 轮（Kimi-05 P0）：原子权重发布 + policy_version ----
        # 问题：同步线程直接遍历 `global_model.state_dict()`，而 Learner 可能正在
        # `optimizer.step()` 里逐个张量更新 -> 复制出的快照可能是**跨张量的新旧混合**，
        # 也就是"半新半旧"，但它看起来完全正常、不会报错。
        # 修法：Learner 在 step 前后持同一把锁，同步线程也持这把锁复制 ——
        # 快照因此只会落在"某次更新之前"或"之后"两个一致状态之一。
        # policy_version 每完成一次更新 +1，供 server 与轨迹记录，做到"权重可追溯"。
        self.pub_lock = pub_lock
        self.policy_version = policy_version

        # 第 65 轮（Kimi-03 的验收手段）：把每个 head 的蒸馏 loss（= KL 项）的
        # **最小值**记下来。判据很硬：重归一化 KL 恒 ≥ 0，所以 30 min 里
        # 只要出现一次负值，就说明 renorm 没生效或实现有洞。
        self.kl_min = {}
        self.kl_max = {}
        self.cuda_device = cuda_device

        self.name = 'learner_' + str(self.rank)

        self.trajectories = []
        self.final_trajectories = []
        self.win_trajectories = []

        self.update_params_interval = update_params_interval

        # PyTorch code
        # self.optimizer = Adam(self.get_parameters(), 
        #                       lr=lr, betas=(THP.beta1, THP.beta2), 
        #                       eps=THP.epsilon, weight_decay=weight_decay)

        self.optimizer = optimizer
        self.global_model = global_model

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True                            # Daemonize thread

        self.max_time_for_training = max_time_for_training
        self.is_running = False

        self.is_rl_training = is_training

        self.buffer_lock = buffer_lock
        self.writer = writer

        self.use_opponent_state = use_opponent_state
        self.no_replay_learn = no_replay_learn

        self.num_epochs = num_epochs
        self.count_of_batches = count_of_batches
        self.buffer_size = buffer_size
        self.use_random_sample = use_random_sample
        self.only_update_baseline = only_update_baseline

        self.need_save_result = need_save_result
        self.baseline_weight = baseline_weight
        self.process_lock = process_lock

    def get_parameters(self):
        return self.player.agent.get_parameters()

    def send_trajectory(self, trajectory):
        self.traj_recv["normal"] += 1
        self.trajectories.append(trajectory)

    def send_final_trajectory(self, trajectory):
        self.traj_recv["final"] += 1
        self.final_trajectories.append(trajectory)

    def send_win_trajectory(self, trajectory):
        self.traj_recv["win"] += 1
        self.win_trajectories.append(trajectory)        

    def get_normal_trajectories(self):
        batch_size = AHP.batch_size
        sample_size = self.count_of_batches * batch_size
        max_size = 1 * sample_size * self.buffer_size

        trajectories = []

        # TODO: change the sampling method
        # if self.use_random_sample:
        #     random.shuffle(self.trajectories)

        if self.use_random_sample:
            sample_index = np.random.choice(len(self.trajectories), sample_size, replace=False)
            trajectories_reduced = [self.trajectories[i] for i in sample_index]
        else:
            trajectories_reduced = self.trajectories[:sample_size]

        trajectories = trajectories_reduced
        del trajectories_reduced

        # assert len(trajectories) == sample_size

        # update self.trajectories
        if len(self.trajectories) > 0 and len(self.trajectories) <= max_size:
            # first-in first out
            self.trajectories = self.trajectories[sample_size:]
        elif len(self.trajectories) > max_size:
            print(self.name, 'len(self.trajectories) larger than max_size, begin to drop!')
            drop_size = len(self.trajectories) - max_size + sample_size
            self.trajectories = self.trajectories[drop_size:]
            print(self.name, 'len(self.trajectories)', len(self.trajectories))

        return trajectories

    def get_mixed_trajectories(self):
        batch_size = AHP.batch_size
        sample_size = self.count_of_batches * batch_size

        split_ratio = [0.7, 0.2, 0.1]
        sample_final = min(1, int(sample_size * split_ratio[1]))
        sample_win = min(1, int(sample_size * split_ratio[2]))

        print('len(self.trajectories)', len(self.trajectories)) if 1 else None
        print('len(self.final_trajectories)', len(self.final_trajectories)) if 1 else None
        print('len(self.win_trajectories)', len(self.win_trajectories)) if 1 else None

        trajectories = []

        reduce_sample_size = sample_size
        if len(self.win_trajectories) > 0:
            random.shuffle(self.win_trajectories)
            trajectories_win = self.win_trajectories[:sample_win]
            trajectories_win_length = len(trajectories_win)
            reduce_sample_size = sample_size - trajectories_win_length
            trajectories.extend(trajectories_win)
            del trajectories_win

            # update self.win_trajectories
            if len(self.win_trajectories) > 128:
                self.win_trajectories = self.win_trajectories[sample_win:]

        if len(self.final_trajectories) > 0:
            random.shuffle(self.final_trajectories)
            trajectories_final = self.final_trajectories[:sample_final]
            trajectories_final_length = len(trajectories_final)
            reduce_sample_size = reduce_sample_size - trajectories_final_length
            trajectories.extend(trajectories_final)
            del trajectories_final

            # update self.final_trajectories
            if len(self.final_trajectories) > 64:
                self.final_trajectories = self.final_trajectories[sample_final:]

        if self.use_random_sample:
            random.shuffle(self.trajectories)
        trajectories_reduced = self.trajectories[:reduce_sample_size]
        trajectories.extend(trajectories_reduced)
        del trajectories_reduced

        assert len(trajectories) == sample_size

        # update self.trajectories
        if len(self.trajectories) > 0:
            self.trajectories = self.trajectories[reduce_sample_size:]

        return trajectories

    def update_parameters(self):
        if not self.is_rl_training:
            return 

        agent = self.player.agent
        batch_size = AHP.batch_size

        learner_name = self.name

        print(learner_name, 'len(self.trajectories)', len(self.trajectories)) if 1 else None

        trajectories = self.get_normal_trajectories()
        print(learner_name, 'len(trajectories)', len(trajectories)) if debug else None

        # FIX(第9轮 缺陷④): 这里原来被注释掉了。
        # 后果不是"少更新 BN 统计量"，而是 backward 直接失败：
        #   cuDNN RNN backward 只允许在 training mode 下调用
        #   -> `RuntimeError: cudnn RNN backward can only be called in training mode`
        # 也就是说这一行注释掉后，整个 RL 训练走不到 optimizer.step()。
        #
        # ⚠️ 打开后 BN 会用 batch 统计量、dropout 也会生效，forward 数值与 eval 模式不同。
        #    如果之前有人用 eval 模式跑过基准，数字不能直接对比。
        agent.agent_nn.model.train()  # for BN and dropout
        print(learner_name, "begin rl update") if debug else None

        # objgraph.show_most_common_types(limit=10)
        # growth = objgraph.growth(limit=5)
        # if len(growth):
        #     print(self.name, os.getpid(), "before update", growth)

        for ep_id in range(self.num_epochs):

            # 2026-11-19：轨迹不足一个完整 batch 时直接跳过本次更新。
            # 原实现会切出半批张量，在 rl_loss 里 reshape 失败（见上面 MIN_TRAJ 的注释）。
            _need = self.count_of_batches * batch_size
            if len(trajectories) < _need:
                # 限流：主循环 0.05s 一轮，不限流会把这行刷成日志洪水（第一版就是）。
                self.skip_update_count += 1
                _now = time()
                if _now - getattr(self, "_last_skip_print", 0.0) > 10.0:
                    self._last_skip_print = _now
                    print("[learner] skip update: %d trajectories < %d needed (total skips=%d)"
                          % (len(trajectories), _need, self.skip_update_count), flush=True)
                return
            for batch_id in range(self.count_of_batches):
                update_trajectories = trajectories[batch_id * batch_size: (batch_id + 1) * batch_size]
                print(learner_name, 'len(update_trajectories)', len(update_trajectories)) if debug else None

                loss_dict_items = None
                loss_item = None

                # with self.process_lock:
                if 1:
                    print(learner_name, "begin update") if debug else None
                    SA.show_datas(agent.agent_nn.model, self.global_model, debug)
                    SA.show_grads(agent.agent_nn.model, self.global_model, debug)

                    print(learner_name, "begin synchronize") if debug else None
                    agent.agent_nn.model.load_state_dict(self.global_model.state_dict())
                    SA.show_datas(agent.agent_nn.model, self.global_model, debug)

                    loss, loss_dict = loss_function(agent, update_trajectories, self.use_opponent_state, 
                                                    self.no_replay_learn, self.only_update_baseline,
                                                    self.baseline_weight)
                    loss_dict_items = loss_dict.items()
                    loss_item = loss.item()

                    # ---- 2026-11-16 RL 技术债 D3：清零"算梯度的本地模型" ----
                    # 原实现只有 optimizer.zero_grad()，而 optimizer 绑定的是 global_model
                    # （构造见 rl_vs_inner_bot_mp.py:1575-1580），backward 却发生在 local 副本上。
                    # load_state_dict 只改 data、不改 grad ⇒ local 的 .grad 跨 batch/epoch 累积，
                    # 每个 batch 的梯度都掺着之前所有批次的残留。
                    # 修法：backward 前显式清 local；optimizer.zero_grad() 保留（清 global 的旧 grad，
                    # 让 ensure_shared_grads 的 "shared_param.grad is not None" 分支按预期工作）。
                    print(learner_name, "begin zero_grad") if debug else None
                    self.optimizer.zero_grad()
                    agent.agent_nn.model.zero_grad(set_to_none=True)
                    SA.show_grads(agent.agent_nn.model, self.global_model, debug)

                    print(learner_name, "begin backward") if debug else None
                    loss.backward()

                    # ---- 2026-11-16 技术债 D14：可选全局范数梯度裁剪 ----
                    # THP.clip=0.5 一直定义但从未被使用（全模块 grep clip_grad 0 命中）。
                    # 15 个 Learner Hogwild 并发、无裁剪，单批出现大优势时 update_norm 可能跳步。
                    # 裁剪对象必须是 local 模型：backward 的梯度在那里，ensure_shared_grads
                    # 之后才搬到 global。默认关闭——这是算法行为改变，需 A/B 判定，
                    # 用 MAS_RL_CLIP=1 打开，可用 MAS_RL_CLIP_NORM 覆盖阈值。
                    if _GRAD_CLIP:
                        torch.nn.utils.clip_grad_norm_(
                            [q for q in agent.agent_nn.model.parameters() if q.grad is not None],
                            max_norm=_GRAD_CLIP_NORM)

                    SA.show_grads(agent.agent_nn.model, self.global_model, debug)

                    print(learner_name, "begin shared_grads") if debug else None
                    SA.ensure_shared_grads(agent.agent_nn.model, self.global_model, debug)
                    SA.show_grads(agent.agent_nn.model, self.global_model, debug)

                    print(learner_name, "begin optimizer_step") if debug else None

                    # ---- 第 73 轮（Kimi-02）：step 前测 grad_norm，并留一份参数快照 ----
                    # grad_norm 用**本地模型**的梯度（backward 的直接产物）；
                    # update_norm 必须用 **global_model** —— optimizer.step() 更新的是它，
                    # 本地模型只是算梯度的副本（A3C 结构）。第一版对本地参数做差，
                    # 结果恒为 0，等于量错了对象（GLM 踩坑，冒烟抓出来的）。
                    _ps = [q for q in agent.agent_nn.model.parameters() if q.grad is not None]
                    _gn = 0.0
                    for _q in _ps:
                        _g = _q.grad.detach()
                        _gn += float(_g.norm(2) ** 2)
                    _gn = _gn ** 0.5
                    from alphastarmini.lib.utils import is_bad_norm as _is_bad_norm
                    _gn_bad = _is_bad_norm(_gn)
                    if _gn_bad:
                        self.nan_grad_steps += 1
                    self.grad_norm_last = _gn
                    if _gn > self.grad_norm_max:
                        self.grad_norm_max = _gn
                    _gps = [q for q in self.global_model.parameters() if q.grad is not None]
                    _before = [q.detach().clone() for q in _gps]

                    # 第 65 轮：step 与"别人拷快照"互斥，见 __init__ 里 pub_lock 的说明。
                    if _gn_bad and _NAN_GUARD:
                        # 跳过这一步：清零梯度，其余流程照走（快照/计数不变），但不污染模型。
                        for _q in _ps:
                            _q.grad.zero_()
                        self.nan_guard_skips += 1
                        print("[NaN] skip optimizer step (bad grad_norm=%r, skips=%d)"
                              % (_gn, self.nan_guard_skips), flush=True)

                    if self.pub_lock is not None:
                        self.pub_lock.acquire()
                    try:
                        if _gn_bad and _NAN_GUARD:
                            # 跳过时**不能**递增 policy_version：版本号是"参数变过了"的信号，
                            # 虚增会让 actor 白同步一次，也会把 optimizer_steps 刷高。
                            pass
                        else:
                            self.optimizer.step()
                            if self.policy_version is not None:
                                self.policy_version.value += 1
                                if _PV_LOG:
                                    _PV_N[0] += 1
                                    print("[PV] learner_update n=%d pv=%d"
                                          % (_PV_N[0], int(self.policy_version.value)), flush=True)
                    finally:
                        if self.pub_lock is not None:
                            self.pub_lock.release()

                    # step 后：参数更新范数（= 这一步实际改了多少）
                    try:
                        _un = 0.0
                        for _q, _b in zip(_gps, _before):
                            _un += float((_q.detach() - _b).norm(2) ** 2)
                        _un = _un ** 0.5
                        self.update_norm_last = _un
                        if _un > self.update_norm_max:
                            self.update_norm_max = _un
                    except Exception:
                        pass
                    del _before, _gps
                    self.update_count += 1
                    self.last_loss = float(loss_item) if loss_item is not None else None
                    # 记录各 head 的 KL 极值（loss_dict 的 key 就是字段名）
                    try:
                        for _k, _v in (loss_dict_items or []):
                            # ⚠️ 第一版按 `action_type` 这样的裸字段名匹配，**一个都没匹配上**：
                            #    rl_loss 里合并时是 `"all_kl_loss_" + key + ":"`，
                            #    所以真实 key 是 `all_kl_loss_action_type:`。
                            _f_name = None
                            if _k.startswith("all_kl_loss_") and _k.endswith(":"):
                                _f_name = _k[len("all_kl_loss_"):-1]
                            elif _k in ("action_type", "delay", "queue",
                                        "units", "target_unit", "target_location"):
                                _f_name = _k
                            if _f_name is None:
                                continue
                            _f = float(_v)
                            if _f_name not in self.kl_min or _f < self.kl_min[_f_name]:
                                self.kl_min[_f_name] = _f
                            if _f_name not in self.kl_max or _f > self.kl_max[_f_name]:
                                self.kl_max[_f_name] = _f
                            self.kl_last[_f_name] = _f
                            if _f_name == "action_type":
                                self.kl_last_action_type = _f
                    except Exception:
                        pass
                    if _KL_LOG and self.update_count % _KL_EVERY == 0 and self.kl_last:
                        print("[KL] step=%d %s" % (
                            self.update_count,
                            " ".join("%s=%.4f[%.4f,%.4f]" % (
                                _n, self.kl_last[_n], self.kl_min.get(_n, float("nan")),
                                self.kl_max.get(_n, float("nan")))
                                for _n in sorted(self.kl_last))), flush=True)
                    SA.show_datas(agent.agent_nn.model, self.global_model, debug)

                    # print(learner_name, "begin load_state_dict") if debug else None
                    # agent.agent_nn.model.load_state_dict(self.global_model.state_dict())
                    # SA.show_datas(agent.agent_nn.model, self.global_model, debug)

                    del loss, loss_dict
                    print(learner_name, "end update") if debug else None

                del update_trajectories
                print(learner_name, "loss:", loss_item) if debug else None

                for i, k in loss_dict_items:
                    print(i, k) if debug else None
                    del i, k

                if self.need_save_result:
                    self.writer.add_scalar('loss/' + learner_name, loss_item, agent.steps)
                    for i, k in loss_dict_items:
                        self.writer.add_scalar(learner_name + '/' + i, k, agent.steps)

                del loss_item, loss_dict_items

                agent.steps += AHP.batch_size * AHP.sequence_length
                # ---- 2026-11-16 RL 技术债 D12：跨进程计数加原子保护 ----
                # mp.Value('d').value += x 是 read-modify-write，15 个 Learner 并发会丢计数
                # （Parameter_Server 收尾已自认"可能少计，只做量级校验"）。
                # 用 Value 自带的锁包住 +=，不改变计数语义、不影响热路径以外的地方。
                if getattr(self.v_steps, "get_lock", None) is not None:
                    with self.v_steps.get_lock():
                        self.v_steps.value += AHP.batch_size * AHP.sequence_length
                else:
                    self.v_steps.value += AHP.batch_size * AHP.sequence_length

        # agent.agent_nn.model.eval()
        print(learner_name, "end rl update") if debug else None

        del trajectories
        gc.collect()
        # if self.rank == 0:
        #     growth = objgraph.growth(limit=5)
        #     if len(growth):
        #         print(self.name, os.getpid(), "growth", growth)

        if self.need_save_result:
            torch.save(agent.agent_nn.model.state_dict(), SAVE_PATH + "" + ".pth")

    def start(self):
        self.thread.start()

    # background
    def run(self):
        try:
            start_time = time()
            update_params_timer = time()
            self.is_running = True

            while time() - start_time < self.max_time_for_training:
                try:
                    # if at least one actor is running, the learner would not stop
                    actor_is_running = False
                    if len(self.player.actors) == 0:
                        actor_is_running = True

                    for actor in self.player.actors:
                        if actor.is_start:
                            actor_is_running = actor_is_running | actor.is_running
                        else:
                            actor_is_running = actor_is_running | 1

                    if actor_is_running:
                        print('learner trajectories size:', len(self.trajectories)) if debug else None

                        if time() - update_params_timer > self.update_params_interval:
                            self.player.agent.agent_nn.model.load_state_dict(self.global_model.state_dict())
                            update_params_timer = time()

                        # 2026-11-17：更新门槛原先隐式等于 buffer_size × batches × batch_size。
                        # 开 USE_BUFFER 会同时把门槛从 16 条改成 80 条，采样与节奏两个变量一起变，
                        # 无法归因。MAS_RL_MIN_TRAJ 允许单独固定节奏（默认仍按原公式）。
                        _min_traj_env = os.environ.get("MAS_RL_MIN_TRAJ", "")
                        # ⚠️ 下限至少是「一次更新要吃的轨迹数」= count_of_batches × batch_size。
                        #    第一版允许把它设到 4，结果只凑到半个 batch 就去 reshape，
                        #    直接 `shape '[8, 8, 564]' is invalid for input of size 18048` 崩掉。
                        _min_traj = max(
                            int(_min_traj_env) if _min_traj_env
                            else self.buffer_size * self.count_of_batches * AHP.batch_size,
                            self.count_of_batches * AHP.batch_size)
                        if len(self.trajectories) >= _min_traj:
                            print("learner begin to update parameters") if debug else None
                            self.update_parameters()
                            print("learner end updating parameters") if debug else None

                        sleep(0.05)
                    else:
                        print("Actor stops!") if debug else None

                        print("Learner also stops!") if debug else None
                        return

                except Exception as e:
                    print("Learner.run() Exception cause break, Detials of the Exception:", e) if debug else None
                    print(traceback.format_exc())
                    break

        except Exception as e:
            print("Learner.run() Exception cause return, Detials of the Exception:", e) if debug else None

        finally:
            self.is_running = False


def test(on_server):
    pass
