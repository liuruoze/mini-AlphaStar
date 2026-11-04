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

# model path
MODEL = "rl"
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + strftime("%y-%m-%d_%H-%M-%S", localtime()))


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

                    print(learner_name, "begin zero_grad") if debug else None
                    self.optimizer.zero_grad()
                    SA.show_grads(agent.agent_nn.model, self.global_model, debug)

                    print(learner_name, "begin backward") if debug else None
                    loss.backward()
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
                    if _gn != _gn or _gn in (float("inf"), float("-inf")):
                        self.nan_grad_steps += 1
                    self.grad_norm_last = _gn
                    if _gn > self.grad_norm_max:
                        self.grad_norm_max = _gn
                    _gps = [q for q in self.global_model.parameters() if q.grad is not None]
                    _before = [q.detach().clone() for q in _gps]

                    # 第 65 轮：step 与"别人拷快照"互斥，见 __init__ 里 pub_lock 的说明。
                    if self.pub_lock is not None:
                        self.pub_lock.acquire()
                    try:
                        self.optimizer.step()
                        if self.policy_version is not None:
                            self.policy_version.value += 1
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
                    except Exception:
                        pass
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

                        if len(self.trajectories) >= self.buffer_size * self.count_of_batches * AHP.batch_size:
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
