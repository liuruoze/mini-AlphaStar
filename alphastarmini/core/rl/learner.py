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
                 update_params_interval=10):
        self.player = player
        self.player.set_learner(self)

        self.rank = rank
        self.v_steps = v_steps
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
        self.trajectories.append(trajectory)

    def send_final_trajectory(self, trajectory):
        self.final_trajectories.append(trajectory)

    def send_win_trajectory(self, trajectory):
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

        # agent.agent_nn.model.train()  # for BN and dropout
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
                    self.optimizer.step()
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
