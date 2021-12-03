#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the learner in the actor-learner mode in the IMPALA architecture"

# modified from AlphaStar pseudo-code
import os
import traceback
from time import time, sleep, strftime, localtime
import threading
import itertools

import torch
from torch.optim import Adam, RMSprop

from alphastarmini.core.rl.rl_loss import loss_function

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

    def __init__(self, player, max_time_for_training=60 * 3):
        self.player = player
        self.player.set_learner(self)

        self.trajectories = []

        # AlphaStar code
        #self.optimizer = AdamOptimizer(learning_rate=3e-5, beta1=0, beta2=0.99, epsilon=1e-5)

        # PyTorch code
        self.optimizer = Adam(self.get_parameters(), 
                              lr=THP.learning_rate, betas=(THP.beta1, THP.beta2), 
                              eps=THP.epsilon, weight_decay=THP.weight_decay)

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True                            # Daemonize thread

        self.max_time_for_training = max_time_for_training
        self.is_running = False

        self.is_rl_training = True

    def get_parameters(self):
        return self.player.agent.get_parameters()

    def send_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def update_parameters(self):
        trajectories = self.trajectories[:AHP.batch_size]
        self.trajectories = self.trajectories[AHP.batch_size:]

        if self.is_rl_training:
            agent = self.player.agent

            agent.agent_nn.model.train()  # for BN and dropout
            print("begin backward") if debug else None
            self.optimizer.zero_grad()

            # with torch.autograd.set_detect_anomaly(True):
            loss = loss_function(agent, trajectories)
            print("loss:", loss) if debug else None

            loss.backward()
            self.optimizer.step()

            agent.agent_nn.model.eval()  # for BN and dropout 
            print("end backward") if debug else None

            # we use new ways to save
            # torch.save(agent.agent_nn.model, SAVE_PATH + "" + ".pkl")
            if agent.steps % (10 * AHP.batch_size * AHP.sequence_length) == 0:
                torch.save(agent.agent_nn.model.state_dict(), SAVE_PATH + "" + ".pth")

            agent.steps += AHP.batch_size * AHP.sequence_length  # num_steps(trajectories)
            # self.player.agent.set_weights(self.optimizer.minimize(loss))

    def start(self):
        self.thread.start()

    # background
    def run(self):
        try:
            start_time = time()
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
                        print('learner trajectories size:', len(self.trajectories))

                        if len(self.trajectories) >= AHP.batch_size:
                            print("learner begin to update parameters")
                            self.update_parameters()
                            print("learner end updating parameters")

                        sleep(1)
                    else:
                        print("Actor stops!")

                        print("Learner also stops!")
                        return

                except Exception as e:
                    print("Learner.run() Exception cause break, Detials of the Exception:", e)
                    print(traceback.format_exc())
                    break

        except Exception as e:
            print("Learner.run() Exception cause return, Detials of the Exception:", e)

        finally:
            self.is_running = False


def test(on_server):
    pass
