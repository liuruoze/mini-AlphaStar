#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by interacting with the environment, using the replay as a plus reward"

import os

import traceback
from time import time, sleep
import threading

from pysc2.env.sc2_env import Race

from alphastarmini.core.rl.rl_utils import get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl.actor_plus_z import ActorLoopPlusZ

# below packages are for test
from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator


__author__ = "Ruo-Ze Liu"

debug = False

RESTORE = False


def test(on_server=False, replay_path=None):
    # model path
    MODEL_TYPE = "sl"
    MODEL_PATH = "./model/"
    ACTOR_NUMS = 1

    league = League(
        initial_agents={
            race: get_supervised_agent(race, path=MODEL_PATH, model_type=MODEL_TYPE, restore=RESTORE)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=0,
        league_exploiters=0)

    coordinator = Coordinator(league)
    learners = []
    actors = []

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = Learner(player, max_time_for_training=60 * 60 * 24)
        learners.append(learner)
        actors.extend([ActorLoopPlusZ(player, coordinator, replay_path=replay_path) for _ in range(ACTOR_NUMS)])

    threads = []
    for l in learners:
        l.start()
        threads.append(l.thread)
        sleep(1)
    for a in actors:
        a.start()
        threads.append(a.thread)
        sleep(1)

    try: 
        # Wait for training to finish.
        for t in threads:
            t.join()
    except Exception as e: 
        print("Exception Handled in Main, Detials of the Exception:", e)
