#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---- 2026-11-16 设计裁定 #4：历史入口，按当前签名不可运行 ----
# get_supervised_agent 缺必填 device；Learner 缺 rank/v_steps/cuda_device/optimizer/global_model。
# 活的训练入口是 `python run_rl.py` -> rl_vs_inner_bot_mp.test()。
# 本文件保留作历史参考（含 replay z 奖励设计），不物理删除；勿直接运行。


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
