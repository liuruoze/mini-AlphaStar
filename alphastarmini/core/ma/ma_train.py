#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the league training"

# modified from AlphaStar pseudo-code

from pysc2.env.sc2_env import Race

from alphastarmini.core.rl.utils import get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl.actor import ActorLoop

from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Training_Races as TR
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False


def league_train():
    """Trains the AlphaStar league."""
    league = League(
        initial_agents={
            race: get_supervised_agent(race)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=1,
        league_exploiters=2)

    coordinator = Coordinator(league)
    learners = []
    actors = []

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = Learner(player)
        learners.append(learner)
        actors.extend([ActorLoop(player, coordinator) for _ in range(1)])

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


def test(on_server):
    # get all the data
    # Note: The feature here is actually feature+label
    pass
