#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the sc2 environment, maybe add some functions for the original pysc2 version "

# modified from AlphaStar pseudo-code

import numpy as np

from s2clientprotocol import sc2api_pb2 as sc_pb

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.env import sc2_env

from alphastarmini.core.rl.alphastar_agent import RandomAgent, AlphaStarAgent
from alphastarmini.core.rl.env_run_loop import run_loop

from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

__author__ = "Ruo-Ze Liu"

debug = False


class SC2Environment:
    """See PySC2 environment."""

    """
    It is noted that AlphaStar only acts on pysc2 3.0 version! If with previouse pysc2 version,
    no raw observations and actions can be used. And the pysc2-3.0 is released on September, 2019, 
    the time AlphaStar was publiciliy accessed in the Nature. These informations are important, please
    do not miss them.

    """

    def __init__(self, settings):
        pass

    def step(self, home_action, away_action):
        pass

    def reset(self):
        pass


# from the player timestep to get the outcome
# not used
def get_env_outcome(timestep):
    outcome = 0
    o = timestep.raw_observation
    player_id = o.observation.player_common.player_id
    for r in o.player_result:
        if r.player_id == player_id:
            outcome = sc2_env.possible_results.get(r.result, default=0)
    frames = o.observation.game_loop
    return outcome


def test_multi_player_env(agent_interface_format):
    steps = 10000
    step_mul = 1
    players = 2
    agent_names = ["Protoss", "Terran"]

    # create env
    with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.protoss, agent_names[0]),
                     sc2_env.Agent(sc2_env.Race.terran, agent_names[1])],
            step_mul=step_mul,
            game_steps_per_episode=steps * step_mul // 2,
            agent_interface_format=agent_interface_format,
            version=None,
            random_seed=1) as env:
            # begin env

        #agents = [RandomAgent(x) for x in agent_names]
        agents = [AlphaStarAgent(x) for x in agent_names]

        run_loop(agents, env, steps)


def random_agent_test():

    aif_1 = sc2_env.AgentInterfaceFormat(**AAIFP._asdict())
    aif_2 = sc2_env.AgentInterfaceFormat(**AAIFP._asdict())

    aif = [aif_1, aif_2]

    test_multi_player_env(aif)


# not used
def run_thread_test():
    run_config = run_configs.get(version="3.16.1")

    camera_width = 24
    interface = sc_pb.InterfaceOptions(
        raw=True, score=True,
        feature_layer=sc_pb.SpatialCameraSetup(width=camera_width))

    screen_resolution = point.Point(32, 32)
    minimap_resolution = point.Point(32, 32)
    screen_resolution.assign_to(interface.feature_layer.resolution)
    minimap_resolution.assign_to(interface.feature_layer.minimap_resolution)

    with run_config.start(full_screen=False) as controller:
        pass


def test(on_server=False):
    random_agent_test()
