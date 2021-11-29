#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Analyze the alphastar replay "

import shutil
import csv
import os
import sys
import traceback
import random
import pickle
import enum

import numpy as np

import torch

from absl import flags
from absl import app
from tqdm import tqdm

from pysc2.lib import point
from pysc2.lib import features as F
from pysc2.lib import actions as A
from pysc2.lib import units as Unit
from pysc2 import run_configs

from s2clientprotocol import sc2api_pb2 as sc_pb

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

from alphastarmini.lib import utils as U

__author__ = "Ruo-Ze Liu"

debug = True

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 0, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", AHP.minimap_size,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_integer("max_replays", 100, "Max replays to save.")
flags.DEFINE_integer("max_steps_of_replay", int(22.4 * 60 * 60), "Max game steps of a replay, max for 1 hour of game.")
flags.DEFINE_integer("small_max_steps_of_replay", 256, "Max game steps of a replay when debug.")
flags.DEFINE_float("no_op_threshold", 0.005, "The threshold to save no op operations.")  # 0.05 is about 1/4, so we choose 0.01

flags.DEFINE_bool("disable_fog", False, "Whether tp disable fog of war.")
flags.DEFINE_integer("observed_player", 2, "Which player to observe. For 2 player game, this can be 1 or 2.")

# note: the replay version for AlphaStar's different agent is not the same
# AlphaStar Final Protoss is "4.10.0",
# AlphaStar Final Terran is "4.9.3",
# AlphaStar Final Zerg is also "4.9.3",
flags.DEFINE_string("replay_version", "4.9.3", "the replays released by alphaStar is 4.10.0")  

flags.DEFINE_bool("save_data", False, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/replay_data/", "path to replays_save replay data")
FLAGS(sys.argv)


def check_info(replay_info):
    map_name = replay_info.map_name
    player1_race = replay_info.player_info[0].player_info.race_actual
    player2_race = replay_info.player_info[1].player_info.race_actual

    print('map_name:', map_name)
    print('player1_race:', player1_race)
    print('player2_race:', player2_race)

    return True


def getFeatureAndLabel(obs, func_call, agent):
    print("begin s:") if debug else None
    s = agent.state_by_obs(obs)
    feature = Feature.state2feature(s)
    print("feature:", feature) if debug else None
    print("feature.shape:", feature.shape) if debug else None

    print("begin a:") if debug else None
    action = agent.func_call_to_action(func_call, obs=obs)
    # tag_list = agent.get_tag_list(obs)
    a = action.toLogits()
    label = Label.action2label(a)
    print("label:", label) if debug else None
    print("label.shape:", label.shape) if debug else None

    return feature, label


def getObsAndFunc(obs, func_call, agent):
    last_actions = obs["last_actions"]
    upgrades = obs["upgrades"]
    unit_counts = obs["unit_counts"] 
    feature_effects = obs["feature_effects"]
    raw_effects = obs["raw_effects"]

    feature_minimap = obs["feature_minimap"]

    height_map = feature_minimap["height_map"]
    visibility_map = feature_minimap["visibility_map"]
    creep = feature_minimap["creep"]
    player_relative = feature_minimap["player_relative"]
    alerts = feature_minimap["alerts"]
    pathable = feature_minimap["pathable"]
    buildable = feature_minimap["buildable"]

    step_dict = {'raw_units': obs["raw_units"][:AHP.max_entities],                 
                 'player': obs["player"],

                 'last_actions': last_actions,
                 'upgrades': upgrades,
                 'unit_counts': unit_counts,
                 'feature_effects': feature_effects,
                 'raw_effects': raw_effects,

                 'height_map': height_map,
                 'visibility_map': visibility_map,
                 'creep': creep,
                 'player_relative': player_relative,
                 'alerts': alerts,
                 'pathable': pathable,
                 'buildable': buildable,

                 'func_call': func_call}

    return step_dict


def getFuncCall(o, feat, prev_obs, use_raw=True):
    func_call = None

    if use_raw:
        if prev_obs is None:
            raise ValueError("use raw function call must input prev_obs")

        raw_func_call = feat.reverse_raw_action(o.actions[0], prev_obs)

        if raw_func_call.function.value == 0:
            # no op
            pass
        elif raw_func_call.function.value == 168:
            # camera move
            #print("find camera move!")
            # print(raw_func_call)
            pass
        elif raw_func_call.function.value == 1:
            # smart screen
            pass        
        else:
            pass

        func_call = raw_func_call

    return func_call


def run_alphastar_replay(on_server=False, race_name='Protoss'):
    if on_server:
        REPLAY_PATH = "/home/liuruoze/mini-AlphaStar/data/filtered_replays_1/" 
        COPY_PATH = None
        SAVE_PATH = "./result.csv"
        max_steps_of_replay = FLAGS.max_steps_of_replay
        max_replays = FLAGS.max_replays
    else:
        REPLAY_PATH = "data/Replays/replays_paper_ready/Final/" + race_name + "/"
        COPY_PATH = None
        SAVE_PATH = "./result.csv"
        max_steps_of_replay = FLAGS.max_steps_of_replay
        max_replays = 10

    run_config = run_configs.get(version=FLAGS.replay_version)
    print('REPLAY_PATH:', REPLAY_PATH)
    replay_files = os.listdir(REPLAY_PATH)
    replay_files.sort(reverse=False)

    screen_resolution = point.Point(FLAGS.screen_resolution, FLAGS.screen_resolution)
    minimap_resolution = point.Point(FLAGS.minimap_resolution, FLAGS.minimap_resolution)
    camera_width = 24

    # By default raw actions select, act and revert the selection. This is useful
    # if you're playing simultaneously with the agent so it doesn't steal your
    # selection. This inflates APM (due to deselect) and makes the actions hard
    # to follow in a replay. Setting this to true will cause raw actions to do
    # select, act, but not revert the selection.
    raw_affects_selection = False

    # Changes the coordinates in raw.proto to be relative to the playable area.
    # The map_size and playable_area will be the diagonal of the real playable area.
    raw_crop_to_playable_area = False

    interface = sc_pb.InterfaceOptions(
        raw=True, 
        score=True,
        # Omit to disable.
        feature_layer=sc_pb.SpatialCameraSetup(width=camera_width),  
        # Omit to disable.
        render=None,  
        # By default cloaked units are completely hidden. This shows some details.
        show_cloaked=False, 
        # By default burrowed units are completely hidden. This shows some details for those that produce a shadow.
        show_burrowed_shadows=False,       
        # Return placeholder units (buildings to be constructed), both for raw and feature layers.
        show_placeholders=False, 
        # see below
        raw_affects_selection=raw_affects_selection,
        # see below
        raw_crop_to_playable_area=raw_crop_to_playable_area
    )

    screen_resolution.assign_to(interface.feature_layer.resolution)
    minimap_resolution.assign_to(interface.feature_layer.minimap_resolution)

    agent = Agent()
    replay_length_list = []
    noop_length_list = []

    with open("cameras.txt", "w") as f:
        pass

    with run_config.start(full_screen=False) as controller:
        for idx in range(2):
            j = 0
            for replay_file in replay_files:

                j += 1        
                if j > max_replays:  # test the first n frames
                    print("max replays test, break out!")
                    break

                try:
                    replay_path = REPLAY_PATH + replay_file
                    print('replay_path:', replay_path)
                    replay_data = run_config.replay_data(replay_path)
                    replay_info = controller.replay_info(replay_data)
                    print('replay_info:', replay_info)
                    base = os.path.basename(replay_path)
                    replay_name = os.path.splitext(base)[0]
                    print('replay_name:', replay_name)

                    info = replay_info.player_info
                    as_id = -1
                    hp_id = -1
                    for x in range(2):
                        player_name = info[x].player_info.player_name
                        print('player_name:', player_name)
                        the_id = x + 1
                        if player_name == "AlphaStar":
                            print('Find AlphaStar!')
                            print('AlphaStar ID is', the_id)
                            as_id = the_id
                        else:
                            hp_id = the_id

                    print("as_id", as_id)
                    print("hp_id", hp_id)

                    if idx == 0:
                        player_id = as_id
                        player_name = info[player_id - 1].player_info.player_name
                        assert player_name == "AlphaStar"
                        player_name = "AS"
                    else:
                        player_id = hp_id
                        player_name = "HP"

                    start_replay = sc_pb.RequestStartReplay(
                        replay_data=replay_data,
                        options=interface,
                        disable_fog=False,  # FLAGS.disable_fog
                        observed_player_id=player_id,
                        map_data=None,
                        realtime=False
                    )

                    #print('stop', stop)
                    #print("-" * 60) if debug else None
                    controller.start_replay(start_replay)
                    # The below several arguments are default set to False, so we shall enable them.

                    # use_feature_units: Whether to include feature_unit observations.

                    # use_raw_units: Whether to include raw unit data in observations. This
                    # differs from feature_units because it includes units outside the
                    # screen and hidden units, and because unit positions are given in
                    # terms of world units instead of screen units.

                    # use_raw_actions: [bool] Whether to use raw actions as the interface.
                    # Same as specifying action_space=ActionSpace.RAW.

                    # use_unit_counts: Whether to include unit_counts observation. Disabled by
                    # default since it gives information outside the visible area. 

                    '''
                    show_cloaked: Whether to show limited information for cloaked units.
                    show_burrowed_shadows: Whether to show limited information for burrowed
                          units that leave a shadow on the ground (ie widow mines and moving
                          roaches and infestors).
                    show_placeholders: Whether to show buildings that are queued for
                          construction.
                    '''

                    feat = F.features_from_game_info(game_info=controller.game_info(), 
                                                     use_feature_units=True, use_raw_units=True,
                                                     use_unit_counts=True, use_raw_actions=True,
                                                     show_cloaked=True, show_burrowed_shadows=True, 
                                                     show_placeholders=True) 
                    #print("feat obs spec:", feat.observation_spec()) if debug else None
                    #print("feat action spec:", feat.action_spec()) if debug else None
                    prev_obs = None
                    i = 0
                    save_steps = 0
                    noop_count = 0
                    camera_count = 0
                    all_op_count = 0

                    feature_list, label_list = [], []
                    step_dict = {}

                    # set the obs and action spec
                    obs_spec = feat.observation_spec()
                    act_spec = feat.action_spec()

                    while True:
                        o = controller.observe()
                        try:
                            obs = feat.transform_obs(o)

                            try:
                                func_call = None
                                no_op = False

                                if o.actions and prev_obs:
                                    func_call = getFuncCall(o, feat, prev_obs)

                                    if func_call.function.value == 168:
                                        camera_count += 1

                                    if func_call.function.value == 0:
                                        no_op = True
                                        func_call = None
                                else:
                                    no_op = True

                                if no_op:
                                    pass
                                else:
                                    all_op_count += 1

                            except Exception as e:
                                traceback.print_exc()

                            if i >= max_steps_of_replay:  # test the first n frames
                                print("max frames test, break out!")
                                break

                            if o.player_result:  # end of game
                                # print(o.player_result)

                                break

                        except Exception as inst:
                            traceback.print_exc() 

                        controller.step()
                        prev_obs = obs
                        i += 1

                    # print("player_id", player_id, "player_name", player_name,
                    #      "camera_count", camera_count, "all_op_count", all_op_count, 
                    #      "no_camera_op_rate", 1.0 - camera_count / (all_op_count + 1e-9))

                    print("player_id", player_id, "player_name", player_name,
                          ",", camera_count, all_op_count, 
                          1.0 - camera_count / (all_op_count + 1e-9))
                    print(" ")

                    with open("cameras.txt", "a") as f:                    
                        print(replay_name, ",", player_id, ",", player_name,
                              ",", camera_count, ",", all_op_count, 
                              ",", 1.0 - camera_count / (all_op_count + 1e-9), file=f)
                    replay_length_list.append(save_steps)
                    noop_length_list.append(noop_count)
                    # We only test the first one replay   

                except Exception as inst:
                    traceback.print_exc() 

    print("end")


def test(on_server=False):
    # protoss version is 4.10.0
    run_alphastar_replay(on_server=on_server, race_name='Zerg')
    #run_alphastar_replay(player_id=2, on_server=on_server)
