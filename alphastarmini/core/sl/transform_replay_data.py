#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Transform replay data: "

import shutil
import csv
import os
import sys
import traceback
import random
import pickle
import enum
import copy

import numpy as np

import torch

from absl import flags
from absl import app
from tqdm import tqdm

from pysc2.lib import point
from pysc2.lib import features as F
from pysc2.lib import actions as A
from pysc2.lib.actions import RAW_FUNCTIONS
from pysc2.lib import units as Unit
from pysc2 import run_configs
from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as com_pb

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import Label_Size as LS

from alphastarmini.lib import utils as U

import param as P

__author__ = "Ruo-Ze Liu"

debug = False

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
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

# no_op_threshold = (22.4 * 60 - 150) / 150 * len(RAW_FUNCTIONS) * scale_factor
flags.DEFINE_float("no_op_threshold", 0.000, "The threshold to save no op operations.")  # 0.005

# move_camera_threshold = (50 / 1) * len(RAW_FUNCTIONS) * scale_factor_2
flags.DEFINE_float("move_camera_threshold", 0.00, "The threshold to save move_camera operations.")  # 0.05
flags.DEFINE_float("Smart_pt_threshold", 0.0, "The threshold to save no op operations.")  # 0.3
flags.DEFINE_float("Smart_unit_threshold", 0.1, "The threshold to save no op operations.")  # 0.3
flags.DEFINE_float("Harvest_Gather_unit_threshold", 0.1, "The threshold to save no op operations.")  # 0.3
flags.DEFINE_float("Attack_pt_threshold", 0.5, "The threshold to save no op operations.")  # 0.3

flags.DEFINE_bool("disable_fog", False, "Whether tp disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe. For 2 player game, this can be 1 or 2.")

flags.DEFINE_integer("save_type", 0, "0 is torch_tensor, 1 is python_pickle, 2 is numpy_array")
flags.DEFINE_string("replay_version", "3.16.1", "the replays released by blizzard are all 3.16.1 version")

# note, replay path should be absoulte path
# D:/work/gitproject/mini-AlphaStar/data/Replays/filtered_replays_1/
# D:/work/gitproject/mini-AlphaStar/data/Replays/small_simple64_replays/
# D:/StarCraft/StarCraft II/Replays/revised_simple64_replays/
flags.DEFINE_string("no_server_replay_path", "D:/work/gitproject/mini-AlphaStar/data/Replays/filtered_replays_1/", "path of replay data")

flags.DEFINE_bool("save_data", False, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/replay_data/", "path to replays_save replay data")
flags.DEFINE_string("save_path_tensor", "./data/replay_data_tensor_new_small/", "path to replays_save replay data tensor")
FLAGS(sys.argv)


RACE = ['Terran', 'Zerg', 'Protoss', 'Random']
RESULT = ['Victory', 'Defeat', 'Tie']

SIMPLE_TEST = not P.on_server
if SIMPLE_TEST:
    DATA_FROM = 0
    DATA_NUM = 2
else:
    DATA_FROM = 0
    DATA_NUM = 90
SMALL_MAX_STEPS = 1000  # 5000, 60 * 60 * 22.4


class SaveType(enum.IntEnum):
    torch_tensor = 0
    python_pickle = 1
    numpy_array = 2


if FLAGS.save_type == 0:
    SAVE_TYPE = SaveType.torch_tensor
elif FLAGS.save_type == 1:
    SAVE_TYPE = SaveType.python_pickle
else:
    SAVE_TYPE = SaveType.numpy_array


def check_info(replay_info):
    map_name = replay_info.map_name
    player1_race = replay_info.player_info[0].player_info.race_actual
    player2_race = replay_info.player_info[1].player_info.race_actual

    print('map_name:', map_name)
    print('player1_race:', player1_race)
    print('player2_race:', player2_race)

    return True


def store_info(replay_info):
    map_name = replay_info.map_name
    player1_race = RACE[replay_info.player_info[0].player_info.race_requested - 1]
    player2_race = RACE[replay_info.player_info[1].player_info.race_requested - 1]
    game_duration_loops = replay_info.game_duration_loops
    game_duration_seconds = replay_info.game_duration_seconds
    game_version = replay_info.game_version
    game_result = RESULT[replay_info.player_info[0].player_result.result - 1]
    return [map_name,
            game_version,
            game_result,
            player1_race,
            player2_race,
            game_duration_loops,
            game_duration_seconds]


def getFeatureAndLabel_numpy(obs, func_call, delay=None, last_list=None, build_order=None):
    print("begin s:") if debug else None

    s = Agent.get_state_and_action_from_pickle_numpy(obs, build_order=build_order, last_list=last_list)

    feature = Feature.state2feature_numpy(s)
    print("feature:", feature) if debug else None
    print("feature.shape:", feature.shape) if debug else None

    print("begin a:") if debug else None
    print('func_call', func_call) if debug else None
    action = Agent.func_call_to_action(func_call)
    action.delay = min(delay, LS.delay_encoding - 1)

    print("action:", action) if debug else None
    action_array = action.toArray()
    print("action_array:", action_array) if debug else None

    a = action_array.toLogits_numpy()
    print("a:", a) if debug else None
    label = Label.action2label_numpy(a)
    print("label:", label) if debug else None
    print("label.shape:", label.shape) if debug else None

    return feature, label


def getObsAndFunc(obs, func_call, z):
    # add z
    [bo, bu] = z

    # for build order has nothing, we make it be a list has only one item:
    # if len(bo) == 0:
    #    bo = [0]

    bo = np.array(bo)
    bu = np.array(bu)

    print("bo", bo) if debug else None
    print("bu", bu) if debug else None

    # Extract build order and build units vectors from replay
    # bo = replay.get_BO(replay.home_player)
    # bu = replay.get_BU(replay.home_player)

    BO_PROBABILITY = 0.8
    BU_PROBABILITY = 0.5

    # Sample Boolean variables bool_BO and bool_BU
    # bool_bo = np.float(np.random.rand(*bo.shape) < BO_PROBABILITY)
    # bool_bu = np.float(np.random.rand(*bu.shape) < BU_PROBABILITY)

    bool_bo = np.random.rand(*bo.shape) < BO_PROBABILITY
    bool_bu = np.random.rand(*bu.shape) < BU_PROBABILITY

    bool_bo = bool_bo.astype('float32')
    bool_bu = bool_bu.astype('float32')

    print("bool_bo", bool_bo) if debug else None
    print("bool_bu", bool_bu) if debug else None

    # Generate masked build order and build units
    masked_bo = bool_bo * bo
    masked_bu = bool_bu * bu

    print("masked_bo", masked_bo) if debug else None
    print("masked_bu", masked_bu) if debug else None
    print("sum(masked_bu)", sum(masked_bu)) if debug else None

    last_actions = obs["last_actions"]
    upgrades = obs["upgrades"]
    unit_counts = obs["unit_counts"] 
    feature_effects = obs["feature_effects"]
    raw_effects = obs["raw_effects"]

    feature_minimap = obs["feature_minimap"]

    height_map = feature_minimap["height_map"]
    camera = feature_minimap["camera"]
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
                 'camera': camera,
                 'visibility_map': visibility_map,
                 'creep': creep,
                 'player_relative': player_relative,
                 'alerts': alerts,
                 'pathable': pathable,
                 'buildable': buildable,

                 'game_loop': obs["game_loop"],

                 'masked_bo': masked_bo, 
                 'masked_bu': masked_bu,

                 'func_call': func_call}

    return step_dict


def getFuncCall(o, feat, obs, use_raw=True):
    func_call = None

    if use_raw:
        # note: o.actions can be larger than 1, (when the replays are generated by 
        # the control of agent and the human player at the sametime, e.g., in the 
        # DAGGER method process),
        # at that case, the first action in o.actions may always be no-op!
        # so only use o.actions[0] can get nothing information!
        # we should iterater over all action in o.actions!
        print('len(o.actions): ', len(o.actions)) if debug else None

        action_list = []
        for action in o.actions:
            # becareful, though here recommand using prev_obs,
            # we should use obs instead. If not it will get the unit index of the last obs, not this one! 
            # not used: raw_func_call = feat.reverse_raw_action(o.actions[0], prev_obs)
            raw_func_call = feat.reverse_raw_action(action, obs)

            if raw_func_call.function.value == 0:
                # no op
                pass
            elif raw_func_call.function.value == 168:
                # camera move
                pass
            elif raw_func_call.function.value == 1:
                # we do not consider the smart point
                pass        
            else:
                print('raw_func_call: ', raw_func_call) if debug else None
                action_list.append(raw_func_call)

        if len(action_list) > 0:
            func_call = action_list
    else:

        raise NotImplementedError

    return func_call


def test(on_server=False):

    if on_server:
        REPLAY_PATH = P.replay_path  # "/home/liuruoze/data4/mini-AlphaStar/data/filtered_replays_1/" 
        COPY_PATH = None
        SAVE_PATH = "./result.csv"
        max_steps_of_replay = FLAGS.max_steps_of_replay
        max_replays = FLAGS.max_replays
    else:
        REPLAY_PATH = FLAGS.no_server_replay_path
        COPY_PATH = None
        SAVE_PATH = "./result.csv"
        max_steps_of_replay = SMALL_MAX_STEPS  # 60 * 60 * 22.4  # 60 * 60 * 22.4
        max_replays = 1  # not used

    run_config = run_configs.get(version=FLAGS.replay_version)
    print('REPLAY_PATH:', REPLAY_PATH)
    replay_files = os.listdir(REPLAY_PATH)
    print('length of replay_files:', len(replay_files))
    replay_files.sort()

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
    crop_to_playable_area = False
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
    interface.feature_layer.crop_to_playable_area = crop_to_playable_area

    agent = Agent()
    #j = 0
    replay_length_list = []
    noop_length_list = []

    from_index = DATA_FROM
    end_index = DATA_FROM + DATA_NUM

    with run_config.start(full_screen=False) as controller:

        for i, replay_file in enumerate(tqdm(replay_files)):
            try:
                replay_path = REPLAY_PATH + replay_file
                print('replay_path:', replay_path)

                do_write = False
                if i >= from_index:
                    if end_index is None:
                        do_write = True
                    elif end_index is not None and i < end_index:
                        do_write = True

                if not do_write:
                    continue 

                replay_data = run_config.replay_data(replay_path)
                replay_info = controller.replay_info(replay_data)

                print('replay_info', replay_info) if debug else None
                print('type(replay_info)', type(replay_info)) if debug else None

                print('replay_info.player_info：', replay_info.player_info) if debug else None
                infos = replay_info.player_info

                observe_id_list = []
                observe_result_list = []
                for info in infos:
                    print('info：', info) if debug else None
                    player_info = info.player_info
                    result = info.player_result.result
                    print('player_info', player_info) if debug else None
                    if player_info.race_actual == com_pb.Protoss:
                        observe_id_list.append(player_info.player_id)
                        observe_result_list.append(result)

                print('observe_id_list', observe_id_list) if debug else None
                print('observe_result_list', observe_result_list) if debug else None

                win_observe_id = 0

                for i, result in enumerate(observe_result_list):
                    if result == sc_pb.Victory:
                        win_observe_id = observe_id_list[i]
                        break

                # we observe the winning one
                print('win_observe_id', win_observe_id)

                if win_observe_id == 0:
                    print('no win_observe_id found! continue')
                    continue

                start_replay = sc_pb.RequestStartReplay(
                    replay_data=replay_data,
                    options=interface,
                    disable_fog=False,  # FLAGS.disable_fog
                    observed_player_id=win_observe_id,  # random.randint(1, 2),  # 1 or 2, wo random select it. FLAGS.observed_player
                    map_data=None,
                    realtime=False
                )

                print(" Replay info ".center(60, "-")) if 1 else None
                print("replay_info", replay_info) if 1 else None
                print("-" * 60) if debug else None
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

                aif = AgentInterfaceFormat(**AAIFP._asdict())

                feat = F.features_from_game_info(game_info=controller.game_info(),
                                                 raw_resolution=AAIFP.raw_resolution, 
                                                 crop_to_playable_area=crop_to_playable_area,
                                                 raw_crop_to_playable_area=raw_crop_to_playable_area,
                                                 hide_specific_actions=AAIFP.hide_specific_actions,
                                                 use_feature_units=True, use_raw_units=True,
                                                 use_unit_counts=True, use_raw_actions=True,
                                                 show_cloaked=True, show_burrowed_shadows=True, 
                                                 show_placeholders=True) 

                # consistent with the SL and RL setting
                # feat = F.features_from_game_info(game_info=controller.game_info(), agent_interface_format=aif)

                print("feat obs spec:", feat.observation_spec()) if debug else None
                print("feat action spec:", feat.action_spec()) if debug else None
                prev_obs = None
                prev_function = None
                i = 0
                record_i = 0
                save_steps = 0
                noop_count = 0
                obs_list, func_call_list, z_list, delay_list = [], [], [], [] 
                feature_list, label_list = [], []
                step_dict = {}

                # initial build order
                player_bo = []
                player_ucb = []
                bo_list = []

                while True:
                    o = controller.observe()
                    try:
                        obs = feat.transform_obs(o)

                        try:
                            func_call = None
                            no_op = False
                            if o.actions:
                                function_calls = getFuncCall(o, feat, obs)

                                if function_calls is not None:
                                    # when remove no_op and other actions
                                    # we can only reserver for the only the first one action
                                    func_call = function_calls[0]

                                    if func_call.function.value == 0:
                                        no_op = True
                                        func_call = None
                                    elif func_call.function.value == RAW_FUNCTIONS.raw_move_camera.id.value:  # raw_move_camera
                                        if random.uniform(0, 1) > FLAGS.move_camera_threshold:
                                            func_call = None
                                    elif func_call.function.value == RAW_FUNCTIONS.Smart_pt.id.value:  # Smart_pt
                                        if random.uniform(0, 1) > FLAGS.Smart_pt_threshold:
                                            func_call = None
                                    elif func_call.function.value == RAW_FUNCTIONS.Smart_unit.id.value:  # Smart_unit
                                        if random.uniform(0, 1) > FLAGS.Smart_unit_threshold:
                                            func_call = None
                                    elif func_call.function.value == RAW_FUNCTIONS.Harvest_Gather_unit.id.value:  # Harvest_Gather_unit
                                        if random.uniform(0, 1) > FLAGS.Harvest_Gather_unit_threshold:
                                            func_call = None
                                    elif func_call.function.value == RAW_FUNCTIONS.Attack_pt.id.value:  # Harvest_Gather_unit
                                        if random.uniform(0, 1) > FLAGS.Attack_pt_threshold:
                                            func_call = None
                            else:
                                no_op = True

                            if no_op:
                                print('expert func: no op') if debug else None
                                if random.uniform(0, 1) < FLAGS.no_op_threshold:
                                    print('get no op !') if debug else None
                                    noop_count += 1
                                    func_call = A.FunctionCall.init_with_validation("no_op", [], raw=True)

                            if func_call is not None:
                                save_steps += 1
                                #z = [player_bo, player_ucb]

                                delay = i - record_i
                                print('two action dealy:', delay, 'steps!') if debug else None
                                record_i = i

                                obs_list.append(obs)

                                func_call_list.append(func_call)
                                print('func_call:', func_call) if 1 else None

                                delay_list.append(delay)

                                if prev_obs is not None:
                                    # calculate the build order
                                    player_bo = U.calculate_build_order(player_bo, prev_obs, obs)

                                    player_bo_show = [U.get_unit_tpye_name_and_race(U.get_unit_tpye_from_index(index))[0].name for index in player_bo]
                                    print("player build order:", player_bo_show) if debug else None

                                bo_list.append(copy.deepcopy(player_bo))
                                prev_obs = obs

                        except Exception as e:
                            traceback.print_exc()

                        if i >= max_steps_of_replay:  # test the first n frames
                            print("max frames test, break out!")
                            break

                        if o.player_result:  # end of game
                            print('o.player_result', o.player_result)
                            break

                    except Exception as inst:
                        traceback.print_exc() 

                    controller.step()

                    i += 1

                print('begin save!')

                # the last delay is 0
                delay_list.append(0)
                func_call_list.append(A.FunctionCall.init_with_validation("no_op", [], raw=True))

                if SAVE_TYPE == SaveType.torch_tensor:
                    length = len(obs_list)
                    for i in range(length):
                        obs = obs_list[i]
                        func_call = func_call_list[i]
                        bo = bo_list[i]

                        # delay should be the next,
                        # means when should we issue the next command from the time we give this command
                        last_delay = delay_list[i]
                        last_func_call = func_call_list[i - 1]
                        last_action_type = last_func_call.function.value
                        action_can_be_queued = U.action_can_be_queued(last_action_type)
                        last_repeat_queued = None
                        last_action_arguments = None
                        if action_can_be_queued:
                            last_action_arguments = last_func_call.arguments
                            last_repeat_queued = last_action_arguments[0][0].value  # the first argument is alway queue
                        else:
                            last_repeat_queued = 0

                        delay = delay_list[i + 1]

                        print('last_func_call', last_func_call) if debug else None
                        print('last_action_type', last_action_type) if debug else None
                        print('last_action_can_be_queued', action_can_be_queued) if debug else None
                        print('last_action_arguments', last_action_arguments) if debug else None
                        print('last_repeat_queued', last_repeat_queued) if debug else None
                        print('last_delay', last_delay) if debug else None

                        print('func_call', func_call) if debug else None
                        print('delay', delay) if debug else None

                        last_list = [last_delay, last_action_type, last_repeat_queued]

                        feature, label = getFeatureAndLabel_numpy(obs, func_call, delay, last_list, bo)

                        feature = torch.tensor(feature)
                        label = torch.tensor(label)
                        feature_list.append(feature)
                        label_list.append(label)

                elif SAVE_TYPE == SaveType.python_pickle:
                    length = len(obs_list)
                    for i in range(length):
                        obs = obs_list[i]
                        func_call = func_call_list[i]
                        delay = delay_list[i + 1]
                        the_dict = getObsAndFunc(obs, func_call, z)
                        step_dict[i] = the_dict

                elif SAVE_TYPE == SaveType.numpy_array:
                    pass

                if SAVE_TYPE == SaveType.torch_tensor:
                    features = torch.cat(feature_list, dim=0)
                    labels = torch.cat(label_list, dim=0)
                    print('features.shape:', features.shape) if debug else None
                    print('labels.shape:', labels.shape) if debug else None
                    #m = {'features': features, 'labels': labels}
                    m = (features, labels)

                    if not os.path.exists(FLAGS.save_path_tensor):
                        os.mkdir(FLAGS.save_path_tensor)
                    file_name = FLAGS.save_path_tensor + replay_file.replace('.SC2Replay', '') + '.pt'
                    torch.save(m, file_name)

                elif SAVE_TYPE == SaveType.python_pickle:
                    file_name = FLAGS.save_path + replay_file.replace('.SC2Replay', '') + '.pickle'
                    with open(file_name, 'wb') as handle:
                        pickle.dump(step_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                elif SAVE_TYPE == SaveType.numpy_array:
                    pass    

                print('end save!')

                replay_length_list.append(save_steps)
                noop_length_list.append(noop_count)
                # We only test the first one replay            
            except Exception as inst:
                traceback.print_exc() 

            # not used
            # if j >= max_replays:  # test the first n frames
            #     print("max replays test, break out!")
            #     break

    print("end")
    print("replay_length_list:", replay_length_list)
    print("noop_length_list:", noop_length_list)
