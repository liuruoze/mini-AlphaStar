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
from alphastarmini.lib import entity_order as EO

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

# 保留率阈值。只有 "func_call 不为空" 的帧才会被写成样本，所以这些值直接决定
# 每个 replay 能产出多少帧，也决定训练集里各类动作的占比。
#
# 原值（0.000 / 0.00 / 0.0 / 0.1 / 0.1 / 0.5）会把 no_op、镜头移动、Smart_pt 全部丢掉，
# 一局 2000+ 个决策点只剩约 57 帧，且训练集里 no_op 样本数为 0 —— 而 RL 推理时
# 80~90% 的动作本该是 no_op，导致训练/推理分布严重错配。注释里作者原本写的
# 0.005 / 0.05 / 0.3 也同样是"几乎不留"，这里按 SL 需要覆盖常见动作分布来重设。
flags.DEFINE_float("no_op_threshold", 0.10, "The threshold to save no op operations.")
flags.DEFINE_float("move_camera_threshold", 0.05, "The threshold to save move_camera operations.")
flags.DEFINE_float("Smart_pt_threshold", 0.3, "The threshold to save Smart_pt operations.")
flags.DEFINE_float("Smart_unit_threshold", 0.3, "The threshold to save Smart_unit operations.")
flags.DEFINE_float("Harvest_Gather_unit_threshold", 0.3, "The threshold to save Harvest_Gather_unit operations.")
flags.DEFINE_float("Attack_pt_threshold", 0.5, "The threshold to save Attack_pt operations.")

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

# R7 (2026-11-12): 原先 `SIMPLE_TEST = not P.on_server` 会在忘设 MAS_ON_SERVER 时
# 静默把 DATA_NUM 降到 2，首屏只显示 "data range: [0, 2)"——正是 11-09 事故的形态。
# 改为显式开关，默认关闭；90 局上限保持不变（训练纪律：统一 90 replay）。
# 冒烟用 MAS_DATA_SMOKE=1；也可用 run_transform.py 的 MAS_DATA_NUM 显式覆盖。
SIMPLE_TEST = os.environ.get("MAS_DATA_SMOKE", "0") == "1"
DATA_FROM = 0
DATA_NUM = 2 if SIMPLE_TEST else 90
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
    # bool_bo = np.float64(np.random.rand(*bo.shape) < BO_PROBABILITY)
    # bool_bu = np.float64(np.random.rand(*bu.shape) < BU_PROBABILITY)

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


# ---- 重排等价性验证（MAS_VERIFY_ORDER=1，默认关闭，只做只读比较） ----
_ORDER_VERIFY = {"n": 0, "n_act": 0, "n_bad": 0, "bad": []}


def _order_verify(o, feat, obsA, obsB, order):
    """同一帧分别用原序(obsA)与新序(obsB)生成 label，比较语义是否等价。

    新序的 unit_tags / target_unit_tag 是「新槽索引」，用 order（new->old）
    映射回原序索引后，应与原序结果完全一致；不一致即说明重排改变了标签语义。

    注意：pysc2 的 FunctionCall.function 是 _Raw_Functions 枚举（只有 .name/.value，
    没有 .args），参数顺序由 reverse_raw_action 固定为
    raw_cmd_unit: (queued, unit_tags, target_unit_tag)
    raw_cmd_pt:   (queued, unit_tags, world)
    ⇒ 索引映射只对 arguments[1] 和（raw_cmd_unit 的）arguments[2] 生效。
    """
    global _ORDER_VERIFY
    _ORDER_VERIFY["n"] += 1
    try:
        fa = getFuncCall(o, feat, obsA)
        fb = getFuncCall(o, feat, obsB)
    except Exception as e:
        _ORDER_VERIFY["n_bad"] += 1
        if len(_ORDER_VERIFY["bad"]) < 5:
            _ORDER_VERIFY["bad"].append({"why": "exception", "err": str(e)[:200]})
        return

    if fa is None and fb is None:
        return
    if (fa is None) != (fb is None):
        _ORDER_VERIFY["n_bad"] += 1
        if len(_ORDER_VERIFY["bad"]) < 5:
            _ORDER_VERIFY["bad"].append({
                "why": "none_mismatch",
                "a": None if fa is None else len(fa),
                "b": None if fb is None else len(fb)})
        return

    _ORDER_VERIFY["n_act"] += 1

    def sig(fc, mp):
        out = []
        for f in fc:
            fname = str(f.function.name)
            fid = int(f.function.value)
            args = list(f.arguments)
            vals = []
            for i, a in enumerate(args):
                if isinstance(a, (list, tuple)):
                    v = [int(x) for x in a]
                else:
                    v = [int(a)]
                if mp is not None:
                    if i == 1:          # unit_tags（索引）
                        v = [int(mp[x]) if 0 <= x < len(mp) else -999 for x in v]
                    elif i == 2 and "unit" in fname:   # target_unit_tag（索引）
                        v = [int(mp[x]) if 0 <= x < len(mp) else -999 for x in v]
                if i == 1:              # unit_tags 是无序集合，两侧都要排序后再比
                    v = sorted(v)
                vals.append(tuple(v))
            out.append((fid, tuple(vals)))
        return sorted(out)

    try:
        sa = sig(fa, None)
        sb = sig(fb, order)
    except Exception as e:
        _ORDER_VERIFY["n_bad"] += 1
        if len(_ORDER_VERIFY["bad"]) < 5:
            _ORDER_VERIFY["bad"].append({"why": "sig_exception", "err": str(e)[:200]})
        return

    if sa != sb:
        _ORDER_VERIFY["n_bad"] += 1
        if len(_ORDER_VERIFY["bad"]) < 5:
            _ORDER_VERIFY["bad"].append({"why": "sig",
                                         "a": str(sa)[:260],
                                         "b": str(sb)[:260]})
    if _ORDER_VERIFY["n"] % 200 == 0:
        print("[order-verify] frames=%d act=%d bad=%d"
              % (_ORDER_VERIFY["n"], _ORDER_VERIFY["n_act"], _ORDER_VERIFY["n_bad"]),
              flush=True)


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

            if os.environ.get("MAS_DEBUG_FUNCALL") == "1":
                _aid = -1
                try:
                    if action.HasField("action_raw") and action.action_raw.HasField("unit_command"):
                        _aid = action.action_raw.unit_command.ability_id
                except Exception:
                    pass
                print("[funcall] aid=%s -> %s (value=%s)"
                      % (_aid, raw_func_call.function.name, raw_func_call.function.value),
                      flush=True)

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


def get_map_data_for_replay(replay_path):
    """把 replay 对应地图的本地字节读出来，用于 RequestStartReplay.map_data。

    背景：SC2 打开 replay 时只会用 replay 里记录的相对地图名（如 "Melee/Simple64.SC2Map"）
    自己去 $SC2PATH/Maps 下找，部分环境（本机实测 Linux + SC2 3.16.1/Base55958）会报
    `LaunchError: Unable to open map.`；同一张图用 SC2Env 新开一局却是正常的。
    把地图字节显式塞进 map_data 即可绕开 SC2 自己的查找逻辑。

    返回 None 时行为与原来的 map_data=None 一致（回退给 SC2 自己找）。
    """
    try:
        import json
        import mpyq

        with open(replay_path, 'rb') as f:
            archive = mpyq.MPQArchive(f)
            meta = archive.read_file('replay.gamemetadata.json')
        if isinstance(meta, bytes):
            meta = meta.decode('utf-8', 'ignore')
        map_name = json.loads(meta).get('MapName', '')
        if not map_name:
            return None

        maps_root = os.path.join(os.environ.get('SC2PATH', os.path.expanduser('~/StarCraftII')), 'Maps')
        map_file = os.path.join(maps_root, map_name)
        if not os.path.exists(map_file):
            target = os.path.basename(map_name)
            # 用 map_data 建局时 SC2 把地图解到 /tmp 并命名为 TempLaunchMap.SC2Map，
            # replay 里记录的就是这个临时路径，转换时必然找不到。
            # 只在识别到该临时名时才兜底，真实地图名找不到仍然抛异常。
            if "templaunchmap" in target.lower():
                _fb = os.environ.get(
                    "MAS_REPLAY_FALLBACK_MAP",
                    os.path.expanduser("~/StarCraftII/Maps/Melee/Simple64.SC2Map"))
                print("replay map is TEMP (%s) -> fallback %s" % (map_name, _fb))
                with open(_fb, "rb") as _f:
                    return _f.read()
            for root, _, files in os.walk(maps_root):
                if target in files:
                    map_file = os.path.join(root, target)
                    break

        print('replay map:', map_name, '->', map_file)
        with open(map_file, 'rb') as f:
            return f.read()
    except Exception as e:
        print('get_map_data_for_replay failed:', e)
        return None


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
    print("[TRANSFORM-config] replay_files=%d data_from=%d data_num=%d smoke=%s max_steps=%d"
          % (len(replay_files), DATA_FROM, DATA_NUM, SIMPLE_TEST, SMALL_MAX_STEPS), flush=True)

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

                # 默认：观察胜方（历史行为）；MAS_REPLAY_OBSERVER=1/2 时强制指定
                # MAS_REPLAY_OBSERVER: "0"(默认)=观察胜方; "proto"=强制观察第一个 Protoss(我方), 输赢都保留; "1"/"2"=指定 player_id
                _obs_env = os.environ.get("MAS_REPLAY_OBSERVER", "0").strip()
                if _obs_env.lower() in ("proto", "protoss"):
                    win_observe_id = observe_id_list[0] if observe_id_list else 0
                    print('[replay] forced observer(proto) =', win_observe_id,
                          'results=', observe_result_list)
                else:
                    win_observe_id = int(_obs_env)
                    if win_observe_id == 0:
                        for i, result in enumerate(observe_result_list):
                            if result == sc_pb.Victory:
                                win_observe_id = observe_id_list[i]
                                break
                    else:
                        print('[replay] forced observer =', win_observe_id)

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
                    map_data=get_map_data_for_replay(replay_path),
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
                    if (os.environ.get("MAS_DEBUG_FUNCALL_OBS") == "1"
                            and len(o.actions) > 0):
                        print("[obs-act] i=%d n=%d gl=%d"
                              % (i, len(o.actions), o.observation.game_loop),
                              flush=True)
                    try:
                        obs = feat.transform_obs(o)
                        _ord_mode = os.environ.get("MAS_ENTITY_ORDER")
                        if _ord_mode in ("mypref", "myall", "myprune"):
                            _raw = obs["raw_units"]
                            _my = np.asarray(_raw[:, F.FeatureUnit.alliance]) == 1
                            if _ord_mode == "myprune":
                                # myprune：**不重排**，只在超限时优先删非我方（保位置先验）。
                                # n <= max_entities 时 order = range(n)，输出与不启用逐位相同。
                                _order = EO.prune_keep_my(_my, AHP.max_entities)
                            else:
                                if _ord_mode == "mypref":
                                    _sel = np.asarray(_raw[:, F.FeatureUnit.is_selected]) > 0
                                else:
                                    # myall：只按 alliance 稳定分区。
                                    # 不用 is_selected 的原因（J135 实测）：评测端 agent 发
                                    # raw command，SC2 的 is_selected 恒为 0（nsel=0），
                                    # 训练端却是人类框选状态 ⇒ 排序键在评测端不可复现。
                                    # ⚠ 该模式已被 J136 否证（破坏位置先验），仅留作复现。
                                    _sel = np.zeros(len(_raw), dtype=bool)
                                _order = EO.reorder_indices(_sel, _my, AHP.max_entities)
                            if os.environ.get("MAS_VERIFY_ORDER") == "1":
                                _obsA = dict(obs)
                                _obsB = dict(obs)
                                _obsB["raw_units"] = _raw[_order]
                                _order_verify(o, feat, _obsA, _obsB, _order)
                            obs["raw_units"] = _raw[_order]

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

                if os.environ.get("MAS_VERIFY_ORDER") == "1":
                    print("[order-verify] FINAL frames=%d act=%d bad=%d"
                          % (_ORDER_VERIFY["n"], _ORDER_VERIFY["n_act"],
                             _ORDER_VERIFY["n_bad"]), flush=True)
                    for _b in _ORDER_VERIFY["bad"]:
                        print("[order-verify] BAD %s" % (_b,), flush=True)

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

                        # 落盘用 float32：原始 numpy 是 float64，
                        # 单帧 feature+label 会从 16.9 MB 涨到两倍，而训练时本来就要 .float() 转回来。
                        feature = torch.tensor(feature, dtype=torch.float32)
                        label = torch.tensor(label, dtype=torch.float32)
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

    # R7: 数量门禁。written 只在成功写盘后累加，单局异常被上面的 except 吞掉时
    # 会少于 expected；不比对就会静默产出缺局的数据集（11-09 事故的教训）。
    written = len(replay_length_list)
    upper = end_index if end_index is not None else len(replay_files)
    expected = max(0, min(upper, len(replay_files)) - from_index)
    print("[TRANSFORM-summary] written=%d expected=%d replay_files=%d range=[%d,%s)"
          % (written, expected, len(replay_files), from_index,
             str(end_index) if end_index is not None else "end"), flush=True)
    if written != expected:
        print("[TRANSFORM-ERROR] written=%d != expected=%d; %d replay(s) failed or were skipped"
              % (written, expected, expected - written), flush=True)
    return written, expected
