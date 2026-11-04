#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by fighting against built-in AI (computer), using no replays"

import os
import random
import traceback
from time import time, sleep, strftime, localtime
import threading

import numpy as np

import torch

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent, Race, Bot, Difficulty, BotBuild
from pysc2.lib import actions as sc2_actions
from pysc2.lib import units as sc2_units

from alphastarmini.core.rl.rl_utils import Trajectory, get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl import rl_utils as RU

from alphastarmini.lib import utils as L

from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


SIMPLE_TEST = not P.on_server
if SIMPLE_TEST:
    # ---- 第 76 轮（GLM，#21）：评测参数全部环境变量化 ----
    # 原来写死 MAX_EPISODES=1 / DIFFICULTY=1 / 随机种子 1 / 输出文件固定，
    # 无法做"N 局 + 分片 + 指定 checkpoint"的胜率评测（Kimi-17 / E0）。
    # 默认值与旧行为完全一致。
    import os as _os
    MAX_EPISODES = int(_os.environ.get("EVAL_EPISODES", "1"))
    GAME_STEPS_PER_EPISODE = 21000    # 9000
    SAVE_STATISTIC = True
    SAVE_REPLAY = True
    REPLAY_DIR = "./revised_simple64_replays/"
else:
    raise NotImplementedError

IS_TRAINING = False
MAP_NAME = SCHP.map_name  # P.map_name "Simple64" "AbyssalReef"
USE_PREDICT_STEP_MUL = AHP.use_predict_step_mul
STEP_MUL = 8

DIFFICULTY = int(_os.environ.get("EVAL_DIFFICULTY", "1"))
RANDOM_SEED = int(_os.environ.get("EVAL_SEED", "1"))
VERSION = SCHP.game_version

RESTORE = True

# model path
MODEL_TYPE = "sl"
MODEL_PATH = "./model/"
ACTOR_NUMS = 1
OUTPUT_FILE = _os.environ.get("EVAL_OUT", './outputs/eval_sl.txt')  # 分片各自写各的

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
if torch.backends.cudnn.is_available():
    print('cudnn available')
    print('cudnn version', torch.backends.cudnn.version())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# set random seed
# is is actually effective
# torch.manual_seed(SLTHP.seed)
# np.random.seed(SLTHP.seed)


class ActorEval:
    """A single actor loop that generates trajectories by playing with built-in AI (computer).

    We don't use batched inference here, but it was used in practice.
    """

    def __init__(self, player, coordinator, max_time_for_training=60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 4,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING,
                 replay_dir=REPLAY_DIR):
        # 第 79 轮（DS，#28）：评测时统计模型**实际输出**的动作分布（MAS_ACT_STATS=1 开启）
        self._act_cnt = {}
        self.player = player

        print('initialed player')
        self.player.add_actor(self)
        self.player.agent.set_rl_training(is_training)

        #model.load_state_dict(torch.load(model_path, map_location=device), strict=False) 
        if ON_GPU:
            self.player.agent.agent_nn.to(DEVICE)

        # self.teacher = get_supervised_agent(player.race, model_type="sl", restore=RESTORE)
        # print('initialed teacher')

        # if ON_GPU:
        #     self.teacher.agent_nn.to(DEVICE)

        self.coordinator = coordinator
        self.max_time_for_training = max_time_for_training
        self.max_time_per_one_opponent = max_time_per_one_opponent
        self.max_frames_per_episode = max_frames_per_episode
        self.max_frames = max_frames
        self.max_episodes = max_episodes
        self.is_training = is_training

        self.thread = threading.Thread(target=self.run, args=())

        self.thread.daemon = True                            # Daemonize thread
        self.is_running = True
        self.is_start = False

        self.replay_dir = replay_dir

    def start(self):
        self.is_start = True
        self.thread.start()

    # background
    def run(self):
        try:
            self.is_running = True

            """A run loop to have agents and an environment interact."""
            total_frames = 0
            total_episodes = 0

            # the total numberv_for_all_episodes as [loss, draw, win]
            results = [0, 0, 0]

            # statistic list
            food_used_list, army_count_list, collected_points_list, used_points_list, killed_points_list, steps_list = [], [], [], [], [], []

            training_start_time = time()
            print("start_time before training:", strftime("%Y-%m-%d %H:%M:%S", localtime(training_start_time)))

            # use max_episodes to end the loop
            while time() - training_start_time < self.max_time_for_training:
                agents = [self.player]

                with self.create_env_one_player(self.player) as env:

                    # set the obs and action spec
                    observation_spec = env.observation_spec()
                    action_spec = env.action_spec()

                    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                        agent.setup(obs_spec, act_spec)

                    # self.teacher.setup(self.player.agent.obs_spec, self.player.agent.action_spec)

                    print('player:', self.player) if debug else None
                    print('opponent:', "Computer bot") if debug else None

                    trajectory = []
                    opponent_start_time = time()  # in seconds.
                    print("start_time before reset:", strftime("%Y-%m-%d %H:%M:%S", localtime(opponent_start_time)))

                    # one opponent match (may include several games) defaultly lasts for no more than 2 hour
                    while time() - opponent_start_time < self.max_time_per_one_opponent:

                        # Note: the pysc2 environment don't return z
                        # AlphaStar: home_observation, away_observation, is_final, z = env.reset()
                        total_episodes += 1
                        print("total_episodes:", total_episodes)

                        timesteps = env.reset()
                        for a in agents:
                            a.reset()

                        [home_obs] = timesteps
                        is_final = home_obs.last()

                        player_memory = self.player.agent.initial_state()
                        # teacher_memory = self.teacher.initial_state()

                        torch.manual_seed(total_episodes)
                        np.random.seed(total_episodes)

                        # initial build order
                        player_bo = []

                        episode_frames = 0
                        # default outcome is 0 (means draw)
                        outcome = 0

                        # initial last list
                        last_list = [0, 0, 0]

                        # in one episode (game)
                        start_episode_time = time()  # in seconds.
                        print("start_episode_time before is_final:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_episode_time)))

                        while not is_final:
                            total_frames += 1
                            episode_frames += 1

                            t = time()

                            with torch.no_grad():
                                state = self.player.agent.agent_nn.preprocess_state_all(home_obs.observation, 
                                                                                        build_order=player_bo, 
                                                                                        last_list=last_list)
                                # FIX(T6): 原实现在这里多调用了一次 step_from_state，
                                # 但返回值 player_step 从头到尾没有被读取（grep 全文件只有这一处赋值）
                                # —— 纯浪费一次完整前向，实测约 30 ms/帧，等于评估速度砍半。
                                #
                                # 为什么删掉是安全的（有实测支撑，不是想当然）：
                                #   _check_t6_double_step.py [A] 验证 forward **不会** in-place 修改
                                #   传入的 hidden_state（调用前后 max|Δ| = 0.000e+00），
                                #   所以第一次调用不会污染 player_memory；
                                #   两次调用共用同一个 state，preprocess 也只做一次。
                                #   -> 删掉后行为完全等价，只是少算一次。
                                player_function_call, player_action, player_logits, \
                                    player_new_memory, player_select_units_num, entity_num = self.player.agent.step_from_state(state, 
                                                                                                                               player_memory, 
                                                                                                                               obs=home_obs.observation)

                            # 第 79 轮（DS，#29）：dump 评测时喂给网络的 state 形状/统计（MAS_DUMP_STATE=1）
                            # ⚠️ 必须放在**完整赋值语句之后**：第一次实现插在了续行中间，
                            #    把 `player_function_call, player_action, ... = step_from_state(...)`
                            #    这段多行赋值打断了，直接 NameError、整局作废（Episode: 0）。
                            if os.environ.get("MAS_DUMP_STATE") and not getattr(self, "_state_dumped", False):
                                self._state_dumped = True
                                try:
                                    print("[STATE-DUMP] entity=%s map=%s scalar_n=%d"
                                          % (tuple(state.entity_state.shape),
                                             tuple(state.map_state.shape),
                                             len(state.statistical_state)), flush=True)
                                    print("[STATE-DUMP] entity absmax=%.3f mean=%.3f nonzero=%.1f%%"
                                          % (float(state.entity_state.abs().max()),
                                             float(state.entity_state.mean()),
                                             100.0 * float((state.entity_state != 0).float().mean())), flush=True)
                                    print("[STATE-DUMP] entity_num=%s select_units_num=%s"
                                          % (entity_num, player_select_units_num), flush=True)
                                except Exception as _e:
                                    print("[STATE-DUMP] 失败: %r" % (_e,), flush=True)

                            # 第 82 轮（DS，#38）：可选把 unit_tags / target_unit_tag 从**下标**翻译成**真 tag**。
                            # 依据：agent.py:358 注释明确说这两者"是 _raw_tags 里的下标，不等于 SC2 用的 tag"，
                            # 而本文件把 func_call 直接送进 env（env_actions = [player_function_call]）——
                            # #37 实测：真实 tag 是 8.6e9 量级，而 func_call 里是 [10] 这种小下标。
                            # 若 pysc2 不做映射，则所有带 unit_tags 的指令（Train_/Build_/Smart/Move/Attack）
                            # 都是空指令 —— 那会完美解释"军队不少但打不死"。
                            # 用一个开关做 A/B，避免改坏既有行为。
                            if os.environ.get("MAS_TAG_MAP"):
                                try:
                                    from pysc2.lib import actions as _A
                                    _tags = [int(u.tag) for u in home_obs.observation.raw_units]
                                    _names = [a.name for a in _A.RAW_FUNCTIONS[player_function_call.function].args]
                                    _new = []
                                    for _n, _v in zip(_names, player_function_call.arguments):
                                        if _n in ("unit_tags", "target_unit_tag"):
                                            _new.append([_tags[int(i)] if 0 <= int(i) < len(_tags) else int(i)
                                                         for i in _v])
                                        else:
                                            _new.append(_v)
                                    # FunctionCall 是 namedtuple: (function, arguments)
                                    player_function_call = _A.FunctionCall(
                                        player_function_call.function, _new)
                                except Exception as _e:
                                    print("[TAG-MAP] 失败: %r" % (_e,), flush=True)

                            # 第 83 轮（DS，#44）：记录"军队/人口随时间"的曲线。
                            # 目的：判断我们是"造不出来"（产能/人口瓶颈）还是"造了但打不赢"。
                            # 每 2000 步记一次 (step, food_army, food_workers, minerals, army_count)。
                            if os.environ.get("MAS_TIME_STATS"):
                                try:
                                    _pc = home_obs.observation.player_common
                                    _st = int(home_obs.observation.game_loop[0])
                                    _last = getattr(self, "_tl_last", -1)
                                    if _st - _last >= 2000:
                                        self._tl_last = _st
                                        self._tl = getattr(self, "_tl", [])
                                        self._tl.append((_st, int(_pc.food_army), int(_pc.food_workers),
                                                         int(_pc.minerals), int(_pc.army_count)))
                                except Exception:
                                    pass

                            # 第 82 轮（DS，#42）：统计 Attack_pt 的目标坐标分布。
                            # 背景：Attack_pt 的参数是 [queued, unit_tags, **world**] —— 它是"攻击移动"，
                            # 位置由 location 头给出。若坐标总在自己基地附近，就永远打不到对手基地。
                            # 这里只统计"攻击坐标到己方建筑质心的距离"的均值/max。
                            if os.environ.get("MAS_LOC_STATS"):
                                try:
                                    _f = int(player_function_call.function)
                                    if _f == 2:      # Attack_pt
                                        _ru = home_obs.observation.raw_units
                                        _mine = [u for u in _ru if u.alliance == 1]
                                        _bld = [u for u in _mine if u.unit_type in (59, 60, 61, 62, 72, 73)]
                                        if _mine and _bld:
                                            _cx = sum(u.x for u in _bld) / len(_bld)
                                            _cy = sum(u.y for u in _bld) / len(_bld)
                                            _args = player_function_call.arguments
                                            _w = None
                                            for _a in _args:
                                                if isinstance(_a, (list, tuple)) and len(_a) == 2:
                                                    _w = _a
                                            if _w is not None:
                                                _d = ((_w[0] - _cx) ** 2 + (_w[1] - _cy) ** 2) ** 0.5
                                                self._loc_d = getattr(self, "_loc_d", [])
                                                self._loc_d.append(_d)
                                except Exception:
                                    pass
                            # 第 82 轮（DS，#37）：核对"下标 vs 真 tag"。
                            # 背景：agent.py:358 的注释明确写 "unit_tags and target_unit_tag are actually
                            # index in _raw_tags! so they are different from the tags really used by a SC2-action!"
                            # 而 rl_eval_sl 把 func_call 直接送进 env.step（下面 env_actions = [player_function_call]），
                            # 中间**看不到任何 index -> tag 的转换**。
                            # 这里实测：func_call 里的 unit_tags/target_unit_tag 到底是不是 obs 里的真 tag。
                            if os.environ.get("MAS_TAG_CHECK") and getattr(self, "_tag_checked", 0) < 3:
                                self._tag_checked = getattr(self, "_tag_checked", 0) + 1
                                try:
                                    _raw_tags = [int(u.tag) for u in home_obs.observation.raw_units]
                                    _args = [list(a) if hasattr(a, "__iter__") else a
                                             for a in player_function_call.arguments]
                                    _flat = []
                                    for _a in _args:
                                        if isinstance(_a, list):
                                            _flat.extend(int(x) for x in _a)
                                        else:
                                            _flat.append(int(_a))
                                    _hit = [x for x in _flat if x in _raw_tags]
                                    print("[TAG-CHECK] func=%s args=%s | n_raw_units=%d tag_range=[%s..%s] "
                                          "| 参数中命中真实 tag 的个数=%d"
                                          % (int(player_function_call.function), _args, len(_raw_tags),
                                             min(_raw_tags) if _raw_tags else -1,
                                             max(_raw_tags) if _raw_tags else -1, len(_hit)),
                                          flush=True)
                                    print("[TAG-CHECK] 真实 tag 前 8 个: %s" % (_raw_tags[:8],), flush=True)
                                except Exception as _e:
                                    print("[TAG-CHECK] 失败: %r" % (_e,), flush=True)

                            print("player_function_call:", player_function_call) if not SAVE_STATISTIC else None
                            print("player_action:", player_action) if debug else None
                            print("player_action.delay:", player_action.delay) if debug else None
                            print("player_select_units_num:", player_select_units_num) if debug else None

                            if True:
                                show_sth(home_obs, player_action)

                            # 第 79 轮（DS，#28）：累计本局实际输出的动作类型
                            if os.environ.get("MAS_ACT_STATS"):
                                try:
                                    _k = int(player_action.action_type.item())
                                    self._act_cnt[_k] = self._act_cnt.get(_k, 0) + 1
                                except Exception:
                                    pass

                            expected_delay = player_action.delay.item()
                            step_mul = max(1, expected_delay)
                            print("step_mul:", step_mul) if debug else None

                            print('run_loop, t1', time() - t) if speed else None
                            t = time()

                            # if False:
                            #     teacher_step = self.teacher.step_based_on_actions(state, teacher_memory, player_action, player_select_units_num)
                            #     teacher_logits, teacher_select_units_num, teacher_new_memory = teacher_step
                            #     print("teacher_logits:", teacher_logits) if debug else None
                            # else:
                            #     teacher_logits = None

                            print('run_loop, t2', time() - t) if speed else None
                            t = time()

                            env_actions = [player_function_call]

                            print('run_loop, t3', time() - t) if speed else None
                            t = time()

                            if USE_PREDICT_STEP_MUL:
                                timesteps = env.step(env_actions, step_mul=step_mul)  # STEP_MUL step_mul
                            else:
                                timesteps = env.step(env_actions, step_mul=STEP_MUL)
                            [home_next_obs] = timesteps
                            reward = home_next_obs.reward
                            print("reward: ", reward) if debug else None

                            print('run_loop, t4', time() - t) if speed else None
                            t = time()

                            is_final = home_next_obs.last()

                            # calculate the build order
                            player_bo = L.calculate_build_order(player_bo, home_obs.observation, home_next_obs.observation)
                            print("player build order:", player_bo) if debug else None

                            print('run_loop, t5', time() - t) if speed else None
                            t = time()

                            # calculate the unit counts of bag
                            # player_ucb = L.calculate_unit_counts_bow(home_obs.observation).reshape(-1).numpy().tolist()
                            # print("player unit count of bow:", sum(player_ucb)) if debug else None

                            print('run_loop, t6', time() - t) if speed else None
                            t = time()

                            game_loop = home_obs.observation.game_loop[0]
                            print("game_loop", game_loop) if debug else None

                            # note, original AlphaStar pseudo-code has some mistakes, we modified 
                            # them here
                            traj_step = None
                            if self.is_training:
                                trajectory.append(traj_step)

                            player_memory = tuple(h.detach() for h in player_new_memory)
                            home_obs = home_next_obs
                            last_delay = expected_delay
                            last_action_type = player_action.action_type.item()
                            last_repeat_queued = player_action.queue.item()
                            last_list = [last_delay, last_action_type, last_repeat_queued]

                            if is_final:
                                outcome = reward
                                print("outcome: ", outcome) if debug else None

                                if SAVE_REPLAY:
                                    env.save_replay(self.replay_dir)

                                if SAVE_STATISTIC:
                                    o = home_next_obs.observation
                                    p = o['player']

                                    food_used = p['food_used']
                                    army_count = p['army_count']

                                    print('food_used', food_used)
                                    print('army_count', army_count)

                                    collected_minerals = np.sum(o['score_cumulative']['collected_minerals'])
                                    collected_vespene = np.sum(o['score_cumulative']['collected_vespene'])

                                    print('collected_minerals', collected_minerals)
                                    print('collected_vespene', collected_vespene)

                                    collected_points = collected_minerals + collected_vespene

                                    used_minerals = np.sum(o['score_by_category']['used_minerals'])
                                    used_vespene = np.sum(o['score_by_category']['used_vespene'])

                                    print('used_minerals', used_minerals)
                                    print('used_vespene', used_vespene)

                                    used_points = used_minerals + used_vespene

                                    killed_minerals = np.sum(o['score_by_category']['killed_minerals'])
                                    killed_vespene = np.sum(o['score_by_category']['killed_vespene'])

                                    print('killed_minerals', killed_minerals)
                                    print('killed_vespene', killed_vespene)

                                    killed_points = killed_minerals + killed_vespene

                                    if killed_points > 4000:
                                        outcome = 1

                                    food_used_list.append(food_used)
                                    army_count_list.append(army_count)
                                    collected_points_list.append(collected_points)
                                    used_points_list.append(used_points)
                                    killed_points_list.append(killed_points)
                                    steps_list.append(game_loop)

                                    with open(OUTPUT_FILE, 'a') as file:
                                        end_episode_time = time()  # in seconds.
                                        end_episode_time = strftime("%Y-%m-%d %H:%M:%S", localtime(start_episode_time))

                                        # 第 76 轮（GLM，#21）：加 native outcome 字段。
                                        # 注意 outcome 默认 0（截断=平局），终局才有真实值。
                                        statistic = 'Episode: [{}/{}]| native_outcome: {} | food_used: {:.1f} | army_count: {:.1f} | collected_points: {:.1f} | used_points: {:.1f} | killed_points: {:.1f} | steps: {:.3f}s \n'.format(
                                            total_episodes, MAX_EPISODES, int(outcome), food_used, army_count, collected_points, used_points, killed_points, game_loop)

                                        statistic = end_episode_time + " " + statistic

                                        file.write(statistic)

                                        # 第 79 轮（DS，#28）：本局实际动作分布（top6）
                                        # 第 82 轮（DS，#39）：MAS_ACT_STATS=2 时 dump **全量**动作直方图，
                                        # 用于与官方模型逐动作对比（top6 看不出"缺了哪一类动作"）。
                                        if os.environ.get("MAS_TIME_STATS") and getattr(self, "_tl", None):
                                            file.write("TIMELINE " + " ".join(
                                                "%d:%d/%d/%d/%d" % t for t in self._tl) + "\n")
                                            self._tl = []
                                            self._tl_last = -1

                                        if os.environ.get("MAS_LOC_STATS") and getattr(self, "_loc_d", None):
                                            _d = self._loc_d
                                            file.write("LOCSTATS n=%d mean=%.1f max=%.1f p90=%.1f\n"
                                                       % (len(_d), sum(_d) / len(_d), max(_d),
                                                          sorted(_d)[int(0.9 * (len(_d) - 1))]))
                                            self._loc_d = []

                                        if os.environ.get("MAS_ACT_STATS") == "2" and self._act_cnt:
                                            _tot2 = max(sum(self._act_cnt.values()), 1)
                                            _all = sorted(self._act_cnt.items(), key=lambda x: -x[1])
                                            file.write("ACTFULL n=%d " % _tot2 +
                                                       " ".join("%d:%d" % (k2, v2) for k2, v2 in _all) + "\n")

                                        if os.environ.get("MAS_ACT_STATS") and self._act_cnt:
                                            _tot = max(sum(self._act_cnt.values()), 1)
                                            _top = sorted(self._act_cnt.items(), key=lambda x: -x[1])[:6]
                                            file.write("ACTSTATS n=%d " % _tot +
                                                       " ".join("%d:%.1f%%" % (k2, 100.0 * v2 / _tot)
                                                                 for k2, v2 in _top) + "\n")
                                            self._act_cnt = {}

                                results[outcome + 1] += 1

                            if self.is_training and len(trajectory) >= AHP.sequence_length:                    
                                trajectories = RU.stack_namedtuple(trajectory)

                                if self.player.learner is not None:

                                    if self.player.learner.is_running:
                                        self.player.learner.send_trajectory(trajectories)                                     
                                        print("Learner send_trajectory!") if debug else None

                                        trajectory = []
                                    else:
                                        print("Learner stops!")

                                        print("Actor also stops!")
                                        raise Exception

                            # use max_frames to end the loop
                            # whether to stop the run
                            if self.max_frames and total_frames >= self.max_frames:
                                print("Beyond the max_frames, return!")
                                raise Exception

                            # use max_frames_per_episode to end the episode
                            if self.max_frames_per_episode and episode_frames >= self.max_frames_per_episode:
                                print("Beyond the max_frames_per_episode, break!")
                                break

                        #self.coordinator.send_outcome(self.player, self.opponent, outcome)

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")
                            raise Exception

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            # 第 76 轮（GLM，#21）：这两行原来在 `if debug` 里 —— 评测模式下不打印，
            # 编排器就汇总不到 results（第一版验证局就是在这里扑空的）。
            print("results: ", results, flush=True)
            win_rate = results[2] / (1e-9 + sum(results))
            print("win rate: ", win_rate, flush=True)

            total_time = time() - training_start_time

            if SAVE_STATISTIC: 
                with open(OUTPUT_FILE, 'a') as file:
                    statistic = 'Avg: [{}/{}] | win_rate: {:.1f} | food_used: {:.1f} | army_count: {:.1f} | std(army_count): {:.1f} | collected_points: {:.1f} | used_points: {:.1f} | killed_points: {:.1f} | steps: {:.3f} | Total time: {:.3f}s \n'.format(
                        total_episodes, MAX_EPISODES, win_rate, np.mean(food_used_list), np.mean(army_count_list), np.std(army_count_list), np.mean(collected_points_list),
                        np.mean(used_points_list), np.mean(killed_points_list), np.mean(steps_list), total_time)

                    print("statistic: ", statistic) if SAVE_STATISTIC else None

                    file.write(statistic)

            self.is_running = False

    # create env function
    def create_env_one_player(self, player, game_steps_per_episode=GAME_STEPS_PER_EPISODE, 
                              step_mul=STEP_MUL, version=VERSION, 
                              map_name=MAP_NAME, random_seed=RANDOM_SEED):

        player_aif = AgentInterfaceFormat(**AAIFP._asdict())
        agent_interface_format = [player_aif]

        # create env
        print('map name:', map_name) 
        print('player.name:', player.name)
        print('player.race:', player.race)

        sc2_computer = Bot([Race.terran],
                           Difficulty(DIFFICULTY),
                           [BotBuild.random])

        env = SC2Env(map_name=map_name,
                     players=[Agent(player.race, player.name),
                              sc2_computer],
                     step_mul=step_mul,
                     game_steps_per_episode=game_steps_per_episode,
                     agent_interface_format=agent_interface_format,
                     version=version,
                     random_seed=random_seed)

        return env


def show_sth(home_obs, player_action):
    obs = home_obs.observation
    raw_units = obs["raw_units"]

    selected_units = player_action.units.reshape(-1)

    selected_units = selected_units.detach().cpu().numpy().tolist()
    print('selected_units', selected_units) if debug else None

    unit_type_list = []
    for i in selected_units:
        if i < len(raw_units):
            u = raw_units[i]
            unit_type = u.unit_type
            unit_type_name = sc2_units.get_unit_type(unit_type).name
            unit_type_list.append(unit_type_name)
        else:
            print('find a EOF!') if debug else None     

    print('unit_type_list', unit_type_list) if not SAVE_STATISTIC else None     


def test(on_server=False, replay_path=None):
    device = DEVICE

    league = League(
        initial_agents={
            race: get_supervised_agent(race, path=MODEL_PATH, model_type=MODEL_TYPE, restore=RESTORE, device=device)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=0,
        league_exploiters=0)

    coordinator = Coordinator(league, output_file=OUTPUT_FILE, winrate_scale=2)
    learners = []
    actors = []

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = None
        learners.append(learner)
        actors.extend([ActorEval(player, coordinator) for _ in range(ACTOR_NUMS)])

    threads = []
    # for l in learners:
    #     l.start()
    #     threads.append(l.thread)
    #     sleep(1)
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
