#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by fighting against built-in AI (computer), using no replays"

import json
import os
import random
import sys
import traceback
from time import time, sleep, strftime, localtime
import threading

import numpy as np

import torch

# ---- 第 84 轮（DS，#80）：确定性解码开关（MAS_GREEDY=1） ----
# 动机：BC 的 val action acc 是 0.84–0.97，意味着**每步有 3%–16% 概率选错**；
# 一局 2625 步，累积下来足以把局面带离演示分布。采样噪声是不是主因，可以用 A/B 判。
# 做法：只在本进程内把 torch.multinomial 换成 argmax，**不改六个 head 的任何代码**，
# 变量只有"解码方式"一个，避免为了做对照引入别的改动。
# 判据：同一 checkpoint、同一批种子，MAS_GREEDY=1 与 =0 的真 native 胜率与 killed 对比。
if os.environ.get("MAS_GREEDY", "0") == "1":
    _orig_multinomial = torch.multinomial

    def _greedy_multinomial(input, num_samples=1, replacement=False, **kwargs):
        idx = torch.argmax(input, dim=-1, keepdim=True)
        if num_samples == 1:
            return idx
        return idx.expand(*([-1] * (idx.dim() - 1)), num_samples)

    torch.multinomial = _greedy_multinomial
    print("[eval] MAS_GREEDY=1：torch.multinomial 已替换为 argmax（确定性解码）", flush=True)

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


# ---- 2026-11-16 RL 技术债 D8：正常收尾不再伪装成 Traceback ----
# 原实现用裸 `raise Exception` 跳出三层循环（max_episodes / max_frames / learner stops），
# 外层 except 一律 print(traceback.format_exc())，于是每次正常打满一局评测都会
# 在日志里留下 Traceback，而 finally 又照常写 Avg 行 —— 排障时无法区分
# "正常收尾"和"真出错"。这里用专用信号类区分两者：finally 的汇总接口不变，
# 编排器抓的 results/win_rate 行不受影响。
class EvalEarlyStop(Exception):
    """评测达到预设上限时的正常退出信号，不是故障。"""
    pass


SIMPLE_TEST = not P.on_server
if SIMPLE_TEST:
    # ---- 第 76 轮（GLM，#21）：评测参数全部环境变量化 ----
    # 原来写死 MAX_EPISODES=1 / DIFFICULTY=1 / 随机种子 1 / 输出文件固定，
    # 无法做"N 局 + 分片 + 指定 checkpoint"的胜率评测（Kimi-17 / E0）。
    # 默认值与旧行为完全一致。
    import os as _os
    MAX_EPISODES = int(_os.environ.get("EVAL_EPISODES", "1"))
    # 第 84 轮（DS，#76）：评测时长可配。
    # 依据（#65）：60 局演示的时长是 5766–17952 loops、中位 6754，而这里固定跑 21000 loops，
    # 比任何一局演示都长。要把"模型慢"与"模型不会收尾"分开，必须能在演示长度上单独测一次。
    # 默认 21000 保持历史可比；对照用 EVAL_MAX_LOOPS=6800。
    GAME_STEPS_PER_EPISODE = int(_os.environ.get("EVAL_MAX_LOOPS", "21000"))
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

# ---- 第 84 轮（DS，#66）：评测局独立化 + native/custom 双口径 ----
# 问题（Hunyuan H1 弱形式的实测版）：原实现 create_env 在**局循环之外**，
# 整个分片共用同一个 SC2 random_seed；策略侧 torch.manual_seed(total_episodes)
# 的 total_episodes 每分片从 1 重数。结果：同分片内对局初始条件完全相同。
# 实测 ICC(胜负)=0.50（官方 RL，n=28）→ design effect 4.0 → 有效样本仅 7。
# 修法：每局重建 env 且 seed = RANDOM_SEED + episode_index（默认开启，可用
# EVAL_SEED_PER_EP=0 退回旧行为做复现对照）。
SEED_PER_EPISODE = _os.environ.get("EVAL_SEED_PER_EP", "1") == "1"
EVAL_SEED_STRIDE = int(_os.environ.get("EVAL_SEED_STRIDE", "1"))

# 问题（Hunyuan H2 的实测升级版）：`outcome = reward` 后被
# `if killed_points > 4000: outcome = 1` 改写成 1，而打印字段仍叫 native_outcome。
# 实测全库 43 个"胜"里 35 个停在帧上限（截断局），冠军 216 局真胜 0。
# 修法：native 与 custom 分开计数、分开打印；results 只统计 native。
CUSTOM_KILL_THRESH = float(_os.environ.get("EVAL_CUSTOM_KILL", "4000"))

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
        # 第 84 轮（DS，#77）：攻击点诊断。存 (x, y, 到最近敌方单位的距离, 敌方单位数)
        self._atk = []
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
            results_custom = [0, 0, 0]   # #66：killed>阈值 改判后的口径，仅作参考

            # statistic list
            food_used_list, army_count_list, collected_points_list, used_points_list, killed_points_list, steps_list = [], [], [], [], [], []

            training_start_time = time()
            print("start_time before training:", strftime("%Y-%m-%d %H:%M:%S", localtime(training_start_time)))

            # use max_episodes to end the loop
            while time() - training_start_time < self.max_time_for_training:
                agents = [self.player]

                # #66：每局一个独立的 SC2 种子（旧行为 = 整个分片共用一个种子）
                _env_seed = RANDOM_SEED + (total_episodes * EVAL_SEED_STRIDE if SEED_PER_EPISODE else 0)
                with self.create_env_one_player(self.player, random_seed=_env_seed) as env:

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

                        # #66：策略随机流也按 (分片种子, 局号) 复合，旧版只按局号
                        # → 所有分片的第 i 局共用同一条随机流。
                        if SEED_PER_EPISODE:
                            _policy_seed = RANDOM_SEED * 1000 + total_episodes
                            torch.manual_seed(_policy_seed)
                            np.random.seed(_policy_seed)
                        else:
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

                            # ---- 第 84 轮（DS，#81）：oracle 选择消融（诊断用，不是模型能力） ----
                            # 依据（#77/#77b，seed 971，n=168）：攻击点瞄得准（mean_d=7.7），
                            # 但每次攻击选中的 9.5 个单位里只有 2.3 个战斗单位，
                            # **61% 的攻击命令一个战斗单位都没选中** → 兵在家、命令发给了农民。
                            # 这里把"选中集合"直接换成全部战斗单位，只改这一件事。
                            # 判读：若 killed 显著上升/出现真胜 → 选择头是瓶颈（因果证据）；
                            #       若仍不动 → 瓶颈在别处（目标点、交战结算），本条否证。
                            # ⚠️ 这是 oracle 干预，**不能**当作模型改进对外报。
                            # 第 85 轮（DS，#90）：`MAS_SEL_GUARD=1` —— 选择性兜底（可部署的解码规则，不是 oracle）。
                            # 依据（#89 SEL-DUMP 实测）：模型对 Attack_pt 只选 **1 个**实体，而且是 Probe，
                            # 甚至出现选中 MineralField 的情况；槽位 1 是 EOF（= entity_num），被正确排除。
                            # 也就是说"有兵不打"的机制是：**攻击命令只发给了一个农民**。
                            # 规则：只在"模型自己选中的实体里没有任何战斗单位"时才兜底成全部战斗单位。
                            # ⚠️ 这仍然是人给的先验，对外必须标注为「受限解码」，不得说成模型能力。
                            _oracle_on = bool(os.environ.get("MAS_ORACLE_SEL"))
                            _guard_on = bool(os.environ.get("MAS_SEL_GUARD"))
                            if (_oracle_on or _guard_on) and int(player_function_call.function) in (2, 3, 13, 14):
                                try:
                                    from pysc2.lib import actions as _A3
                                    _rw3 = home_obs.observation.raw_units
                                    _combat = []
                                    # 第 93 轮（DS，#105）：战斗单位判定抽成局部函数，
                                    # 供"全量兜底"（旧 guard）与"清洗选中集合"（新 clean）共用。
                                    # 旧版把同一段 13 个类型名的判断写了两遍，容易改漏。
                                    _CT3 = ("Zealot", "Stalker", "Adept", "Immortal", "Sentry",
                                            "Phoenix", "VoidRay", "Carrier", "Colossus",
                                            "Archon", "DarkTemplar", "Oracle", "Tempest")

                                    def _is_combat3(_u):
                                        try:
                                            _nm0 = sc2_units.get_unit_type(_u.unit_type).name
                                        except Exception:
                                            return False
                                        return any(t in _nm0 for t in _CT3)

                                    for _u3 in _rw3:
                                        if int(_u3.alliance) != 1:
                                            continue
                                        if _is_combat3(_u3):
                                            _combat.append(_u3)
                                    # 第 116 轮（DS，#116）：MAE_ARMY_THR=n ——「不够 n 个战斗单位就不出击」。
                                    # 动机（#116 的难度阶梯）：difficulty=3 下 gtb 输在**开局经济崩**，
                                    # 终局快照 food=0 / army=0，即"兵一个个走出去被吃掉，家里没有防守"。
                                    # guard 的原语义是"把**全部**战斗单位派出去"，在 very_easy 上是优点
                                    # （对方不还手），在 d3 上就是送。这里给一个"攒够再打"的闸门：
                                    # 我方战斗单位总数 < n 时不接管选择（让模型自己打），够了才全量派出。
                                    # 计数器 `army_hold/army_go` 会写进 ATKDUMP —— 按 #108 的教训，
                                    # **新规则必须自带执行计数，否则"没报错"不等于"生效了"**。
                                    _n_all3 = len(_combat)
                                    _thr3 = int(os.environ.get("MAS_ARMY_THR", "0") or 0)
                                    # 兜底只在"模型自己一个战斗单位都没选中"时触发；选中了就完全不动。
                                    if _guard_on and not _oracle_on and _combat:
                                        try:
                                            _n_now = int(player_select_units_num.item())
                                        except Exception:
                                            _n_now = int(player_action.units.numel())
                                        _sel_now = player_action.units.reshape(-1).tolist()[:max(1, _n_now)]
                                        # 第 93 轮（DS，#105）：MAS_SEL_GUARD_CLEAN=1 —— 把选中集合里的
                                        # 非战斗单位剔掉，只留战斗单位；剔空了才退回"全部战斗单位"。
                                        # 动机：#89 实测模型对 Attack_pt 常选 "1 个 Probe"，_hit=1 > 0
                                        # 于是旧 guard 完全不介入 —— 但送出去的仍然是农民。
                                        _clean3 = bool(os.environ.get("MAS_SEL_GUARD_CLEAN"))
                                        # 第 93 轮（DS，#106）：MAS_SEL_GUARD_MIN=k —— 参数化"兜底激进程度"。
                                        # 模型自己选中的战斗单位数 >= k 才采用它的选中集合；否则退回"全部战斗单位"。
                                        # k=1 等价于 clean（只要有 1 个能打就用模型的，实测只有 1.9 个 → 越打越弱）；
                                        # k 越大越接近旧 guard（旧 guard 相当于 k=∞）。
                                        # 先用它扫一遍，找"一次派多少兵"的最优点。
                                        _min3 = int(os.environ.get("MAS_SEL_GUARD_MIN", "0") or 0)
                                        _hit = 0
                                        _keep = []
                                        for _i5 in _sel_now:
                                            _i5 = int(_i5)
                                            if 0 <= _i5 < len(_rw3) and _is_combat3(_rw3[_i5]):
                                                _hit += 1
                                                _keep.append(_rw3[_i5])
                                        if _min3 > 0:
                                            if _hit >= _min3:
                                                _combat = _keep          # 模型选的够多 → 用它的
                                            # else: _combat 保持"全部战斗单位"（兜底）
                                            self._guard_fired = getattr(self, "_guard_fired", 0) + (1 if _hit < _min3 else 0)
                                        else:
                                            if _hit > 0:
                                                _combat = _keep if _clean3 else []
                                                if _clean3:
                                                    self._guard_clean_n = getattr(self, "_guard_clean_n", 0) + 1
                                            self._guard_fired = getattr(self, "_guard_fired", 0) + (1 if _hit == 0 else 0)
                                        self._guard_seen = getattr(self, "_guard_seen", 0) + 1

                                    if _thr3 > 0:
                                        if _n_all3 < _thr3:
                                            self._army_hold = getattr(self, "_army_hold", 0) + 1
                                            _combat = []
                                        else:
                                            self._army_go = getattr(self, "_army_go", 0) + 1

                                    if _combat:
                                        _maxsel = int(player_action.units.numel())
                                        # 第 93 轮（DS，#105）：MAS_SEL_MAX_SEL=k —— 覆盖"送多少个单位"的上限。
                                        # 原上限 = 模型动作张量的槽位数（AHP.max_selected=12）。
                                        # 军队 23 个战斗单位时只送 12 个，可能打不完（54.6% 的对局打满帧数）。
                                        _ovr3 = int(os.environ.get("MAS_SEL_MAX_SEL", "0") or 0)
                                        if _ovr3 > 0:
                                            _maxsel = _ovr3
                                        _combat = _combat[:_maxsel]
                                        _all_tags = [int(u.tag) for u in _rw3]
                                        _tagset = set(_all_tags)
                                        _names3 = [a.name for a in
                                                   _A3.RAW_FUNCTIONS[player_function_call.function].args]
                                        _new3 = []
                                        for _n3, _v3 in zip(_names3, player_function_call.arguments):
                                            if _n3 == "unit_tags":
                                                if any(int(x) in _tagset for x in _v3):
                                                    _new3.append([int(u.tag) for u in _combat])
                                                else:
                                                    _new3.append([_all_tags.index(int(u.tag))
                                                                  for u in _combat])
                                            else:
                                                _new3.append(_v3)
                                        player_function_call = _A3.FunctionCall(
                                            player_function_call.function, _new3)
                                        self._oracle_n = getattr(self, "_oracle_n", 0) + 1
                                except Exception as _e3:
                                    print("[ORACLE-SEL] 失败: %r" % (_e3,), flush=True)

                            # 第 93 轮（DS，#106）：MAS_TARGET_BUILDING=1 —— 诊断用（**受限解码，不是模型能力**）。
                            # 依据（#106 冒烟）：一局 186 次攻击、mean_d=5.8（瞄得很准）、killed 6100、army 27，
                            # 但 native=0 且打满 20992 帧 —— 兵打死了却**拆不掉家**。
                            # SC2 的胜利条件是摧毁对方所有建筑，而现有攻击点瞄的是"最近的敌方单位"。
                            # 本开关只改 Attack_pt 的 world 参数：改成"离我方部队最近的敌方建筑"。
                            if os.environ.get("MAS_TARGET_BUILDING") and int(player_function_call.function) == 2:
                                try:
                                    from pysc2.lib import actions as _A4
                                    _rw4 = home_obs.observation.raw_units
                                    self._tb_seen = getattr(self, "_tb_seen", 0) + 1
                                    # 第 93 轮（DS，#108 修正）：原来只写了 Protoss 建筑名，
                                    # 而对手是 **Terran**（--bot-race terran）→ `_bld4` 恒空，
                                    # 整条分支静默失效（smoke_tb 的 target_bld 计数为 0，但没有报错）。
                                    # 现在把 Terran / Zerg / Protoss 的建筑名都列上。
                                    _BT4 = ("Nexus", "Pylon", "Gateway", "Assimilator", "CyberneticsCore",
                                            "Forge", "PhotonCannon", "Stargate", "RoboticsFacility",
                                            "TwilightCouncil", "TemplarArchives", "FleetBeacon",
                                            "RoboticsBay", "DarkShrine", "ShieldBattery",
                                            "CommandCenter", "SupplyDepot", "Barracks", "Refinery",
                                            "Factory", "Starport", "EngineeringBay", "Bunker",
                                            "MissileTurret", "SensorTower", "Armory", "GhostAcademy",
                                            "FusionCore", "PlanetaryFortress", "OrbitalCommand",
                                            "Reactor", "TechLab",
                                            "Hatchery", "Lair", "Hive", "Extractor", "SpawningPool",
                                            "EvolutionChamber", "RoachWarren", "BanelingNest",
                                            "SpineCrawler", "SporeCrawler", "Spire", "GreaterSpire",
                                            "NydusNetwork", "InfestationPit", "UltraliskCavern",
                                            "HydraliskDen", "LurkerDen")
                                    _bld4, _ours4 = [], []
                                    for _u4 in _rw4:
                                        try:
                                            _nm4 = sc2_units.get_unit_type(_u4.unit_type).name
                                        except Exception:
                                            continue
                                        _isb4 = any(t in _nm4 for t in _BT4)
                                        if int(_u4.alliance) != 1:
                                            if _isb4:
                                                _bld4.append(_u4)
                                        elif not _isb4:
                                            _ours4.append(_u4)
                                    if _bld4 and _ours4:
                                        _cx4 = sum(float(u.x) for u in _ours4) / len(_ours4)
                                        _cy4 = sum(float(u.y) for u in _ours4) / len(_ours4)
                                        # 第 118 轮（DS，#118）：MAS_DEFEND_R=R ——「家附近有敌人就先回防」。
                                        # 动机（#116/#117）：d3 的负局几乎全是 food≈0 / army=0，
                                        # 即"部队全被派出去、家里没有一兵"，而 `MAS_ARMY_THR` 只做了
                                        # "晚点出兵"（按住不动），实测仍被打死（unit 站着不还手）。
                                        # 真正缺的是**回防**：敌方战斗单位进入我方建筑质心 R 格内时，
                                        # 把 Attack_pt 的落点从"最近建筑"改回"最近的入侵者"。
                                        # 计数器 `defend=` 写进 ATKDUMP，证明分支真的执行（#108 的教训）。
                                        _R5 = int(os.environ.get("MAS_DEFEND_R", "0") or 0)
                                        _t4 = None
                                        if _R5 > 0:
                                            _CT5 = ("Zealot", "Stalker", "Adept", "Immortal", "Sentry",
                                                    "Phoenix", "VoidRay", "Carrier", "Colossus",
                                                    "Archon", "DarkTemplar", "Oracle", "Tempest",
                                                    "Marine", "Marauder", "Medivac", "SiegeTank",
                                                    "Hellion", "Viking", "Banshee", "Thor",
                                                    "Zergling", "Roach", "Hydralisk", "Mutalisk")
                                            _near5 = []
                                            for _u5 in _rw4:
                                                if int(_u5.alliance) == 1:
                                                    continue
                                                try:
                                                    _nm5 = sc2_units.get_unit_type(_u5.unit_type).name
                                                except Exception:
                                                    continue
                                                if not any(t in _nm5 for t in _CT5):
                                                    continue
                                                _d5 = ((float(_u5.x) - _cx4) ** 2
                                                       + (float(_u5.y) - _cy4) ** 2)
                                                if _d5 <= _R5 * _R5:
                                                    _near5.append((_d5, _u5))
                                            if _near5:
                                                _near5.sort(key=lambda p: p[0])
                                                _t4 = _near5[0][1]
                                                self._defend_n = getattr(self, "_defend_n", 0) + 1
                                        if _t4 is None:
                                            _t4 = min(_bld4, key=lambda u: (float(u.x) - _cx4) ** 2
                                                      + (float(u.y) - _cy4) ** 2)
                                        _names4 = [a.name for a in
                                                   _A4.RAW_FUNCTIONS[player_function_call.function].args]
                                        _new4 = []
                                        for _n4, _v4 in zip(_names4, player_function_call.arguments):
                                            if _n4 == "world":
                                                _new4.append([float(_t4.x), float(_t4.y)])
                                            else:
                                                _new4.append(_v4)
                                        player_function_call = _A4.FunctionCall(
                                            player_function_call.function, _new4)
                                        self._tb_n = getattr(self, "_tb_n", 0) + 1
                                except Exception as _e4:
                                    print("[TARGET-BLD] 失败: %r" % (_e4,), flush=True)

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

                            # ---- 第 84 轮（DS，#77）：攻击点 vs 敌方单位的距离 ----
                            # 动机：实测 3 局（#64/#76）军队 18–39、人口 70–115，15 分钟只杀 875–1900。
                            # 说明不是"造不出兵"，是"有兵不打/打空"。Attack_pt 的参数是 world 坐标，
                            # 与镜头无关，所以只需量：攻击点离最近的敌方单位多远。
                            # 判据：若 mean_d 很大（>20）而 n 不小 → 攻击点在打空气，问题在坐标/目标选择；
                            #       若 mean_d 很小（<8）却仍无击杀 → 问题在交战结算（单位没被选中/没输出）。
                            if os.environ.get("MAS_ATK_DUMP"):
                                try:
                                    from pysc2.lib import actions as _A2
                                    _fn2 = int(player_function_call.function)
                                    if _fn2 in (2, 3):
                                        _argnames = [a.name for a in _A2.RAW_FUNCTIONS[player_function_call.function].args]
                                        _pt = None
                                        for _n2, _v2 in zip(_argnames, player_function_call.arguments):
                                            if _n2 == "world":
                                                _pt = (float(_v2[0]), float(_v2[1]))
                                        if _pt is not None:
                                            _enemy = [(float(u.x), float(u.y))
                                                      for u in home_obs.observation.raw_units
                                                      if int(u.alliance) != 1]
                                            # #77b：攻击时**选中的单位**里有多少是战斗单位。
                                            # 依据：seed 970 一局 176 次 Attack_pt，mean_d=5.9（瞄得准），
                                            # 却只杀 1350。目标没问题 -> 怀疑"下命令的兵"不对
                                            # （选中了农民/没选中/选中已死单位）。
                                            _rw = home_obs.observation.raw_units
                                            _sel = player_action.units.reshape(-1).tolist()
                                            _nsel = 0
                                            _ncombat = 0
                                            for _ix in _sel:
                                                if 0 <= int(_ix) < len(_rw):
                                                    _nsel += 1
                                                    try:
                                                        _nm = sc2_units.get_unit_type(
                                                            _rw[int(_ix)].unit_type).name
                                                    except Exception:
                                                        _nm = ""
                                                    if any(t in _nm for t in (
                                                            "Zealot", "Stalker", "Adept", "Immortal",
                                                            "Sentry", "Phoenix", "VoidRay", "Carrier",
                                                            "Colossus", "Archon", "DarkTemplar",
                                                            "Oracle", "Tempest")):
                                                        _ncombat += 1
                                            if _enemy:
                                                _d = min(((_pt[0] - ex) ** 2 + (_pt[1] - ey) ** 2) ** 0.5
                                                         for ex, ey in _enemy)
                                                self._atk.append((_pt[0], _pt[1], _d, _nsel, _ncombat))
                                            else:
                                                self._atk.append((_pt[0], _pt[1], -1.0, _nsel, _ncombat))

                                            # 第 85 轮（DS，#89）：把"模型到底选了哪几个下标"直接打出来。
                                            # 目的：sel_acc 长期停在 1–4%，需要分辨是"随机选"还是"按位置 0,1,2.. 选"
                                            # （后者说明选择头退化成了"按序输出"，是标签/特征问题而不是训练不足）。
                                            if _os.environ.get("MAS_SEL_DUMP") and getattr(self, "_sel_dump_n", 0) < 8:
                                                self._sel_dump_n = getattr(self, "_sel_dump_n", 0) + 1
                                                _rw4 = home_obs.observation.raw_units
                                                _ty = []
                                                for _i4 in _sel[:12]:
                                                    _i4 = int(_i4)
                                                    if 0 <= _i4 < len(_rw4):
                                                        try:
                                                            _ty.append(sc2_units.get_unit_type(_rw4[_i4].unit_type).name)
                                                        except Exception:
                                                            _ty.append("?")
                                                    else:
                                                        _ty.append("OOR")
                                                print("[SEL-DUMP] step=%d fn=%d n_unit=%d nsel=%d idx=%r types=%r"
                                                      % (int(home_obs.observation.game_loop[0]), _fn2,
                                                         len(_rw4), _nsel, [int(x) for x in _sel[:12]], _ty),
                                                      flush=True)
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

                            # ------------------------------------------------------------------
                            # 第 97 轮（DS，#147）：`MAS_FINISH=1` —— **收尾规则**。
                            #
                            # 与 `MAS_TARGET_BUILDING` 的**本质区别**：`tb` 只改写"已经被选中的 Attack_pt"
                            # 的 world 落点（见上面 536-548 行）；模型若这一步输出 `no_op`/`Train_*`，
                            # `tb` 整段不执行。而 `J44` 实测这个权重的输出 **84–86% 是四类"造"动作**
                            # → `tb` 只能在"模型自己想打"时帮忙瞄准，救不了"压根不下进攻令"。
                            # 本条在动作送入环境前**替换函数 ID 本身**。
                            #
                            # 触发条件（刻意收窄，否则就变成第二个 tb）：
                            #   敌方**战斗单位数 == 0** 且 敌方还有单位（建筑/农民） 且 我方战斗单位 >= N。
                            # 依据：`J49` ⑤（`thr8` 输出最高 194/千帧却只兑现 19%）、
                            #       `J51` ⑥（官方权重 custom-native 的差从 8 局塌到 1 局 = 规则在"收尾"）。
                            #
                            # `MAS_FINISH_FORCE=1` 是**仅用于连通性冒烟**的开关（跳过条件判断），
                            # 只验证"分支能执行、参数构造对、env 不报非法动作"，绝不可用于正式评测。
                            # ------------------------------------------------------------------
                            if os.environ.get("MAS_FINISH") and int(player_function_call.function) != 2:
                                try:
                                    from pysc2.lib import actions as _A5
                                    _CT5 = ("Zealot", "Stalker", "Adept", "Immortal", "Sentry", "Phoenix",
                                            "VoidRay", "Carrier", "Colossus", "Archon", "DarkTemplar",
                                            "Oracle", "Tempest", "Disruptor",
                                            "Marine", "Marauder", "Medivac", "SiegeTank", "Hellion",
                                            "Hellbat", "Viking", "Banshee", "Thor", "Liberator", "Raven",
                                            "Ghost", "Cyclone", "WidowMine",
                                            "Zergling", "Baneling", "Roach", "Hydralisk", "Mutalisk",
                                            "Corruptor", "Ultralisk", "Infestor", "SwarmHost", "Viper",
                                            "BroodLord", "Ravager", "Lurker", "Queen")
                                    _ru5 = home_obs.observation.raw_units
                                    _enemy_combat5, _enemy_any5, _our_combat5 = 0, [], []
                                    _our_idx5 = []          # raw_units 的**下标**（不是 tag，见 :595 的注释）
                                    _cx5 = _cy5 = 0.0
                                    _n5 = 0
                                    for _idx5, _u5 in enumerate(_ru5):
                                        try:
                                            _nm5 = sc2_units.get_unit_type(_u5.unit_type).name
                                        except Exception:
                                            continue
                                        _is_c5 = any(t in _nm5 for t in _CT5)
                                        if int(_u5.alliance) != 1:
                                            if _is_c5:
                                                _enemy_combat5 += 1
                                            else:
                                                _enemy_any5.append(_u5)
                                        elif _is_c5:
                                            _our_combat5.append(_u5)
                                            _our_idx5.append(_idx5)
                                            _cx5 += float(_u5.x)
                                            _cy5 += float(_u5.y)
                                            _n5 += 1
                                    _force5 = bool(os.environ.get("MAS_FINISH_FORCE"))
                                    _thr5 = int(os.environ.get("MAS_FINISH_ARMY", "8") or 8)
                                    # `_our_idx5` 必须非空：pysc2 `features.py:1614` 对 count 型
                                    # 参数的校验是 `1 <= len(arg) <= count`，**空列表直接抛
                                    # ValueError**；而开局只有 Probe/Nexus 时我方没有战斗单位，
                                    # 强制模式若不加这一条，第 1 步就崩（08:26 实测踩中）
                                    _cond5 = _force5 or (_enemy_combat5 == 0 and _enemy_any5
                                                         and len(_our_combat5) >= _thr5)
                                    if _cond5 and not _our_idx5:
                                        # "够格开火但没有可指派的战斗单位" —— 这才是规则不生效的真原因，
                                        # 必须与"条件不满足"分开计数（FINSTAT skip=）
                                        self._fin_skip5 = getattr(self, "_fin_skip5", 0) + 1
                                    if _our_idx5 and _cond5:
                                        if _n5:
                                            _cx5 /= _n5
                                            _cy5 /= _n5
                                        else:
                                            _cx5 = _cy5 = 0.0
                                        if _force5 and not _enemy_any5:
                                            # 强制模式下若连敌人都没有，就拿地图中心当靶子，只为验证参数构造
                                            _tgt5 = None
                                            _wx5, _wy5 = 64.0, 64.0
                                        else:
                                            _tgt5 = min(_enemy_any5,
                                                        key=lambda u: (float(u.x) - _cx5) ** 2
                                                        + (float(u.y) - _cy5) ** 2)
                                            _wx5, _wy5 = float(_tgt5.x), float(_tgt5.y)
                                        # 参数**按名字**填（`RAW_FUNCTIONS[2].args` 的顺序不可假设）
                                        _names5 = [a.name for a in _A5.RAW_FUNCTIONS[2].args]
                                        _args5 = []
                                        for _n5b in _names5:
                                            if _n5b == "queued":
                                                _args5.append([0])
                                            elif _n5b == "unit_tags":
                                                # 必须给下标，否则动作作用于"当前选中单位"，
                                                # 没开 `MAS_SEL_GUARD` 时等于白打（这条不填就是空转）
                                                _args5.append(list(_our_idx5))
                                            elif _n5b == "world":
                                                _args5.append([_wx5, _wy5])
                                            else:
                                                _args5.append([])
                                        player_function_call = _A5.FunctionCall(2, _args5)
                                        self._fin_n = getattr(self, "_fin_n", 0) + 1
                                        self._fin_seen5 = getattr(self, "_fin_seen5", 0) + 1
                                except Exception as _e5:
                                    # #108 的教训：分支静默失效比报错更危险 → 必须打印
                                    print("[FINISH] 失败: %r" % (_e5,), flush=True)

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

                            # 2026-11-20：宏观指标时序采样（每 15 游戏秒），供 macro_metrics.py 计算 APU/PBR/RUR/TR。
                            # 仅当 EVAL_MACRO_TRACE 指向目录（或文件）时生效，默认零开销。
                            _mt = os.environ.get("EVAL_MACRO_TRACE", "").strip()
                            if _mt and (_mt.endswith(os.sep) or os.path.isdir(_mt)):
                                _base = os.path.basename(os.environ.get("EVAL_OUT", "eval.txt"))
                                _mt = os.path.join(_mt, _base.replace(".txt", "") + ".jsonl")
                            if _mt and game_loop - getattr(self, "_macro_last_loop", -10 ** 9) >= 15 * 22.4:
                                self._macro_last_loop = game_loop
                                try:
                                    _ob = home_obs.observation
                                    _pl = _ob.player
                                    _rec = {
                                        "loop": int(game_loop),
                                        "t_s": round(game_loop / 22.4, 1),
                                        "food_used": int(_pl.food_used),
                                        "food_cap": int(_pl.food_cap),
                                        "minerals": int(_pl.minerals),
                                        "vespene": int(_pl.vespene),
                                        "army": int(getattr(_pl, "army_count", -1)),
                                        "idle_workers": int(getattr(_pl, "idle_worker_count", -1)),
                                        "workers": int(getattr(_pl, "food_workers", -1)),
                                    }
                                    # 资源采集量：优先 score_cumulative（pysc2 每个观测都有），
                                    # 再退回 score_by_category（只在终局稳定）。RUR 依赖这两个字段。
                                    _scc = getattr(_ob, "score_cumulative", None)
                                    if _scc is not None:
                                        for _k in ("collected_minerals", "collected_vespene"):
                                            _v = getattr(_scc, _k, None)
                                            if _v is not None:
                                                _rec[_k] = float(_v)
                                    _sc = getattr(_ob, "score_by_category", None)
                                    if _sc is not None:
                                        for _k in ("collected_minerals", "collected_vespene", "killed_minerals"):
                                            if _k in _rec:
                                                continue
                                            try:
                                                _rec[_k] = float(_sc[_k])
                                            except Exception:
                                                pass
                                    try:
                                        _cnt = {}
                                        for _u in _ob.raw_units:
                                            if _u.alliance == 1:
                                                _k = str(int(_u.unit_type))
                                                _cnt[_k] = _cnt.get(_k, 0) + 1
                                        _rec["own"] = _cnt
                                    except Exception:
                                        _rec["own"] = {}
                                    with open(_mt, "a") as _mf:
                                        _mf.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                                except Exception as _e:
                                    try:
                                        import traceback as _tb
                                        with open(str(_mt) + ".err", "a") as _ef:
                                            _ef.write("loop=%s %r\n%s\n" % (game_loop, _e, _tb.format_exc()[:800]))
                                    except Exception:
                                        pass

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

                            # 2026-11-20（hybrid）：规则指挥官 + 动作遮罩。默认关闭，MAS_HYBRID=1 开启。
                            # 每个对局独立进程，所以 _hb 状态挂在 self 上不会串局。
                            if os.environ.get("MAS_HYBRID") == "1":
                                try:
                                    if not hasattr(self, "_hb"):
                                        _hbp = os.path.join(os.getcwd(), "experiments", "hybrid")
                                        if _hbp not in sys.path:
                                            sys.path.insert(0, _hbp)
                                        from arbiter import Arbiter
                                        from alphastarmini.core.arch import agent as _agent_mod
                                        _mode = os.environ.get("MAS_HYBRID_MODE", "rule").lower()
                                        if _mode == "llm":
                                            # 走共享推理服务（8 个对局共用一个 7B，不能各加载一份）
                                            import llm_client as _llmc
                                            from commander import Commander as _Cmd
                                            from obs_to_text import summarize_obs as _summ
                                            _llm = _llmc.build("openai",
                                                               base_url=os.environ.get("LLM_BASE_URL",
                                                                                       "http://127.0.0.1:8765/v1"))
                                            _cmd_obj, _sum_fn = _Cmd(_llm), _summ
                                        else:
                                            from policy import RuleCommander
                                            _cmd_obj, _sum_fn = RuleCommander(), None
                                        self._hb = {"cmd": _cmd_obj, "sum": _sum_fn, "arb": Arbiter(),
                                                    "mod": _agent_mod, "last": -1e9}
                                        self._hb_log = open(os.environ.get("EVAL_OUT", "eval.txt") + ".hybrid.jsonl", "a")
                                        try:
                                            import hashlib as _hl
                                            _fp = {}
                                            for _n in ("arbiter.py", "policy.py", "commander.py", "macro_mask.py"):
                                                for _d in ("experiments/hybrid", "alphastarmini/core/arch"):
                                                    _p2 = os.path.join(_d, _n)
                                                    if os.path.exists(_p2):
                                                        _fp[_n] = _hl.md5(open(_p2, "rb").read()).hexdigest()[:8]
                                                        break
                                            self._hb_log.write(json.dumps(
                                                {"config": {"mode": os.environ.get("MAS_HYBRID_MODE", "rule"),
                                                            "bias": os.environ.get("MAS_HYBRID_BIAS", ""),
                                                            "interval": os.environ.get("MAS_HYBRID_INTERVAL", "8"),
                                                            "weight": os.path.basename(os.environ.get("EVAL_MODEL", "")),
                                                            "code": _fp}}, ensure_ascii=False) + "\n")
                                            self._hb_log.flush()
                                        except Exception:
                                            pass
                                    _hb = self._hb
                                    _t_hb = float(game_loop) / 22.4
                                    _iv = float(os.environ.get("MAS_HYBRID_INTERVAL", "8"))
                                    if _t_hb - _hb["last"] >= _iv:
                                        _hb["last"] = _t_hb
                                        _arg = (_hb["sum"](home_obs.observation)
                                                if _hb.get("sum") else home_obs.observation)
                                        _cmd, _key, _ = _hb["cmd"].propose(_arg)
                                        _res = "none"
                                        # 同类型指令已生效时不再重复提交：否则队列 4 格瞬间塞满，
                                        # 之后全部 rejected（首轮 A/B 里 1070 条指令被拒 621 条）。
                                        if _cmd is not None and _cmd.kind not in _hb["arb"].active:
                                            _res = _hb["arb"].submit(_cmd, _t_hb)
                                        elif _cmd is not None:
                                            _res = "skip_active"
                                        _hb["arb"].tick(_t_hb)
                                        # 两种干预模式：MAS_HYBRID_BIAS=3.0 走软偏置，否则走硬遮罩
                                        _bw = os.environ.get("MAS_HYBRID_BIAS", "").strip()
                                        if _bw and _cmd is not None:
                                            from arbiter import ACTIONS as _ACTS
                                            _ids = _ACTS.get(_cmd.kind, set())
                                            _hb["mod"].set_macro_mask(None)
                                            _hb["mod"].set_macro_bias(_ids, float(_bw))
                                            _mask = None
                                        else:
                                            _hb["mod"].set_macro_bias(None, 0.0)
                                            _mask = _hb["arb"].mask()
                                            _hb["mod"].set_macro_mask(_mask)
                                        try:
                                            self._hb_log.write(json.dumps(
                                                {"t": round(_t_hb, 1), "key": _key,
                                                 "cmd": (_cmd.kind + ":" + _cmd.target) if _cmd else None,
                                                 "res": _res, "active": sorted(_hb["arb"].active.keys()),
                                                 "eff": _hb["arb"].effective_kinds(),
                                                 "mode": os.environ.get("MAS_HYBRID_BIAS") and "bias" or "mask",
                                                 "mask_n": (len(_mask) if _mask else 0)}, ensure_ascii=False) + "\n")
                                            self._hb_log.flush()
                                        except Exception:
                                            pass
                                except Exception as _e:
                                    print("[hybrid] hook failed: %r" % (_e,), flush=True)

                            if is_final:
                                outcome = reward
                                # #66：native 口径先落定，后续任何"改判"都不再回写它。
                                native_outcome = outcome
                                custom_outcome = outcome
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

                                    # #66：原代码是 `outcome = 1`，把原生平局改判成胜，
                                    # 而打印字段仍叫 native_outcome —— 这是 H2 的实质。
                                    # 现在改判只写进 custom_outcome，native 保持 SC2 原生值。
                                    if killed_points > CUSTOM_KILL_THRESH:
                                        custom_outcome = 1

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
                                        # #66：native 与 custom 两个口径同时打印；
                                        # trunc=1 表示这局停在了 game_steps_per_episode 上限，
                                        # 机制上不可能是真胜（实测对照：-1 从不落上限，0 从不提前结束）。
                                        _trunc = int(game_loop >= GAME_STEPS_PER_EPISODE - 2 * STEP_MUL)
                                        statistic = ('Episode: [{}/{}]| native_outcome: {} | custom_outcome: {} '
                                                     '| trunc: {} | sc2_seed: {} | food_used: {:.1f} | army_count: {:.1f} '
                                                     '| collected_points: {:.1f} | used_points: {:.1f} '
                                                     '| killed_points: {:.1f} | steps: {:.3f}s \n').format(
                                            total_episodes, MAX_EPISODES, int(native_outcome), int(custom_outcome),
                                            _trunc, _env_seed,
                                            food_used, army_count, collected_points, used_points, killed_points, game_loop)

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

                                        # #147b：收尾规则的计数**必须无条件落盘**。
                                        # 原先 `fin=` 只在 `_selinfo` 里（条件是 `self._atk` 非空），
                                        # 而"整局没攻击"恰好是最需要看 fin 的情形 → 会出现
                                        # "没有 ATKDUMP 行"就无法判断规则到底触发没有（08:37 实测踩中）。
                                        if os.environ.get("MAS_FINISH"):
                                            _finN = getattr(self, "_fin_n", 0)
                                            _finS = getattr(self, "_fin_seen5", 0)
                                            _finSkip = getattr(self, "_fin_skip5", 0)
                                            file.write("FINSTAT fin=%d seen=%d skip=%d army_thr=%s\n"
                                                       % (_finN, _finS, _finSkip,
                                                          os.environ.get("MAS_FINISH_ARMY", "8")))

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

                                        # #77：攻击点诊断汇总
                                        if os.environ.get("MAS_ATK_DUMP") and self._atk:
                                            _na = len(self._atk)
                                            _xs = [a[0] for a in self._atk]
                                            _ys = [a[1] for a in self._atk]
                                            _ds = sorted(a[2] for a in self._atk if a[2] >= 0)
                                            # #77b：选中单位构成
                                            _sels = [a[3] for a in self._atk]
                                            _combs = [a[4] for a in self._atk]
                                            _zero = sum(1 for c in _combs if c == 0) / float(_na)
                                            _selinfo = ("sel_mean=%.1f sel_max=%d combat_mean=%.1f "
                                                        "combat_max=%d frac_combat0=%.2f"
                                                        % (sum(_sels) / float(_na), max(_sels),
                                                           sum(_combs) / float(_na), max(_combs), _zero))
                                            # #90：兜底触发率（只在 MAS_SEL_GUARD=1 时有意义）
                                            if getattr(self, "_guard_seen", 0):
                                                _selinfo += (" guard_fired=%d/%d"
                                                             % (getattr(self, "_guard_fired", 0),
                                                                getattr(self, "_guard_seen", 0)))
                                                if getattr(self, "_guard_clean_n", 0):
                                                    _selinfo += (" guard_clean=%d" % getattr(self, "_guard_clean_n", 0))
                                            if getattr(self, "_tb_seen", 0):
                                                _selinfo += (" tb_seen=%d tb_hit=%d"
                                                             % (getattr(self, "_tb_seen", 0),
                                                                getattr(self, "_tb_n", 0)))
                                            # #116：出击闸门的执行计数（hold=因兵不够而按住，go=放行）
                                            if getattr(self, "_army_hold", 0) or getattr(self, "_army_go", 0):
                                                _selinfo += (" army_hold=%d army_go=%d"
                                                             % (getattr(self, "_army_hold", 0),
                                                                getattr(self, "_army_go", 0)))
                                            # #118：回防触发次数
                                            if getattr(self, "_defend_n", 0):
                                                _selinfo += (" defend=%d" % getattr(self, "_defend_n", 0))
                                            # #147：收尾规则的接管次数（MAS_FINISH）
                                            if getattr(self, "_fin_n", 0):
                                                _selinfo += (" fin=%d" % getattr(self, "_fin_n", 0))
                                            if _ds:
                                                _mean = sum(_ds) / len(_ds)
                                                _p50 = _ds[len(_ds) // 2]
                                                _lt = sum(1 for d in _ds if d < 10) / float(len(_ds))
                                                file.write("ATKDUMP n=%d with_enemy=%d mean_d=%.1f p50_d=%.1f "
                                                           "frac_d_lt10=%.2f %s x=[%.1f,%.1f] y=[%.1f,%.1f]\n"
                                                           % (_na, len(_ds), _mean, _p50, _lt, _selinfo,
                                                              min(_xs), max(_xs), min(_ys), max(_ys)))
                                            else:
                                                file.write("ATKDUMP n=%d with_enemy=0 %s x=[%.1f,%.1f] y=[%.1f,%.1f]\n"
                                                           % (_na, _selinfo, min(_xs), max(_xs), min(_ys), max(_ys)))
                                            self._atk = []

                                # #66：results 只统计 native；custom 另计。
                                results[native_outcome + 1] += 1
                                results_custom[custom_outcome + 1] += 1

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
                                        raise EvalEarlyStop

                            # use max_frames to end the loop
                            # whether to stop the run
                            if self.max_frames and total_frames >= self.max_frames:
                                print("Beyond the max_frames, return!")
                                raise EvalEarlyStop

                            # use max_frames_per_episode to end the episode
                            if self.max_frames_per_episode and episode_frames >= self.max_frames_per_episode:
                                print("Beyond the max_frames_per_episode, break!")
                                break

                        #self.coordinator.send_outcome(self.player, self.opponent, outcome)

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")
                            raise EvalEarlyStop

                        # #66：每局独立种子 ⇒ 一局一个 env。放在 max_episodes 检查之后，
                        # 否则会多打一局（先 break 就跳过了上面的收尾判断）。
                        if SEED_PER_EPISODE:
                            break

        except EvalEarlyStop as e:
            # 正常收尾（打满 max_episodes / max_frames / learner stopped）：
            # 不打 Traceback，只在日志里明确写"正常结束"。
            print("[eval] 正常收尾: %s（非故障）" % (e.__class__.__name__), flush=True)
        except Exception as e:
            # 真异常才打栈，避免再出现"日志有 Traceback 但结果是好的"的误判。
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            # 第 76 轮（GLM，#21）：这两行原来在 `if debug` 里 —— 评测模式下不打印，
            # 编排器就汇总不到 results（第一版验证局就是在这里扑空的）。
            print("results: ", results, flush=True)
            print("results_custom: ", results_custom, flush=True)   # #66
            win_rate = results[2] / (1e-9 + sum(results))
            win_rate_custom = results_custom[2] / (1e-9 + sum(results_custom))
            print("win rate: ", win_rate, flush=True)
            print("win rate custom: ", win_rate_custom, flush=True)

            total_time = time() - training_start_time

            if SAVE_STATISTIC: 
                with open(OUTPUT_FILE, 'a') as file:
                    statistic = 'Avg: [{}/{}] | win_rate: {:.1f} | win_rate_custom: {:.1f} | food_used: {:.1f} | army_count: {:.1f} | std(army_count): {:.1f} | collected_points: {:.1f} | used_points: {:.1f} | killed_points: {:.1f} | steps: {:.3f} | Total time: {:.3f}s \n'.format(
                        total_episodes, MAX_EPISODES, win_rate, win_rate_custom, np.mean(food_used_list), np.mean(army_count_list), np.std(army_count_list), np.mean(collected_points_list),
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

        # ---- 第 84 轮（DS，#75）：对手种族必须可配 ----
        # 实测（22:22，replay_info 逐局扫描）：60 局训练演示全部是
        # **Protoss(P1, 胜) vs Zerg(P2)**，而这里写死 `Race.terran`。
        # 也就是说：模型学的是"对虫族的开局与打法"，却被放在"对人类"里评测。
        # 这是评测分布与训练分布不一致，属于测量口径问题，不是策略问题。
        # 默认仍 terran（保持历史结果可比），用 EVAL_BOT_RACE=zerg 做对照。
        _RACE_MAP = {"terran": Race.terran, "zerg": Race.zerg,
                     "protoss": Race.protoss, "random": Race.random}
        _bot_race_name = _os.environ.get("EVAL_BOT_RACE", "terran").strip().lower()
        _bot_race = _RACE_MAP.get(_bot_race_name, Race.terran)
        print('[eval] bot race = %s (EVAL_BOT_RACE=%s)' % (_bot_race_name, _bot_race_name), flush=True)

        sc2_computer = Bot([_bot_race],
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
