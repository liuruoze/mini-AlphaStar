#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by fighting against built-in AI (computer), using no replays"

# import objgraph

from time import time, sleep, strftime, localtime
import gc
import os

# FIX(T3a): 多进程 RL 必设。nproc=128 时每个进程的 BLAS/OMP 默认开 128 个线程，
# 8 个 actor 就是 1024 个线程抢 128 核，实测把 preprocess_state_all 从 10 ms 拖到 200 ms（20 倍），
# 吞吐从 85 step/s 掉到 14 step/s。必须在 import numpy / torch 之前设置才有效。
for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

import random
import traceback
import threading
import datetime
# import multiprocessing as mp

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent, Race, Bot, Difficulty, BotBuild
from pysc2.lib import actions as sc2_actions
from pysc2.lib import units as sc2_units

from alphastarmini.core.rl.rl_utils import Trajectory, get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl.actor_inference import ActorInferenceClient   # 第 18 轮：推理路由
from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer   # 第 23 轮：中心批推理
from alphastarmini.core.rl.batch_server import SharedSnapshot   # 第 73 轮：全局不可变快照
from alphastarmini.core.rl import rl_utils as RU
from alphastarmini.core.rl import shared_adam as SA

from alphastarmini.lib import utils as L

from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import RL_Training_Hyper_Parameters as THP

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


# ---- 第 84 轮（DS，#50）：spawn 子进程的常量必须与父进程一致 ----
# spawn 模式会**重新 import 本模块**，而 `P.on_server` 来自 param.py（硬编码 False），
# 于是子进程里 SIMPLE_TEST = True -> MAX_EPISODES=1 / ACTOR_NUMS=1 / PARALLEL=1
# -> TRAIN_ITERS = 1。而 `Parameter_Server` **正是一个子进程**：
# 它会在收到第 1 局后满足 `while update_counter < train_iters` 而退出，
# 之后 Worker 还在跑，`q_winloss` 很快被填满（maxsize=TRAIN_ITERS*24=24）→ actor 的 put 阻塞 → 看起来像死锁。
# 环境变量会被子进程继承，所以以 MAS_ON_SERVER 为准（父进程 run_rl.py 已设置它）。
SIMPLE_TEST = not (P.on_server or os.environ.get("MAS_ON_SERVER", "0") == "1")
if SIMPLE_TEST:
    MAX_EPISODES = 1
    ACTOR_NUMS = 1
    PARALLEL = 1
    GAME_STEPS_PER_EPISODE = 18000  
    MAX_FRAMES = 18000 * 5
else:
    MAX_EPISODES = 4 * 4 * 500
    ACTOR_NUMS = 2  # 2
    PARALLEL = 8 + 7 * 1
    GAME_STEPS_PER_EPISODE = 18000
    MAX_FRAMES = 18000 * MAX_EPISODES

# T2/T4 灰度与扩容：允许环境变量覆盖并发度（不改变默认行为）
#   MAS_PARALLEL=4 MAS_ACTOR_NUMS=12  -> 4 个 Worker，各 12 个 actor
ACTOR_NUMS = int(os.environ.get("MAS_ACTOR_NUMS", ACTOR_NUMS))
PARALLEL = int(os.environ.get("MAS_PARALLEL", PARALLEL))

# ---- 第 84 轮（DS，#48a）：胜率聚合窗口可配 ----
# 原实现硬编码 100：`Parameter_Server` 只有凑满 100 局才算一次 win_rate 并写 tensorboard。
# 按 8 actor、约 5 分钟一局估算，30-45 分钟最多 40-70 局 —— **win_rate 曲线根本不会出现**。
# 用户本轮的要求是"各 loss 能正常采集，最好看到胜率"，所以必须能调小。
#   MAS_STATIC_NUM=10  -> 每 10 局出一个点
STATIC_NUM = int(os.environ.get("MAS_STATIC_NUM", "100"))
TRAIN_ITERS = MAX_EPISODES * ACTOR_NUMS * PARALLEL

USE_UPDATE_LOCK = False
DIFFICULTY = 1
ONLY_UPDATE_BASELINE = False
SAVE_LATEST = False
BASELINE_WEIGHT = 1
LR = 1e-6  # 1e-5
WEIGHT_DECAY = 1e-5

MAX_TIME_FOR_TRAINING = 60 * 60 * 24 * 7

# ---- 第 24 轮：smoke test 用的时长覆盖（不影响默认行为）----
# 为什么需要：`ActorVSComputer.__init__` 的默认参数是 `max_time_for_training=MAX_TIME_FOR_TRAINING`
# 这种写法，**默认参数在函数定义时求值** —— 所以测试脚本里 `M.MAX_TIME_FOR_TRAINING = 120`
# 是无效的（函数早已绑定旧值）。必须在模块加载时、函数定义之前覆盖，环境变量是唯一干净的办法。
#   MAS_MAX_TIME=120 MAS_MAX_EPISODES=1 MAS_ACTOR_NUMS=1 MAS_PARALLEL=1
MAX_TIME_FOR_TRAINING = int(os.environ.get("MAS_MAX_TIME", MAX_TIME_FOR_TRAINING))

# ---- 第 84 轮（DS，#52）：单局帧上限**不能**跟着训练时长缩小 ----
# 原实现：`ActorVSComputer.__init__(max_frames_per_episode=22.4 * MAX_TIME_FOR_TRAINING)`。
# 冒烟/短跑时 MAS_MAX_TIME=240 -> 上限 5376 帧，而一局正常要打满 ~21000 帧。
# 后果：每局都在 5376 帧被 `break`，而那个分支**不投递 outcome**
#   （只有 `is_final` 分支才 put 到 q_winloss），于是：
#   队列永远为空 -> Parameter_Server 等 timeout -> 整个训练看似"跑起来了"但一局都没记。
# 实测：4 分钟冒烟，v_steps 涨到 5568、optimizer_steps=46，但 native_outcome 计数 = 0。
# 现在默认 0（不限制，由 SC2 自己在 ~21000 帧结束），需要时用 MAS_MAX_FRAMES_PER_EP 覆盖。
MAX_FRAMES_PER_EPISODE = int(os.environ.get("MAS_MAX_FRAMES_PER_EP", "0"))
MAX_EPISODES = int(os.environ.get("MAS_MAX_EPISODES", MAX_EPISODES))
MAX_FRAMES = int(os.environ.get("MAS_MAX_FRAMES", MAX_FRAMES))
TRAIN_ITERS = MAX_EPISODES * ACTOR_NUMS * PARALLEL

USE_MIDDLE_REWARD = False
USE_DEFINED_REWARD_AS_REWARD = False
USE_RESULT_REWARD = True
REWARD_SCALE = 1e-3
WINRATE_SCALE = 2

USE_BUFFER = False
if USE_BUFFER:
    BUFFER_SIZE = 10 
    COUNT_OF_BATCHES = 1
    NUM_EPOCHS = 1
    USE_RANDOM_SAMPLE = True
else:
    BUFFER_SIZE = 2
    COUNT_OF_BATCHES = 1
    NUM_EPOCHS = 2
    USE_RANDOM_SAMPLE = False

STEP_MUL = 8
UPDATE_PARAMS_INTERVAL = 10

RESTORE = True
SAVE_STATISTIC = True
RANDOM_SEED = 1

# model path
MODEL_TYPE = "sl"
MODEL_PATH = "./model/"
OUTPUT_FILE = './outputs/rl_vs_inner_bot.txt'

VERSION = SCHP.game_version
MAP_NAME = SCHP.map_name

SAVE_REPLAY = False
IS_TRAINING = True

WIN_THRESHOLD = 4000
USE_OPPONENT_STATE = False
NO_REPLAY_LEARN = True
NEED_SAVE_RESULT = True

NOW = datetime.datetime.now()
now_str = NOW.strftime("%Y%m%d-%H%M%S")
now_model_str = NOW.strftime("%y-%m-%d_%H-%M-%S")
SUMMARY_PATH = "./log/" + now_str

model_save_type = "rl"
MODEL_SAVE_PATH = os.path.join("./model/", model_save_type + "_" + now_model_str)

# strftime("%y-%m-%d_%H-%M-%S", localtime()

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
if torch.backends.cudnn.is_available():
    print('cudnn available')
    print('cudnn version', torch.backends.cudnn.version())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# TODO: fix the bug ValueError: The game didn't advance to the expected game loop. Expected: 2512, got: 2507


class ActorVSComputer:
    """A single actor loop that generates trajectories by playing with built-in AI (computer).

    We don't use batched inference here, but it was used in practice.
    """

    def __init__(self, player, q_winloss, q_points, device, global_model, coordinator, 
                 teacher, idx, buffer_lock=None, results_lock=None, 
                 writer=None, max_time_for_training=MAX_TIME_FOR_TRAINING,
                 max_time_per_one_opponent=MAX_TIME_FOR_TRAINING,
                 max_frames_per_episode=MAX_FRAMES_PER_EPISODE, max_frames=MAX_FRAMES, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING,
                 replay_dir="./added_simple64_replays/",
                 update_params_interval=UPDATE_PARAMS_INTERVAL,
                 need_save_result=NEED_SAVE_RESULT,
                 batch_srv=None, batch_idx=None):
        self.player = player
        self.player.add_actor(self)
        self.idx = idx
        self.name = 'agent_' + str(self.idx)
        self.teacher = teacher

        self.q_winloss = q_winloss
        self.q_points = q_points

        self.global_model = global_model
        self.coordinator = coordinator

        # self.agent = self.player.agent
        self.agent = get_supervised_agent(player.race, path=MODEL_PATH, model_type=MODEL_TYPE, restore=RESTORE, device=device)

        # ---- 第 64 轮（Kimi-06 P0）：acting 必须在 eval 模式下 ----
        # 原实现**从来没调用过 eval()**：nn.Module 默认 training=True，
        # 而模型里有 4 层 BatchNorm2d（spatial_encoder）+ 1 层 Dropout（entity_encoder），
        # 所以 local actor 采样时：
        #   ① BN 用**当前单样本的统计量**归一化，而不是 running stats；
        #   ② BN 的 running_mean/var 被**持续改写** —— 这条最致命：
        #      行为策略不再只是权重的函数，而是"权重 + 被采样过程自己改写的 BN 统计量"；
        #   ③ dropout 生效，动作带一层与策略无关的噪声。
        # 后果：`Trajectory.behavior_logits` 记的不是策略分布，rho 失去意义。
        # 注意执行顺序：必须在**被 global_model 替换之前**调用。
        # eval() 是 in-place 的，若放到 batch 分支之后调用，就会改掉 learner 的共享模型。
        try:
            self.agent.agent_nn.model.eval()
        except Exception as e:
            print("[ActorVSComputer %d] eval() 失败: %r" % (self.idx, e))
        if self.idx == 0:
            audit_model_mode("actor_0", self.agent.agent_nn.model)

        # ---- 第 22 轮：batch 模式下复用「共享内存的模型」，不再各持一份 ----
        # 为什么必须做：`get_supervised_agent` 会**新建**一个模型并加载权重，
        # 于是每个 Worker/actor 各占一份显存。第 16 轮已量过模型是 2,664,028 参数
        # （且 32.9% 不训练），N 个 actor 就是 N 份。
        # batch 模式下 actor 侧**不做 forward**，模型只是为了 preprocess_state_all 等
        # 不需要权重的工具方法而存在，所以直接指向 `global_model`（= test() 里
        # `model_learner.share_memory()` 的那份）即可 —— 权重天然与 learner 同步，
        # 连 `set_weights` 都不需要了。
        if batch_srv is not None and os.environ.get("MAS_BATCH_ACTING", "0") == "1":
            if global_model is not None:
                self.agent.agent_nn.model = global_model
                # ⚠️ **不要**在这里调 `.to(device)`：`nn.Module.to()` 是 **in-place** 的，
                #    而 global_model 是 learner 正在用的共享内存模型 ——
                #    若 actor 的 device 与 learner 不同，这一步会把 learner 的模型搬走。
                #    actor 侧只用 `preprocess_state_all`（纯 CPU 张量运算，不需要权重），
                #    所以模型留在原卡上完全没问题。
            print("[ActorVSComputer %d] batch 模式：复用共享模型（只省显存，不 forward、不写）"
                  % self.idx)
        # self.agent = get_supervised_agent(player.race, path=MODEL_PATH, model_type=MODEL_TYPE, restore=RESTORE, device=device)
        # if ON_GPU:
        #     self.agent.agent_nn.to(device)

        self.max_time_for_training = max_time_for_training
        self.max_time_per_one_opponent = max_time_per_one_opponent
        self.max_frames_per_episode = max_frames_per_episode
        self.max_frames = max_frames
        self.max_episodes = max_episodes
        self.is_training = is_training

        self.thread = threading.Thread(target=self.run, args=())

        self.thread.daemon = True                            # Daemonize thread
        self.buffer_lock = buffer_lock
        self.results_lock = results_lock

        self.is_running = True
        self.is_start = False

        # 第 33 轮：轻量步数统计。
        # 为什么需要：local 模式不经过 batch server，拿不到 `srv.async_summary()` 的
        # steps/rounds，于是「local vs batch」的吞吐对照**没法用同一个口径比**。
        # 这个计数器对两种模式都生效，退出时打印一行，便于直接对比。
        self.step_count = 0

        # ---- 第 18 轮：推理路由 ----
        # `self.infer` 在 run() 里、env 建好（拿到 action_spec）之后才创建。
        # `batch_srv=None`（默认）时它是 local 模式，`step()` 直接转调
        # `self.agent.step_from_state()`，**与改造前逐位等价**（见 _test_r15_client.py [D]）。
        self._batch_srv = batch_srv
        self._batch_idx = batch_idx
        self.infer = None

        self.replay_dir = replay_dir
        self.writer = writer
        self.update_params_interval = update_params_interval
        self.need_save_result = need_save_result

    def start(self):
        self.is_start = True
        self.thread.start()

    # background
    def run(self):
        try:
            with torch.no_grad():
                self.is_running = True

                """A run loop to have agents and an environment interact."""
                total_frames = 0
                total_episodes = 0

                # the total numberv_for_all_episodes as [loss, draw, win]
                # results = [0, 0, 0]

                # statistic list
                # food_used_list, army_count_list, collected_points_list, used_points_list, killed_points_list, steps_list = [], [], [], [], [], []

                training_start_time = time()
                print("start_time before training:", strftime("%Y-%m-%d %H:%M:%S", localtime(training_start_time)))

                # judge the trajectory whether contains final
                is_final_trajectory = False
                is_win_trajectory = False

                player_bo = None

                # use max_episodes to end the loop
                while time() - training_start_time < self.max_time_for_training:
                    agents = [self.agent]

                    with self.create_env_one_player(self.player) as env:

                        # set the obs and action spec
                        observation_spec = env.observation_spec()
                        action_spec = env.action_spec()

                        for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                            agent.setup(obs_spec, act_spec)

                        self.teacher.setup(self.agent.obs_spec, self.agent.action_spec)

                        # ---- 第 18 轮：建立推理路由 ----
                        # 必须放在这里：client 需要 action_spec，而它上面几行才由 agent.setup() 设好。
                        # 默认（batch_srv=None 或 MAS_BATCH_ACTING!=1）走 local，
                        # 此时 `step()` 就是 `self.agent.step_from_state()` 的薄封装，
                        # **行为与改造前逐位一致**，所以这个改动对现有训练是零风险的。
                        self.infer = ActorInferenceClient(
                            agent=self.agent,
                            action_spec=self.agent.action_spec,
                            srv=(self._batch_srv
                                 if os.environ.get("MAS_BATCH_ACTING", "0") == "1" else None),
                            idx=(self._batch_idx if self._batch_idx is not None else self.idx))

                        print('player:', self.player) if debug else None
                        print('opponent:', "Computer bot") if debug else None

                        trajectory = []

                        update_params_timer = time()

                        opponent_start_time = time()  # in seconds.
                        print("opponent_start_time before reset:", strftime("%Y-%m-%d %H:%M:%S", localtime(opponent_start_time)))

                        # one opponent match (may include several games) defaultly lasts for no more than 2 hour
                        while time() - opponent_start_time < self.max_time_per_one_opponent:

                            # Note: the pysc2 environment don't return z
                            # AlphaStar: home_observation, away_observation, is_final, z = env.reset()
                            total_episodes += 1
                            print(self.name, "total_episodes:", total_episodes)

                            timesteps = env.reset()
                            for a in agents:
                                a.reset()

                            [home_obs] = timesteps
                            is_final = home_obs.last()

                            player_memory = self.agent.initial_state()
                            # 第 18 轮：batch 模式下 hidden 存在 server 里，actor 拿不到也清不了，
                            # 必须显式通知 server 清零 —— 否则新 episode 第一步会读到**上一局**
                            # 的 LSTM 状态，不崩不报错，但策略在错的状态上决策。
                            if self.infer is not None and self.infer.is_batch:
                                self.infer.reset_hidden()
                            # teacher_memory = self.teacher.initial_state()

                            episode_frames = 0

                            # initial build order
                            if player_bo is not None:
                                del player_bo
                            player_bo = []

                            # default outcome is 0 (means draw)
                            outcome = 0

                            # initial last list
                            last_list = [0, 0, 0]

                            # points for defined reward
                            points, last_points = 0, None

                            # in one episode (game)
                            start_episode_time = time()  # in seconds.
                            print("start_episode_time before is_final:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_episode_time)))

                            # growth = objgraph.growth(limit=5)
                            # if len(growth):
                            #     print(self.name, os.getpid(), "after one episode", growth)

                            while not is_final:

                                t = time()

                                # every 10s, the actor get the params from the learner
                                # if time() - update_params_timer > self.update_params_interval:
                                #     print("agent_{:d} update params".format(self.idx)) if debug else None
                                #     self.agent.set_weights(self.player.agent.get_weights())
                                #     self.agent.agent_nn.model.load_state_dict(self.global_model.state_dict())
                                #     update_params_timer = time()

                                # every 10s, the actor get the params from the learner
                                if time() - update_params_timer > self.update_params_interval:
                                    # 第 23 轮：batch 模式下**必须**跳过 set_weights。
                                    # 原因：`self.agent.agent_nn.model` 已指向共享的 global_model
                                    # （见 __init__），而 set_weights 会**写**它 ——
                                    # 与 learner 的 optimizer.step() 抢写，产生权重错乱。
                                    # batch 模式下 actor 根本不做 forward，也不需要本地权重。
                                    if self.infer is None or not self.infer.is_batch:
                                        print("agent_{:d} update params".format(self.idx)) if debug else None
                                        self.agent.set_weights(self.player.agent.get_weights())
                                    update_params_timer = time()

                                state = self.agent.agent_nn.preprocess_state_all(home_obs.observation, 
                                                                                 build_order=player_bo, 
                                                                                 last_list=last_list)
                                baseline_state = self.agent.agent_nn.get_baseline_state_from_multi_source_state(home_obs.observation, state)

                                with torch.no_grad():
                                    # 第 18 轮：改走推理路由。local 模式下 `self.infer.step`
                                    # 就是 `self.agent.step_from_state` 的薄封装（逐位等价，
                                    # 见 _test_r15_client.py），所以这条改动对现有训练零风险。
                                    player_function_call, player_action, player_logits, \
                                        player_new_memory, player_select_units_num, entity_num = self.infer.step(state, 
                                                                                                                    player_memory, 
                                                                                                                    home_obs.observation)

                                print("player_function_call:", player_function_call) if debug else None
                                print("player_action.delay:", player_action.delay) if debug else None

                                print("entity_num:", entity_num) if debug else None
                                print("player_select_units_num:", player_select_units_num) if debug else None
                                print("player_action:", player_action) if debug else None

                                if False:
                                    show_sth(home_obs, player_action)

                                expected_delay = player_action.delay.item()
                                # ---- 第 73 轮（GLM，Kimi-13）：delay 设计决定 ----
                                # 原来这里算 `step_mul = max(1, expected_delay)`，但下面
                                # env.step 传的是常量 STEP_MUL —— 算了不用，日志却宣称
                                # "delay 决定决策频率"。这是误导。
                                # 设计决定（冻结）：固定决策频率 STEP_MUL=8；
                                # delay 头照常输出（其主 loss 权重为 0，见 DS 记录），
                                # 但**不**驱动 env.step。若将来启用动态频率，必须：
                                #   ① 把 step_mul 接进 env.step 并做有界裁剪；
                                #   ② 给 delay 监督非零权重；
                                #   ③ 单独做消融。三者缺一不可，不许只改其一。
                                # step_mul_local = max(1, expected_delay)   # 冻结，见上
                                print("step_mul(冻结为常量):", STEP_MUL) if debug else None

                                with torch.no_grad():
                                    teacher_logits = self.teacher.step_based_on_actions(state, player_memory, player_action, player_select_units_num)
                                    print("teacher_logits:", teacher_logits) if debug else None

                                env_actions = [player_function_call]

                                player_action_spec = action_spec[0]
                                action_masks = RU.get_mask(player_action, player_action_spec)
                                unit_type_entity_mask = RU.get_unit_type_mask(player_action, home_obs.observation)
                                print('unit_type_entity_mask', unit_type_entity_mask) if debug else None

                                z = None

                                timesteps = env.step(env_actions, step_mul=STEP_MUL)  # STEP_MUL step_mul
                                [home_next_obs] = timesteps
                                total_frames += 1 * STEP_MUL
                                episode_frames += 1 * STEP_MUL
                                del env_actions, timesteps

                                # fix the action delay
                                # player_action.delay = torch.tensor([[STEP_MUL]], dtype=player_action.delay.dtype,
                                #                                    device=player_action.delay.device)

                                reward = float(home_next_obs.reward)
                                print("reward: ", reward) if 0 else None

                                is_final = home_next_obs.last()

                                # calculate the build order
                                player_bo = L.calculate_build_order(player_bo, home_obs.observation, home_next_obs.observation)
                                print("player build order:", player_bo) if debug else None

                                # calculate the unit counts of bag
                                player_ucb = None  # L.calculate_unit_counts_bow(home_obs.observation).reshape(-1).numpy().tolist()

                                game_loop = home_obs.observation.game_loop[0]
                                print("game_loop", game_loop) if debug else None

                                points = get_points(home_next_obs)

                                if USE_MIDDLE_REWARD:
                                    if last_points is not None:
                                        reward = points - last_points
                                    else:
                                        reward = 0.
                                last_points = points

                                if is_final:
                                    game_outcome = home_next_obs.reward

                                    o = home_next_obs.observation
                                    killed_minerals = np.sum(o['score_by_category']['killed_minerals'])
                                    killed_vespene = np.sum(o['score_by_category']['killed_vespene'])

                                    killed_points = float(killed_minerals + killed_vespene)

                                    del killed_minerals, killed_vespene, o

                                    if game_outcome == 1:
                                        outcome = 1
                                    elif game_outcome == 0:
                                        # outcome = 0
                                        if killed_points > WIN_THRESHOLD:
                                            outcome = 1
                                        else:
                                            outcome = 0
                                        #     # if killed_points > 1000 and killed_points <= WIN_THRESHOLD:
                                        #     #     outcome = 0
                                        #     # else:
                                        #     #     outcome = -1
                                        #     outcome = 0
                                        #     # print("agent_{:d} get outcome".format(self.idx), outcome) if 1 else None
                                    else:
                                        outcome = -1

                                    if not USE_DEFINED_REWARD_AS_REWARD:
                                        reward = float(outcome)
                                        if outcome == 0:
                                            reward = killed_points / float(WIN_THRESHOLD)

                                    print("agent_{:d} get final reward".format(self.idx), reward) if 1 else None
                                    # ---- 第 73 轮（GLM，Kimi-13）：native 与 custom 分开记 ----
                                    # `game_outcome` 是 SC2 原生结果（1胜/0平/-1负）；
                                    # `outcome` 是自定义口径（平局按 killed_points 重分类）。
                                    # **对外报告的"胜率"默认只用 native**，custom 另名，
                                    # 两者混用会把"击杀多的平局"算成胜利，虚高涨势。
                                    native_outcome = int(game_outcome)
                                    print("agent_{:d} get outcome".format(self.idx), outcome,
                                          flush=True) if 1 else None          # custom 口径（旧 grep 兼容）
                                    print("agent_{:d} get native_outcome".format(self.idx),
                                          native_outcome, flush=True) if 1 else None  # SC2 原生口径

                                    final_outcome = outcome
                                    final_points = points  # killed_points / float(WIN_THRESHOLD)

                                    self.q_winloss.put(final_outcome)
                                    self.q_points.put(final_points)

                                    reward = final_outcome

                                    is_final_trajectory = True
                                    if outcome == 1:
                                        is_win_trajectory = True

                                    gc.collect() 
                                else:
                                    pass

                                # note, original AlphaStar pseudo-code has some mistakes, we modified 
                                # them here

                                del points

                                if 0:
                                    state.to('cpu')
                                    baseline_state = [l.to('cpu') for l in baseline_state]
                                    player_memory = [l.to('cpu') for l in player_memory]
                                    player_logits.to('cpu')
                                    teacher_logits.to('cpu')
                                    player_action.to('cpu')
                                    player_select_units_num = player_select_units_num.to('cpu')
                                    entity_num = entity_num.to('cpu')

                                print("agent_{:d} get reward".format(self.idx), reward) if 0 else None
                                print("player_action.delay:", player_action.delay) if debug else None

                                traj_step = Trajectory(
                                    state=state,
                                    baseline_state=baseline_state,
                                    baseline_state_op=None,  # when fighting with computer, we don't use opponent state
                                    # 第 18 轮：Trajectory.memory 要的是「本步 forward **之前**」的
                                    # hidden。local 模式下 `last_trajectory_memory` 就是传进去的
                                    # `player_memory`（R15 [D] 实测 0.000e+00），两者等价；
                                    # batch 模式下 actor 不再持有 hidden，只能由 server 回传。
                                    memory=(self.infer.last_trajectory_memory
                                            if self.infer is not None else player_memory),
                                    z=z,
                                    masks=action_masks,
                                    unit_type_entity_mask=unit_type_entity_mask,
                                    action=player_action,
                                    behavior_logits=player_logits,
                                    teacher_logits=teacher_logits,      
                                    is_final=is_final,                                          
                                    reward=reward,
                                    player_select_units_num=player_select_units_num,
                                    entity_num=entity_num,
                                    build_order=player_bo,
                                    z_build_order=None,  # we change it to the sampled build order
                                    unit_counts=None,     # player_ucb,  # player_ucb,
                                    z_unit_counts=None,  # player_ucb,  # we change it to the sampled unit counts
                                    game_loop=game_loop,
                                    last_list=last_list,
                                )

                                del state, baseline_state, player_memory, z
                                del action_masks, unit_type_entity_mask, player_logits, teacher_logits
                                del player_select_units_num, entity_num
                                del reward, game_loop
                                if last_list is not None:
                                    del last_list

                                if self.is_training:
                                    print('is_final_trajectory', is_final_trajectory) if debug else None
                                    trajectory.append(traj_step)
                                del traj_step
                                self.step_count += 1

                                # player_memory = tuple(h.detach().clone() for h in player_new_memory)
                                # 第 24 轮修正（重要）：这里**必须**给 `player_memory` 重新绑定，
                                # 无论哪种模式。原因：上面第 563 行有
                                #     del state, baseline_state, player_memory, z
                                # 把它**解绑**了，而下一帧 `self.infer.step(state, player_memory, ...)`
                                # 和第 419 行的 `teacher.step_based_on_actions(..., player_memory, ...)`
                                # 仍然引用它 —— 不重新绑定就是
                                #     UnboundLocalError: local variable 'player_memory' referenced before assignment
                                # （第 18 轮我在这里加了 `if not is_batch:` 守卫，batch 模式下跳过了赋值，
                                #   于是第一次 smoke test 就撞上这个错误。守卫"少做一件事"和
                                #   "多做一件事"一样会破坏既有假设。）
                                #
                                # batch 模式下赋什么：用 `last_trajectory_memory`（server 回传的
                                # **本步输入** hidden）。这样：
                                #   - 语义一致（与填进 Trajectory.memory 的是同一个量）；
                                #   - teacher 的 teacher-forcing 也用同一份 hidden，不会错位；
                                #   - `step()` 在 batch 模式本来就会忽略这个入参，所以不会双重推进。
                                if self.infer is None or not self.infer.is_batch:
                                    player_memory = player_new_memory
                                else:
                                    player_memory = self.infer.last_trajectory_memory
                                del home_obs
                                home_obs = home_next_obs
                                del home_next_obs
                                last_delay = expected_delay
                                last_action_type = player_action.action_type.item()
                                last_repeat_queued = player_action.queue.item()
                                last_list = [last_delay, last_action_type, last_repeat_queued]

                                del last_delay, last_action_type, last_repeat_queued
                                del player_action, player_new_memory

                                if self.is_training and len(trajectory) >= AHP.sequence_length:                    
                                    trajectories = RU.stack_namedtuple(trajectory)
                                    del trajectory

                                    if self.player.learner is not None:
                                        if self.player.learner.is_running:
                                            print("Learner send_trajectory!") if debug else None
                                            # with self.buffer_lock:

                                            self.player.learner.send_trajectory(trajectories)

                                            # ---- 第 58 轮：真实轨迹落盘（只用于验收，默认关闭）----
                                            # 为什么需要：T8/T9 的 rho 都是在**手工拼的 tensor** 或
                                            # 假 reward 上算的。要证明「batch 路径产出的轨迹能直接喂
                                            # 训练」，必须拿**真实 env 观测 + 真实权重**跑出来的轨迹，
                                            # 再离线重算 rho。
                                            # 触发：MAS_DUMP_TRAJ=<文件路径>；只落**第一条**（幂等）。
                                            # 计时开销：每个 16 步做一次 os.path.exists，可忽略。
                                            #
                                            # ⚠️ 必须把**当时的权重**一起落盘：rho 的分母是
                                            # behavior_logits，分子要拿同一份权重重算才有意义。
                                            # 只存轨迹、事后用 SL checkpoint 重算，rho 天然不等于 1，
                                            # 那样测出来的是「训练进程走了多远」，不是「链路对不对」。
                                            _dump = os.environ.get("MAS_DUMP_TRAJ", "")
                                            # ⚠️ 只让 actor 0 攒：8 条轨迹的 state 是实体级张量
                                            # （128×1856×4B×8 步 ≈ 60 MB/条），30 个 actor 一起攒
                                            # 就是 ~14 GB，会直接把显存/内存打爆。
                                            if _dump and self.idx == 0 and not os.path.exists(_dump):
                                                try:
                                                    # ---- 第 66 轮：落**一整批**轨迹，不是一条 ----
                                                    # 为什么：`rl_loss` 里的 `AHP.sequence_length × AHP.batch_size`
                                                    # 是硬编码的（= 8 × 8），所以离线复算六头 rho 必须
                                                    # 凑够 `batch_size` 条轨迹，否则形状永远对不上
                                                    # （第 64 轮四次失败的根因，见 DS11 §64.13）。
                                                    _need = int(AHP.batch_size)
                                                    if not hasattr(self, "_dump_buf"):
                                                        self._dump_buf = []
                                                    if len(self._dump_buf) < _need:
                                                        self._dump_buf.append(trajectories)
                                                    if len(self._dump_buf) == _need:
                                                        # 第 73 轮（Kimi-05 第 3 条）：**优先落共享快照**。
                                                        # 原来落的是 `global_model` 的当前权重，而这条轨迹是 server 当时的快照产生的 ——
                                                        # 两者可能差很多个 step，拿它去离线重算 rho 等于换了一份权重。
                                                        _snap = (getattr(self._batch_srv, "snapshot", None)
                                                                 if self._batch_srv is not None else None)
                                                        _snap_got = _snap.read() if _snap is not None else None
                                                        if _snap_got is not None:
                                                            _pv_snap, _hash, _sd = _snap_got
                                                            _src_name = "shared_snapshot"
                                                        else:
                                                            _sd_src = (self.global_model
                                                                       if self.global_model is not None
                                                                       else self.agent.agent_nn.model)
                                                            _sd = {k: v.detach().cpu().clone()
                                                                   for k, v in _sd_src.state_dict().items()}
                                                            _hash = ""
                                                            _src_name = "global_model"
                                                        # 30 个 actor 会同时看到"文件不存在"，
                                                        # 直接 save 到同一路径会互相截断 -> 先写临时文件再原子替换。
                                                        _tmp = "%s.tmp%d" % (_dump, os.getpid())
                                                        _pv = (int(self._batch_srv.read_policy_version())
                                                               if self._batch_srv is not None else -1)
                                                        torch.save({"trajs": self._dump_buf,
                                                                    "traj": self._dump_buf[0],
                                                                    "state_dict": _sd,
                                                                    "policy_version": _pv,
                                                                    "actor": self.idx,
                                                                    "batch_size": _need},
                                                                   _tmp)
                                                        os.replace(_tmp, _dump)
                                                        self._dump_buf = []   # 落盘后立刻释放
                                                        print("[%s] 已落盘 %d 条真实轨迹 -> %s（mode=%s, %d 个参数张量, pv=%d, src=%s, hash=%s）"
                                                              % (self.name, _need, _dump,
                                                                 "batch" if (self.infer is not None
                                                                             and self.infer.is_batch) else "local",
                                                                 len(_sd), _pv, _src_name,
                                                                 _hash or "n/a"),
                                                              flush=True)
                                                except Exception as _e:
                                                    print("[%s] 轨迹落盘失败: %r" % (self.name, _e), flush=True)

                                            # if 0 and is_final_trajectory:
                                            #     self.player.learner.send_final_trajectory(trajectories)

                                            # if 0 and is_win_trajectory:
                                            #     self.player.learner.send_win_trajectory(trajectories)

                                        else:
                                            print("Learner stops!")

                                            print("Actor also stops!")
                                            return

                                    trajectory = []
                                    del trajectories

                                    is_final_trajectory = False
                                    is_win_trajectory = False

                                # use max_frames to end the loop
                                # whether to stop the run
                                if self.max_frames and total_frames >= self.max_frames:
                                    print("Beyond the max_frames, return!")
                                    raise Exception

                                # use max_frames_per_episode to end the episode
                                if self.max_frames_per_episode and episode_frames >= self.max_frames_per_episode:
                                    print("Beyond the max_frames_per_episode, break!")
                                    break

                            # if False:
                            #     with self.results_lock:
                            #         self.coordinator.only_send_outcome(self.player, outcome)

                            # use max_frames_per_episode to end the episode
                            if self.max_episodes and total_episodes >= self.max_episodes:
                                print("Beyond the max_episodes, return!")
                                raise Exception

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e) if debug else None
            print(traceback.format_exc()) if 1 else None
            pass

        finally:
            # print("results: ", results) if debug else None
            # print("win rate: ", results[2] / (1e-9 + sum(results))) if debug else None

            total_time = time() - training_start_time

            # 第 33 轮：local / batch 通用的吞吐口径。
            # local 模式不经过 batch server，拿不到 `srv.async_summary()`，
            # 所以这里自己算一份，保证两种模式可以用同一个数字对比。
            try:
                _st = getattr(self, "step_count", 0)
                if total_time > 0:
                    print("[%s] steps=%d | 总耗时 %.1f s | 吞吐 %.2f step/s（mode=%s）"
                          % (self.name, _st, total_time, _st / total_time,
                             "batch" if (self.infer is not None and self.infer.is_batch) else "local"),
                          flush=True)
            except Exception:
                pass
            # print('agent_', self.idx, "total_time: ", total_time / 60.0, "min") if debug else None

            # if debug and SAVE_STATISTIC: 
            #     with self.results_lock:
            #         self.coordinator.send_eval_results(self.player, DIFFICULTY, food_used_list, army_count_list, 
            #                                            collected_points_list, used_points_list, 
            #                                            killed_points_list, steps_list, total_time)

            self.is_running = False

            # 第 50 轮：actor 退出时向 server 注销槽位。
            # 作用：让 `AsyncBatchInferenceServer.run()` 的 `alive == 0` 条件**立刻**成立，
            # 而不是靠 `idle_timeout=120 s` 兜底 —— 长跑收尾从 2 min 降到 <1 s。
            # 幂等：重复调用只是再写一次同一个 Array 元素。
            try:
                if self.infer is not None:
                    self.infer.close()
            except Exception:
                pass

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

        # class BotBuild(enum.IntEnum):
        #   """Bot build strategies."""
        #   random = sc_pb.RandomBuild
        #   rush = sc_pb.Rush
        #   timing = sc_pb.Timing
        #   power = sc_pb.Power
        #   macro = sc_pb.Macro
        #   air = sc_pb.Air

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


def get_points(obs):
    o = obs.observation
    p = o['player']

    food_used = p['food_used']
    army_count = p['army_count']

    collected_minerals = np.sum(o['score_cumulative']['collected_minerals'])
    collected_vespene = np.sum(o['score_cumulative']['collected_vespene'])

    collected_points = collected_minerals + collected_vespene

    used_minerals = np.sum(o['score_by_category']['used_minerals'])
    used_vespene = np.sum(o['score_by_category']['used_vespene'])

    used_points = used_minerals + used_vespene

    killed_minerals = np.sum(o['score_by_category']['killed_minerals'])
    killed_vespene = np.sum(o['score_by_category']['killed_vespene'])

    killed_points = killed_minerals + killed_vespene

    # points = float(0.1 * collected_points + 0.3 * used_points + 0.6 * killed_points)
    # points = points * REWARD_SCALE

    points = min(float(killed_points) / float(WIN_THRESHOLD), 1.0)

    return points


def audit_model_mode(tag, model):
    """第 64 轮（Kimi-06）：打印模型**实际**处于什么模式，而不是假设它已经是 eval。

    为什么需要：R15 我把"两次结果不同"直接归因为 dropout，但从没做过消融；
    而模型里真正会受模式影响的是 **4 层 BatchNorm2d**（spatial_encoder）——
    train 模式下 BN 用当前 batch 的统计量并改写 running_mean/var，
    batch=1 时等于"用单个样本自己归一化自己"。这比 dropout 更隐蔽。
    """
    try:
        bn = [m for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        do = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        # 第一个 BN 的 running 统计量：用来判断它有没有被采样过程改写
        rm = float(bn[0].running_mean.abs().mean()) if bn and bn[0].running_mean is not None else -1
        # 只汇总"非零 p"的层数，别把 85 个模块的 p 全打出来（日志会被刷屏）
        _nz = [m for m in do if float(m.p) > 0]
        ds = ("%d 个非零（p=%s）" % (len(_nz), "/".join(sorted({("%.2f" % m.p) for m in _nz})))
              if _nz else "全 0")
        print("[mode] %-16s training=%-5s | BatchNorm %d 个（首个 |running_mean|=%.3g）| Dropout %d 个（p=%s）"
              % (tag, model.training, len(bn), rm, len(do), ds), flush=True)
    except Exception as e:
        print("[mode] %s 审计失败: %r" % (tag, e), flush=True)


def pick_cuda_device(rank, use_cuda_device=True):
    """按 rank 选择设备（T0 修复 cuda 越界 bug，抽成函数以便单测）。

    原实现有两个越界分支：
      rank < 8   -> cuda:<rank>            rank 4-7 在 4 卡机器上不存在
      rank >= 8  -> cuda:(rank-8)%7+1      rank 8-14 映射到 cuda:1-7，同样越界
    现改为按实际可见卡数取模；MAS_GPU_LIST 可指定白名单（实测 cuda:0 偏慢 20~45%）。
    """
    if not (use_cuda_device and torch.cuda.is_available()):
        return 'cpu'
    env = os.environ.get("MAS_GPU_LIST", "").strip()
    if env:
        gpus = [int(x) for x in env.replace(" ", "").split(",") if x != ""]
    else:
        gpus = list(range(torch.cuda.device_count()))
    if not gpus:
        return 'cpu'
    return "cuda:" + str(gpus[rank % len(gpus)])


def make_batch_state_shapes():
    """批推理 server 需要的共享内存形状（第 23 轮）。

    这些数值来自 `_probe_state_shape.py` 的实测（第 3 轮）：
      entity (128, 1856) / map (24, 64, 64) / 16 个 scalar 项。
    ⚠️ 不能从模型动态推导：`StateShapes` 必须在 **fork 之前** 用来分配共享内存，
       而那时还没有任何真实 obs。
    """
    from alphastarmini.core.rl.batch_server import StateShapes
    return StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])


def Batch_Server_Worker(srv, model_learner, device, ready_ev, idle_timeout=120.0,
                        sync_interval=8.0, pub_lock=None, pv=None, snapshot=None):
    """中心批推理 server 进程（第 23 轮）。

    职责：持有**一份**可 forward 的 agent，跑 `AsyncBatchInferenceServer.run()`。

    ## ⚠️ 为什么不能直接 `a.model = model_learner`（第 23 轮自我更正）

    第一版写的就是「直接指向父进程 share_memory 的那份模型」，理由是"权重天然同步、
    省一份显存"。**这是错的**，它破坏了两件事：

    1. **on-policy 假设**：learner 进程随时在 `optimizer.step()` 写共享模型，
       而 server 随时可能在做 forward —— 一次 forward 内部可能读到"半新半旧"的权重，
       采样分布与任何时刻的策略都对不上，VTrace 的 rho 也就失去意义。
       `batch_server` 原本设计了 `pending_weights`，**只在批边界应用**，正是为了避免这个。
    2. **`ActorVSComputer.set_weights` 会写它**：actor 里定期
       `self.agent.set_weights(...)` 会直接改写共享模型，与 learner 抢写。

    ## 正确做法

    - server 用**自己的模型副本**（1 份，不是每 Worker 一份）；
    - 起一个同步线程，定期把 `model_learner` 的权重塞进 `srv.pending_weights`，
      由 `_serve_batch` 在**批边界**应用 —— 保持原有的正确语义；
    - `ActorVSComputer` 在 batch 模式下**禁用** `set_weights`（见那边的注释）。

    ## ready_ev 的必要性

    **不能**用 `srv.model` 这种 Python 属性通知父进程 —— fork 后是写时复制，
    父进程读不到子进程设的属性（第 21 轮踩过，父进程卡 60 秒超时）。
    """
    try:
        import threading
        from alphastarmini.core.arch.agent import Agent

        a = Agent()                      # 自己的副本（不指向 model_learner）
        a.model.to(device)
        a.model.eval()
        audit_model_mode("batch_server", a.model)
        srv.agent = a

        # 先把当前权重灌一次，避免 server 用随机初始化跑第一批
        try:
            srv.pending_weights = {k: v.detach().clone()
                                   for k, v in model_learner.state_dict().items()}
        except Exception as e:
            print("[batch_server] 初始权重拷贝失败（下一轮会重试）: %r" % e, flush=True)

        stop_ev = threading.Event()

        def _sync_loop():
            _last = -1
            while not stop_ev.is_set():
                try:
                    if snapshot is not None:
                        # 第 73 轮（Kimi-05）：从**全局唯一快照**读。
                        # 只读变化的版本，且读到的字节与 (ver, hash) 严格对应。
                        # 轮询 0.5 s 只是检查版本号，真正拷贝只在版本变化时发生。
                        got = snapshot.read_if_newer(_last)
                        if got is not None:
                            _ver, _hash, _sd = got
                            srv.pending_weights = _sd
                            srv.pending_version = _ver
                            srv.pending_hash = _hash
                            _last = _ver
                    else:
                        # 回落路径：无共享快照时维持旧行为（每 Worker 各拷一份）
                        if pub_lock is not None:
                            pub_lock.acquire()
                        try:
                            _sd = {k: v.detach().clone()
                                   for k, v in model_learner.state_dict().items()}
                            _ver = int(pv.value) if pv is not None else -1
                        finally:
                            if pub_lock is not None:
                                pub_lock.release()
                        srv.pending_weights = _sd
                        srv.pending_version = _ver
                except Exception:
                    pass
                stop_ev.wait(0.5 if snapshot is not None else sync_interval)

        th = threading.Thread(target=_sync_loop, daemon=True)
        th.start()

        srv.policy_version_ref = pv
        print("[batch_server] 就绪（独立模型副本 + %.0fs 周期同步到批边界，pub_lock=%s，"
              "snapshot=%s），device=%s"
              % (sync_interval, pub_lock is not None,
                 "共享快照" if snapshot is not None else "各自拷贝", device), flush=True)
        ready_ev.set()

        rounds = srv.run(max_rounds=None, idle_timeout=idle_timeout, supervise=False)
        stop_ev.set()
        print("[batch_server] 退出，rounds=%s %s" % (rounds, srv.async_summary()), flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            srv.mark_exited(0)
        except Exception:
            pass
        ready_ev.set()
        print("[batch_server] 异常退出: %r" % e, flush=True)


def Worker(synchronizer, rank, optimizer, q_winloss, q_points, v_steps, use_cuda_device, model_learner, device_learner, log_path, model_teacher=None, device_teacher=None, batch_srv=None, pub_lock=None, pv=None):
    torch.manual_seed(RANDOM_SEED + rank)

    # ---- 第 24 轮：spawn 子进程必须自己把 absl flags 解析掉 ----
    # 现象：spawn 模式下子进程重新 import 本模块，flags 处于「未解析」状态；
    #       而创建 SC2Env 时会访问 `--sc2_run_config`，于是抛
    #         absl.flags._exceptions.UnparsedFlagAccessError:
    #         Trying to access flag --sc2_run_config before flags were parsed.
    #       fork 模式不会有这个问题（继承父进程已解析的状态），
    #       但 `test()` 在父进程已经初始化了 CUDA，fork 又会报
    #         Cannot re-initialize CUDA in forked subprocess
    #       —— 所以只能 spawn + 在这里补解析。这是 spawn 的常规做法。
    try:
        from absl import flags as _absl_flags
        if not _absl_flags.FLAGS.is_parsed():
            try:
                _absl_flags.FLAGS(["x"])          # 用全部默认值解析
            except Exception:
                _absl_flags.FLAGS.mark_as_parsed()
    except Exception:
        pass

    # with synchronizer:
    #     print('module name:', "worker")
    #     print('parent process:', os.getppid())
    #     print('process id:', os.getpid())

    # ---- 第 66 轮：把 policy_version 的共享变量挂到**本进程**的 srv 上 ----
    # 踩过的坑：`Batch_Server_Worker` 里也有一行 `srv.policy_version_ref = pv`，
    # 但那是在 **server 进程**里设的，actor 所在的 Worker 进程看不到
    # （fork/spawn 后是两份内存，同第 21 轮"父进程 getattr(srv,'model') 永远 None"）。
    # 结果 `read_policy_version()` 回落到 `applied_version`（也只有 server 进程有），
    # 落盘里记成 -1。这里在 Worker 进程里再挂一次，actor 就能读到真正的版本号。
    if batch_srv is not None and pv is not None:
        try:
            batch_srv.policy_version_ref = pv
        except Exception:
            pass

    # FIX(T0): 原实现 `str(rank)` 在 rank>=4 时会取到不存在的 cuda:4..7，
    # 且 rank>=8 分支 `(rank-8)%7+1` 同样越界 -> 真实训练配置下必然报错。
    cuda_device = pick_cuda_device(rank, use_cuda_device)

    league = League(
        initial_agents={
            race: get_supervised_agent(race, path=MODEL_PATH, model_type=MODEL_TYPE, 
                                       restore=True, device=cuda_device)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=0,
        league_exploiters=0)

    # we only call the first process (rank 0) to save the loss info
    if rank == 0:
        need_save_result = NEED_SAVE_RESULT
    else:
        need_save_result = 0

    if need_save_result:
        summary_path = log_path + "_" + str(rank) + "/"
        writer = SummaryWriter(summary_path) if need_save_result else None
    else:
        writer = None

    # results_lock = threading.Lock()
    # coordinator = Coordinator(league, winrate_scale=WINRATE_SCALE, output_file=OUTPUT_FILE, results_lock=results_lock, writer=writer)
    # coordinator.set_uninitialed_results(actor_nums=ACTOR_NUMS, episode_nums=MAX_EPISODES)

    learners = []
    actors = []

    process_lock = synchronizer if USE_UPDATE_LOCK else None

    try:

        for idx in range(league.get_learning_players_num()):
            player = league.get_learning_player(idx)

            # player.agent.agent_nn.model = model_learner
            # player.agent.agent_nn.model.load_state_dict(model_learner.state_dict())
            if use_cuda_device:
                player.agent.agent_nn.model.to(cuda_device)

            player.agent.set_rl_training(IS_TRAINING)

            buffer_lock = threading.Lock()
            learner = Learner(player, rank, v_steps, cuda_device, optimizer=optimizer, global_model=model_learner, 
                              max_time_for_training=MAX_TIME_FOR_TRAINING, lr=LR, 
                              weight_decay=WEIGHT_DECAY, baseline_weight=BASELINE_WEIGHT, is_training=IS_TRAINING, 
                              buffer_lock=buffer_lock, writer=writer, use_opponent_state=USE_OPPONENT_STATE,
                              no_replay_learn=NO_REPLAY_LEARN, num_epochs=NUM_EPOCHS,
                              count_of_batches=COUNT_OF_BATCHES, buffer_size=BUFFER_SIZE,
                              use_random_sample=USE_RANDOM_SAMPLE, only_update_baseline=ONLY_UPDATE_BASELINE,
                              need_save_result=need_save_result, process_lock=process_lock,
                              update_params_interval=UPDATE_PARAMS_INTERVAL)
            learners.append(learner)

            teacher = get_supervised_agent(player.race, model_type="sl", restore=True, device=cuda_device)
            teacher.set_rl_training(IS_TRAINING)
            # 第 64 轮（Kimi-06）：teacher 是**固定的参考分布**，必须 eval。
            # train 模式下 BN 会用单样本统计量，teacher_logits 会随每个状态抖动，
            # 而 KL(student||teacher) 的 teacher 一侧本应是确定的。
            try:
                teacher.agent_nn.model.eval()
            except Exception as e:
                print("[Worker %d] teacher.eval() 失败: %r" % (rank, e))
            if rank == 0:
                audit_model_mode("teacher", teacher.agent_nn.model)

            # teacher.agent_nn.model = model_teacher
            if use_cuda_device:
                teacher.agent_nn.model.to(cuda_device)

            for z in range(ACTOR_NUMS):
                device = torch.device(cuda_device if use_cuda_device else "cpu")
                agent_id = rank * ACTOR_NUMS + z
                # 第 22 轮：把中心批推理 server 传到 actor 里。
                # `batch_idx` 用 **agent_id**（= rank*ACTOR_NUMS+z，全局唯一），
                # 不是 rank —— server 的 N 是「总 actor 数」，用 rank 会撞车。
                actor = ActorVSComputer(player, q_winloss, q_points, device, model_learner, None, teacher, agent_id, None, None, None,
                                        batch_srv=batch_srv, batch_idx=agent_id)
                actors.append(actor)

        threads = []
        for l in learners:
            l.start()
            threads.append(l.thread)
            sleep(1)
        for a in actors:
            a.start()
            threads.append(a.thread)
            sleep(1)

        # Wait for training to finish.
        for t in threads:
            t.join()

        # coordinator.write_eval_results()

    except Exception as e:
        print("Worker Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())
        pass

    finally:
        # 第 64 轮（Kimi-02）：把"Learner 到底更新了几次"打出来。
        # 之前只有 loss 有限/backward 不报错这类一次性证据，
        # 无法区分"真的在学"和"Learner 线程早就静默退出了但主进程 rc=0"。
        try:
            _tot = 0
            for _i, _l in enumerate(learners):
                _n = int(getattr(_l, "update_count", -1))
                _tot += max(_n, 0)
                print("[Worker %d] learner[%d] optimizer_steps=%d last_loss=%s"
                      % (rank, _i, _n, getattr(_l, "last_loss", None)), flush=True)
                # 第 73 轮（Kimi-02）：更新是否健康 / 轨迹吞吐
                # ⚠️ 打印不能用 `x or -1`：update_norm 合法值可能是 0.0，
                #    `0.0 or -1` 会变成 -1，看起来像"没测"。第一版就栽在这里。
                _gn_last = getattr(_l, "grad_norm_last", None)
                _un_last = getattr(_l, "update_norm_last", None)
                print("[Worker %d] learner[%d] grad_norm last=%s max=%.4g nan=%d | "
                      "update_norm last=%s max=%s | traj recv=%s"
                      % (rank, _i,
                         "n/a" if _gn_last is None else "%.4g" % _gn_last,
                         getattr(_l, "grad_norm_max", -1),
                         getattr(_l, "nan_grad_steps", -1),
                         "n/a" if _un_last is None else "%.4g" % _un_last,
                         getattr(_l, "update_norm_max", None),
                         getattr(_l, "traj_recv", {})), flush=True)
                _klmin = getattr(_l, "kl_min", {}) or {}
                if _klmin:
                    print("[Worker %d] KL 极值（renorm 下应全部 >= 0）: %s"
                          % (rank, ", ".join("%s∈[%.3e, %.3e]"
                                             % (k, _klmin[k], (_l.kl_max or {}).get(k, float("nan")))
                                             for k in sorted(_klmin))), flush=True)
            print("[Worker %d] optimizer_steps 合计=%d" % (rank, _tot), flush=True)
        except Exception as _e:
            print("[Worker %d] 更新计数打印失败: %r" % (rank, _e), flush=True)


def Parameter_Server(synchronizer, q_winloss, q_points, v_steps, use_cuda_device, model, log_path, model_path):
    torch.manual_seed(RANDOM_SEED)

    # with synchronizer:
    #     print('module name:', "Parameter_Server")
    #     print('parent process:', os.getppid())
    #     print('process id:', os.getpid())

    summary_path = log_path + "/"
    writer = SummaryWriter(summary_path)

    update_counter = 0
    max_win_rate, latest_win_rate = 0., 0.
    max_mean_points, latest_mean_points = 0., 0.
    outcome_list, points_list = [], []
    win_rate_list, mean_points_list = [], []

    # 第 50 轮：队列超时可配（默认保持 1800 s，不改变既有行为）。
    # 原来硬编码 `60*30`：所有 actor 退出后，Parameter_Server 还要**空转等 30 min**
    # 才因 `queue.Empty` 退出，而 `test()` 是在 `p.join()` 上等它 ——
    # 一次 30 min 的长跑会变成 60 min 才返回（长跑测试的成本直接翻倍）。
    # 现在有两条更快的退出路径：① 结束哨兵（test() 的 reaper 投递）；
    # ② 把本变量调小。哨兵是主路径，超时是兜底。
    _q_timeout = float(os.environ.get("MAS_QUEUE_TIMEOUT", 60 * 30))

    train_iters = TRAIN_ITERS 
    static_num = STATIC_NUM

    # 第 84 轮（DS，#48b）：`int(train_iters / static_num)` 在小规模验证时会算成 0，
    # 于是 shape = (0, static_num)，第一次 `episode_outcome[row, col] = ...` 就 IndexError。
    # 冒烟/短跑（MAX_EPISODES × ACTOR_NUMS × PARALLEL < STATIC_NUM）必然触发。
    # 至少留一行，`row` 已经有 min(row, shape[0]-1) 的钳位。
    _n_rows = max(1, int(train_iters / static_num))
    episode_outcome = np.ones([_n_rows, static_num], dtype=np.float64) * (-1e9)
    episode_points = np.ones([_n_rows, static_num], dtype=np.float64) * (-1e9)
    print("[Parameter_Server] STATIC_NUM=%d  train_iters=%d  统计表 shape=%s"
          % (static_num, train_iters, episode_outcome.shape), flush=True)

    try: 
        while update_counter < train_iters:
            outcome = q_winloss.get(timeout=_q_timeout)
            # 第 50 轮：结束哨兵。`test()` 在所有 Worker 退出后投递 None，
            # 这里必须**先判空**再 append —— 否则 None 会流进 `(outcome == 1)` 的统计里。
            if outcome is None:
                print("[Parameter_Server] 收到结束哨兵（Worker 已全部退出），正常收尾", flush=True)
                break
            outcome_list.append(outcome)
            print("Parameter_Server winloss_list", outcome_list) if 1 else None

            points = q_points.get(timeout=_q_timeout)
            if points is None:
                print("[Parameter_Server] 收到结束哨兵（points 队列），正常收尾", flush=True)
                break
            points_list.append(points)
            print("Parameter_Server points_list", points_list) if 1 else None

            row = int(update_counter / static_num)
            col = int(update_counter % static_num)
            row = min(row, episode_outcome.shape[0] - 1)
            col = min(col, episode_outcome.shape[1] - 1)

            episode_outcome[row, col] = outcome
            print("episode_outcome", episode_outcome) if debug else None

            single_episode_outcome = episode_outcome[row]
            if not (single_episode_outcome == (-1e9)).any():
                win_rate = (single_episode_outcome == 1).sum() / len(single_episode_outcome)
                print("Iter:", row, "win_rate:", win_rate) if 1 else None

                writer.add_scalar('winrate/every_' + str(static_num) + '_episodes', win_rate, row + 1)
                writer.add_scalar('winrate/update_steps', win_rate, int(v_steps.value))
                win_rate_list.append(format(win_rate, '.3f'))

                if win_rate >= max_win_rate:
                    with synchronizer:
                        torch.save(model.state_dict(), model_path + ".pth")
                    max_win_rate = win_rate
                elif ONLY_UPDATE_BASELINE or SAVE_LATEST:
                    with synchronizer:
                        torch.save(model.state_dict(), model_path + ".pth")

                latest_win_rate = win_rate
                del win_rate

            del single_episode_outcome

            episode_points[row, col] = points
            print("episode_points", episode_points) if debug else None

            single_episode_points = episode_points[row]
            if not (single_episode_points == (-1e9)).any():
                mean_points = np.mean(single_episode_points)
                print("Iter:", row, "mean_points:", mean_points) if 1 else None

                writer.add_scalar('meanpoints/every_' + str(static_num) + '_episodes', mean_points, row + 1)
                writer.add_scalar('meanpoints/update_steps', mean_points, int(v_steps.value))
                mean_points_list.append(format(mean_points, '.3f'))

                if mean_points >= max_mean_points:
                    with synchronizer:
                        torch.save(model.state_dict(), model_path + ".pth")
                    max_mean_points = mean_points
                elif ONLY_UPDATE_BASELINE or SAVE_LATEST:
                    with synchronizer:
                        torch.save(model.state_dict(), model_path + ".pth")

                latest_mean_points = mean_points
                del mean_points

            del single_episode_points

            # print("Parameter_Server", os.getpid())
            # objgraph.show_growth(limit=5)

            update_counter += 1
            gc.collect()

    except Exception as e: 
        print("Parameter_Server Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())
        pass

    finally:
        print("Parameter_Server end:")
        print("--------------------")
        # 第 64 轮（Kimi-02）：全局样本计数（由各 Learner 累加，注意 mp.Value 无锁，
        # 是多进程 read-modify-write，可能少计；这里只做量级校验，不做精确统计）
        try:
            print("v_steps(samples)=%.0f  ≈ optimizer 更新数 %.0f（按 bs×seq 反算）"
                  % (v_steps.value,
                     v_steps.value / max(1, AHP.batch_size * AHP.sequence_length)), flush=True)
        except Exception:
            pass
        print("mean_points_list:", mean_points_list)
        print("latest_mean_points:", latest_mean_points)
        print("max_mean_points:", max_mean_points)
        print("win_rate_list:", win_rate_list)
        print("latest_win_rate:", latest_win_rate)
        print("max_win_rate:", max_win_rate)
        print("--------------------")


def test(on_server=False, replay_path=None):
    if SIMPLE_TEST:
        use_cuda_device = False
    else:
        use_cuda_device = True

    torch.manual_seed(RANDOM_SEED)
    # 第 24 轮：原实现是无条件 `mp.set_start_method('spawn')`，实测抛
    #   RuntimeError: context has already been set
    # 而 `test()` 崩在开头会让已启动的子进程没人 join，父进程卡死（实测 rc=124 超时 290 s）。
    #
    # 另外：**spawn 对本项目不友好** —— 子进程会重新 import 主模块，
    # 若调用方脚本没有 `if __name__ == "__main__":` 保护，会递归再跑一遍 `test()`
    # （实测 `[test] start method` 打印了两次，然后抛空消息的 RuntimeError，
    #   那正是 spawn 的 "bootstrapping phase" 保护触发的）。
    # fork 不重新 import，且共享内存张量直接继承 —— 对本项目的 `batch_srv` 更可靠。
    # 默认仍保持 spawn（不改变原有行为），需要 fork 时用环境变量：
    #   MAS_START_METHOD=fork
    _method = os.environ.get("MAS_START_METHOD", "spawn")
    try:
        mp.set_start_method(_method, force=True)
    except Exception as e:
        print("[test] set_start_method(%r) 失败: %r" % (_method, e), flush=True)
    print("[test] multiprocessing start method = %s（请求 %s）"
          % (mp.get_start_method(), _method), flush=True)

    model_save_path = MODEL_SAVE_PATH 

    log_path = SUMMARY_PATH

    device_learner = torch.device("cuda:0" if use_cuda_device else "cpu")
    league = League(
        initial_agents={
            race: get_supervised_agent(race, path=MODEL_PATH, model_type=MODEL_TYPE, 
                                       restore=RESTORE, device=device_learner)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=0,
        league_exploiters=0)

    player = league.get_learning_player(0)
    player.agent.set_rl_training(IS_TRAINING)
    if ON_GPU:
        player.agent.agent_nn.to(device_learner)

    model_learner = player.agent.agent_nn.model
    model_learner.share_memory()

    if 0:
        optimizer = SA.MorvanZhouSharedAdam(model_learner.parameters(), lr=LR, betas=(THP.beta1, THP.beta2), 
                                            eps=THP.epsilon, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = SA.IkostrikovSharedAdam(model_learner.parameters(), lr=LR, betas=(THP.beta1, THP.beta2), 
                                            eps=THP.epsilon, weight_decay=WEIGHT_DECAY)
        optimizer.share_memory()

    synchronizer = mp.Lock()
    processes = []

    q_winloss = mp.Queue(maxsize=TRAIN_ITERS * 24)
    q_points = mp.Queue(maxsize=TRAIN_ITERS * 24)
    v_steps = mp.Value('d', 0.0)

    # ---- 第 65 轮（Kimi-05 P0）：原子权重发布 + policy_version ----
    # pub_lock: Learner 的 optimizer.step() 与 server 的"拷快照"互斥，
    #           保证快照只会落在某次更新的前/后，不会跨张量半新半旧。
    # policy_version: 每次更新 +1，单调递增，供 server/轨迹追溯"这一步用的是哪版策略"。
    pub_lock = mp.Lock()
    policy_version = mp.Value('i', 0)

    # ---- 第 73 轮（Kimi-05）：全局唯一不可变快照 ----
    # 原实现是"每个 Worker 各拷一份 state_dict"，问题：15 份拷贝、版本不可比、
    # 而且没有哈希能证明"这份字节 = 那个版本"。改成一块共享缓冲：
    # 父进程的发布线程写入字节 -> 版本 +1 -> 记 md5；所有 server 从同一块缓冲读。
    snapshot = None
    if os.environ.get("MAS_BATCH_ACTING", "0") == "1":
        try:
            snapshot = SharedSnapshot(model_learner.state_dict())
            print("[snapshot] 共享快照缓冲就绪：%d 个张量 / %.2f MB"
                  % (len(snapshot.meta), snapshot.total / 1024.0 / 1024.0), flush=True)
        except Exception as e:
            print("[snapshot] 建立失败，回落到每 Worker 各自拷贝: %r" % (e,), flush=True)
            snapshot = None

    if snapshot is not None:
        _freeze = os.environ.get("MAS_DUMP_FREEZE", "0") == "1"
        _snap_int = float(os.environ.get("MAS_SNAPSHOT_INTERVAL", "8.0"))

        def _publish_loop():
            _logged = False
            while True:
                try:
                    if pub_lock is not None:
                        pub_lock.acquire()
                    try:
                        _v, _h = snapshot.publish(model_learner.state_dict())
                    finally:
                        if pub_lock is not None:
                            pub_lock.release()
                    if not _logged:
                        _logged = True
                        print("[snapshot] 首次发布 ver=%d hash=%s" % (_v, _h), flush=True)
                    if _freeze:
                        print("[snapshot] MAS_DUMP_FREEZE=1 -> 首发布后冻结（ver=%d, hash=%s）"
                              % (_v, _h), flush=True)
                        return
                except Exception as e:
                    print("[snapshot] 发布失败: %r" % (e,), flush=True)
                sleep(_snap_int)

        threading.Thread(target=_publish_loop, daemon=True).start()

    # ---- 第 23 轮：中心批推理 server ----
    # ⚠️ 必须在 fork Worker **之前**创建 `AsyncBatchInferenceServer`：
    #    它的所有槽位（state / 结果 / hidden / logits）都是 share_memory 张量，
    #    创建之后再 fork 才能让所有 Worker 看到同一块。
    #    若在 Worker 里创建，每个 Worker 会拿到**各自独立**的缓冲，批推理彻底失效。
    batch_srv = None
    if os.environ.get("MAS_BATCH_ACTING", "0") == "1":
        total_actors = ACTOR_NUMS * PARALLEL
        # 第 64 轮（Kimi-14 第 3 条）：允许把 server 放到**指定的一张卡**上。
        # 动机：现在 batch_dev = pick_cuda_device(0) = cuda:1，与 rank0 的 Worker 撞车；
        # §61.4 已测出 forward 在抢卡时慢 7.7 倍（斜率 3.5 vs 0.455 ms/sample）。
        # 默认不设 -> 行为与之前**完全一致**；设 MAS_SERVER_GPU=3 则 server 独占 cuda:3。
        # ⚠️ 注意：Worker 的卡由 MAS_GPU_LIST 决定，两者要配合着改，否则又会撞。
        _srv_gpu = os.environ.get("MAS_SERVER_GPU", "").strip()
        batch_dev = torch.device(("cuda:" + _srv_gpu) if _srv_gpu
                                 else pick_cuda_device(0, use_cuda_device))
        batch_srv = AsyncBatchInferenceServer(
            None, total_actors, batch_dev, make_batch_state_shapes(),
            max_units=128, max_selected=AHP.max_selected,
            max_batch=total_actors, return_logits=True)
        # 挂到 batch_srv 上：fork 之前设置，所有 Worker/actor 都能读到同一块缓冲。
        # 轨迹落盘要用的正是"产生这条轨迹的那份权重"，而不是 global_model 的当前值。
        batch_srv.snapshot = snapshot
        ready_ev = mp.Event()
        bs = mp.Process(target=Batch_Server_Worker,
                        args=(batch_srv, model_learner, batch_dev, ready_ev),
                        kwargs=dict(pub_lock=pub_lock, pv=policy_version,
                                    snapshot=snapshot), daemon=True)
        bs.start()
        processes.append(bs)
        if not ready_ev.wait(timeout=180):
            raise RuntimeError("batch server 180 s 内未就绪，拒绝继续（避免 Worker 空转死等）")
        print("[test] batch acting 已开启：%d 个 actor 共用一个 server（%s）"
              % (total_actors, batch_dev), flush=True)

    worker_procs = []
    for rank in range(PARALLEL):
        p = mp.Process(target=Worker, args=(synchronizer, rank, optimizer, q_winloss, q_points, v_steps,
                                            use_cuda_device, model_learner, device_learner, log_path,
                                            None, None, batch_srv),
                       kwargs=dict(pub_lock=pub_lock, pv=policy_version))
        p.start()
        processes.append(p)
        worker_procs.append(p)

    ps = mp.Process(target=Parameter_Server, args=(synchronizer, q_winloss, q_points, v_steps, 
                                                   use_cuda_device, model_learner, log_path, model_save_path))
    ps.start()
    processes.append(ps)

    # ---- 第 50 轮：收尾哨兵 ----
    # 为什么需要：`Parameter_Server` 的唯一退出条件是「收满 TRAIN_ITERS 条 episode 结果」，
    # 而 actor 是**按墙钟**退出的（`max_time_for_training`）——
    # 长跑里两者永远对不上，于是 4 个 actor 都退出后 Parameter_Server 还在
    # `q_winloss.get()` 上阻塞最多 30 min，`p.join()` 也就跟着卡 30 min。
    # 实测一次 MAS_MAX_TIME=1800 的跑要 60 min 才返回，长跑测试成本翻倍。
    #
    # 解法：起一个 reaper 线程，只等 **Worker** 进程（不含 Parameter_Server，否则死锁），
    # 全部退出后投递哨兵。投 2 份是因为两个队列各要一份，且可能有竞态下多取一次。
    def _reaper():
        for p in worker_procs:
            try:
                p.join()
            except Exception:
                pass
        print("[test] 所有 Worker 已退出，向 Parameter_Server 投递结束哨兵", flush=True)
        for _ in range(2):
            try:
                q_winloss.put_nowait(None)
                q_points.put_nowait(None)
            except Exception as e:
                print("[test] 投递哨兵失败（不影响正确性，最多等满队列超时）: %r" % (e,), flush=True)

    threading.Thread(target=_reaper, daemon=True).start()

    # ---- 第 84 轮（DS，#49）：运行期心跳 ----
    # 为什么需要：RL 是 8+ 个进程，出问题时日志会"一片安静"——分不清是
    #   ① 在正常跑（只是慢）② 死锁 ③ 某个 Worker 已经崩了但父进程还在 join。
    # 前几轮的教训是"没有观测就没有诊断"，这里每 N 秒打一行可判读的状态。
    #   MAS_HEARTBEAT_SEC=60
    _hb_interval = float(os.environ.get("MAS_HEARTBEAT_SEC", "0") or 0)
    if _hb_interval > 0:
        def _mem_avail_gb():
            try:
                with open("/proc/meminfo") as _f:
                    for _ln in _f:
                        if _ln.startswith("MemAvailable:"):
                            return int(_ln.split()[1]) / 1024.0 / 1024.0
            except Exception:
                pass
            return -1.0

        def _heartbeat():
            _t0 = time()
            while True:
                sleep(_hb_interval)
                try:
                    _alive = sum(1 for _p in processes if _p.is_alive())
                    try:
                        _q = q_winloss.qsize()
                    except Exception:
                        _q = -1
                    _gpu = ""
                    try:
                        if torch.cuda.is_available():
                            _gpu = " gpu0=%.1fGB" % (torch.cuda.memory_allocated(0) / 1e9)
                    except Exception:
                        pass
                    print("[hb] t=%5.0fs  proc=%d/%d  q_winloss=%d  v_steps=%.0f  mem_avail=%.1fGB%s"
                          % (time() - _t0, _alive, len(processes), _q,
                             v_steps.value, _mem_avail_gb(), _gpu), flush=True)
                except Exception as _e:
                    print("[hb] 心跳失败: %r" % (_e,), flush=True)

        threading.Thread(target=_heartbeat, daemon=True).start()
        print("[test] 心跳已开启：每 %.0f 秒一行 [hb]" % _hb_interval, flush=True)

    for p in processes:
        p.join()
