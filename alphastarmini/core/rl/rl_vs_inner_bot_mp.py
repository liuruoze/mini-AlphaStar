#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by fighting against built-in AI (computer), using no replays"

# import objgraph

from time import time, sleep, strftime, localtime
import gc
import os
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


SIMPLE_TEST = not P.on_server
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

STATIC_NUM = 100  # ACTOR_NUMS * PARALLEL
TRAIN_ITERS = MAX_EPISODES * ACTOR_NUMS * PARALLEL

USE_UPDATE_LOCK = False
DIFFICULTY = 1
ONLY_UPDATE_BASELINE = False
SAVE_LATEST = False
BASELINE_WEIGHT = 1
LR = 1e-6  # 1e-5
WEIGHT_DECAY = 1e-5

MAX_TIME_FOR_TRAINING = 60 * 60 * 24 * 7

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
                 max_frames_per_episode=22.4 * MAX_TIME_FOR_TRAINING, max_frames=MAX_FRAMES, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING,
                 replay_dir="./added_simple64_replays/",
                 update_params_interval=UPDATE_PARAMS_INTERVAL,
                 need_save_result=NEED_SAVE_RESULT):
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
                                    print("agent_{:d} update params".format(self.idx)) if debug else None
                                    self.agent.set_weights(self.player.agent.get_weights())
                                    update_params_timer = time()

                                state = self.agent.agent_nn.preprocess_state_all(home_obs.observation, 
                                                                                 build_order=player_bo, 
                                                                                 last_list=last_list)
                                baseline_state = self.agent.agent_nn.get_baseline_state_from_multi_source_state(home_obs.observation, state)

                                with torch.no_grad():
                                    player_function_call, player_action, player_logits, \
                                        player_new_memory, player_select_units_num, entity_num = self.agent.step_from_state(state, 
                                                                                                                            player_memory, 
                                                                                                                            obs=home_obs.observation)

                                print("player_function_call:", player_function_call) if debug else None
                                print("player_action.delay:", player_action.delay) if debug else None

                                print("entity_num:", entity_num) if debug else None
                                print("player_select_units_num:", player_select_units_num) if debug else None
                                print("player_action:", player_action) if debug else None

                                if False:
                                    show_sth(home_obs, player_action)

                                expected_delay = player_action.delay.item()
                                step_mul = max(1, expected_delay)
                                print("step_mul:", step_mul) if debug else None

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
                                    print("agent_{:d} get outcome".format(self.idx), outcome) if 1 else None

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
                                    memory=player_memory,
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

                                # player_memory = tuple(h.detach().clone() for h in player_new_memory)
                                player_memory = player_new_memory
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
            # print('agent_', self.idx, "total_time: ", total_time / 60.0, "min") if debug else None

            # if debug and SAVE_STATISTIC: 
            #     with self.results_lock:
            #         self.coordinator.send_eval_results(self.player, DIFFICULTY, food_used_list, army_count_list, 
            #                                            collected_points_list, used_points_list, 
            #                                            killed_points_list, steps_list, total_time)

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


def Worker(synchronizer, rank, optimizer, q_winloss, q_points, v_steps, use_cuda_device, model_learner, device_learner, log_path, model_teacher=None, device_teacher=None):
    torch.manual_seed(RANDOM_SEED + rank)

    # with synchronizer:
    #     print('module name:', "worker")
    #     print('parent process:', os.getppid())
    #     print('process id:', os.getpid())

    if rank < 8:
        cuda_device = "cuda:" + str(rank) if use_cuda_device else 'cpu'
    else:
        new_rank = (rank - 8) % 7 + 1
        cuda_device = "cuda:" + str(new_rank) if use_cuda_device else 'cpu'

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

            # teacher.agent_nn.model = model_teacher
            if use_cuda_device:
                teacher.agent_nn.model.to(cuda_device)

            for z in range(ACTOR_NUMS):
                device = torch.device(cuda_device if use_cuda_device else "cpu")
                agent_id = rank * ACTOR_NUMS + z
                actor = ActorVSComputer(player, q_winloss, q_points, device, model_learner, None, teacher, agent_id, None, None, None)
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
        pass


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

    train_iters = TRAIN_ITERS 
    static_num = STATIC_NUM

    episode_outcome = np.ones([int(train_iters / static_num), static_num], dtype=np.float) * (-1e9)
    episode_points = np.ones([int(train_iters / static_num), static_num], dtype=np.float) * (-1e9)

    try: 
        while update_counter < train_iters:
            outcome = q_winloss.get(timeout=60 * 30)
            outcome_list.append(outcome)
            print("Parameter_Server winloss_list", outcome_list) if 1 else None

            points = q_points.get(timeout=60 * 30)
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
                writer.add_scalar('winrate/update_steps', win_rate, v_steps.value)
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
                writer.add_scalar('meanpoints/update_steps', mean_points, v_steps.value)
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
    mp.set_start_method('spawn')

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

    for rank in range(PARALLEL):
        p = mp.Process(target=Worker, args=(synchronizer, rank, optimizer, q_winloss, q_points, v_steps,
                                            use_cuda_device, model_learner, device_learner, log_path))
        p.start()
        processes.append(p)

    ps = mp.Process(target=Parameter_Server, args=(synchronizer, q_winloss, q_points, v_steps, 
                                                   use_cuda_device, model_learner, log_path, model_save_path))
    ps.start()
    processes.append(ps)

    for p in processes:
        p.join()
