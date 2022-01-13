#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by fighting against built-in AI (computer), using no replays"

from time import time, sleep, strftime, localtime
import gc
import os
import random
import traceback
from multiprocessing import Process, Manager, Lock
import datetime

import numpy as np

import torch

from tensorboardX import SummaryWriter

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent, Race, Bot, Difficulty, BotBuild
from pysc2.lib import actions as sc2_actions
from pysc2.lib import units as sc2_units

from alphastarmini.core.rl.rl_utils import Trajectory, get_supervised_agent
from alphastarmini.core.rl.learner_process import LearnerProcess
from alphastarmini.core.rl import rl_utils as RU

from alphastarmini.lib import utils as L

from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


SIMPLE_TEST = not P.on_server
if SIMPLE_TEST:
    MAX_EPISODES = 2
    ACTOR_NUMS = 1
    GAME_STEPS_PER_EPISODE = 500  
else:
    MAX_EPISODES = 6
    ACTOR_NUMS = 4
    GAME_STEPS_PER_EPISODE = 18000 

DIFFICULTY = 2
ONLY_UPDATE_BASELINE = False
LR = 1e-5  # 0  # 1e-5

USE_DEFINED_REWARD_AS_REWARD = False
USE_RESULT_REWARD = True
REWARD_SCALE = 1e-3

BUFFER_SIZE = 1  # 100
COUNT_OF_BATCHES = 1  # 10

NUM_EPOCHS = 1  # 20
USE_RANDOM_SAMPLE = False

UPDATE_PARAMS_INTERVAL = 60

RESTORE = True
SAVE_STATISTIC = True
RANDOM_SEED = 1

# model path
MODEL_TYPE = "rl"
MODEL_PATH = "./model/"
OUTPUT_FILE = './outputs/rl_vs_inner_bot.txt'

VERSION = SCHP.game_version
MAP_NAME = SCHP.map_name
STEP_MUL = 8

SAVE_REPLAY = False
IS_TRAINING = True

WIN_THRESHOLD = 4000
USE_OPPONENT_STATE = False
NO_REPLAY_LEARN = True

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


class ActorVSComputerProcess:
    """A single actor loop that generates trajectories by playing with built-in AI (computer).

    We don't use batched inference here, but it was used in practice.
    """

    def __init__(self, idx, 
                 trajectories=None, final_trajectories=None,
                 win_trajectories=None,
                 max_time_for_training=60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 4,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING,
                 replay_dir="./added_simple64_replays/",
                 update_params_interval=UPDATE_PARAMS_INTERVAL):

        self.idx = idx

        teacher = get_supervised_agent(Race.protoss, model_type="sl", restore=True)
        device_teacher = torch.device("cuda:1" if True else "cpu")
        if ON_GPU:
            teacher.agent_nn.to(device_teacher)
        self.teacher = teacher

        self.agent = get_supervised_agent(Race.protoss, path=MODEL_PATH, model_type=MODEL_TYPE, restore=RESTORE)
        device = torch.device("cuda:" + str(1) if True else "cpu")
        if ON_GPU:
            self.agent.agent_nn.to(device)

        self.max_time_for_training = max_time_for_training
        self.max_time_per_one_opponent = max_time_per_one_opponent
        self.max_frames_per_episode = max_frames_per_episode
        self.max_frames = max_frames
        self.max_episodes = max_episodes
        self.is_training = is_training

        self.process = Process(target=self.run, args=())

        self.process.daemon = True                            
        self.is_running = True
        self.is_start = False

        self.replay_dir = replay_dir
        self.update_params_interval = update_params_interval

        self.trajectories = trajectories
        self.final_trajectories = final_trajectories
        self.win_trajectories = win_trajectories

    def start(self):
        self.is_start = True
        self.process.start()

    # background
    def run(self):
        try:
            with torch.no_grad():
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

                # judge the trajectory whether contains final
                is_final_trajectory = False
                is_win_trajectory = False

                # use max_episodes to end the loop
                while time() - training_start_time < self.max_time_for_training:
                    agents = [self.agent]

                    with self.create_env_one_player() as env:

                        # set the obs and action spec
                        observation_spec = env.observation_spec()
                        action_spec = env.action_spec()

                        for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                            agent.setup(obs_spec, act_spec)

                        self.teacher.setup(self.agent.obs_spec, self.agent.action_spec)

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
                            print("total_episodes:", total_episodes)

                            timesteps = env.reset()
                            for a in agents:
                                a.reset()

                            [home_obs] = timesteps
                            is_final = home_obs.last()

                            player_memory = self.agent.initial_state()
                            #teacher_memory = self.teacher.initial_state()

                            episode_frames = 0

                            # initial build order
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

                            while not is_final:
                                total_frames += 1
                                episode_frames += 1

                                t = time()

                                # every 10s, the actor get the params from the learner
                                if time() - update_params_timer > self.update_params_interval:
                                    print("agent_{:d} update params".format(self.idx)) if debug else None
                                    # self.agent.set_weights(self.player.agent.get_weights())
                                    update_params_timer = time()

                                state = self.agent.agent_nn.preprocess_state_all(home_obs.observation, 
                                                                                 build_order=player_bo, 
                                                                                 last_list=last_list)
                                baseline_state = self.agent.agent_nn.get_baseline_state_from_multi_source_state(home_obs.observation, state)

                                with torch.no_grad():
                                    player_function_call, player_action, player_logits, \
                                        player_new_memory, player_select_units_num, entity_num = self.agent.step_from_state(state, player_memory)

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

                                reward = home_next_obs.reward
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

                                if USE_DEFINED_REWARD_AS_REWARD:
                                    if last_points is not None:
                                        reward = points - last_points
                                    else:
                                        reward = 0
                                last_points = points

                                if is_final:
                                    outcome = home_next_obs.reward

                                    o = home_next_obs.observation
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

                                    if outcome == 1:
                                        pass
                                    elif outcome == 0:
                                        if killed_points > WIN_THRESHOLD:
                                            outcome = 1
                                        # elif killed_points > 1000 and killed_points < WIN_THRESHOLD:
                                        #     outcome = 0
                                        # else:
                                        #     outcome = -1
                                    else:
                                        outcome = -1

                                    if not USE_DEFINED_REWARD_AS_REWARD:
                                        reward = outcome

                                    food_used_list.append(food_used)
                                    army_count_list.append(army_count)
                                    collected_points_list.append(collected_points)
                                    used_points_list.append(used_points)
                                    killed_points_list.append(killed_points)
                                    steps_list.append(game_loop)

                                    results[outcome + 1] += 1
                                    print("agent_{:d} get final reward".format(self.idx), reward) if 1 else None
                                    print("agent_{:d} get outcome".format(self.idx), outcome) if 1 else None

                                    final_points = killed_points * REWARD_SCALE
                                    #self.writer.add_scalar('final_points/' + 'agent_' + str(self.idx), final_points, total_episodes)
                                    #self.coordinator.send_episode_points(self.idx, total_episodes, final_points)

                                    final_outcome = outcome
                                    #self.writer.add_scalar('final_outcome/' + 'agent_' + str(self.idx), final_outcome, total_episodes)
                                    #self.coordinator.send_episode_outcome(self.idx, total_episodes, final_outcome)

                                    is_final_trajectory = True
                                    if outcome == 1:
                                        is_win_trajectory = True 
                                else:
                                    print("agent_{:d} get reward".format(self.idx), reward) if debug else None

                                # note, original AlphaStar pseudo-code has some mistakes, we modified 
                                # them here

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

                                if self.is_training:
                                    print('is_final_trajectory', is_final_trajectory) if debug else None
                                    trajectory.append(traj_step)

                                #player_memory = tuple(h.detach().clone() for h in player_new_memory)
                                player_memory = player_new_memory
                                home_obs = home_next_obs
                                last_delay = expected_delay
                                last_action_type = player_action.action_type.item()
                                last_repeat_queued = player_action.queue.item()
                                last_list = [last_delay, last_action_type, last_repeat_queued]

                                del player_action, player_new_memory

                                if self.is_training and len(trajectory) >= AHP.sequence_length:                    
                                    trajectories = RU.stack_namedtuple(trajectory)

                                    print("Learner send_trajectory!") if debug else None
                                    # with self.buffer_lock:
                                    self.trajectories.append(trajectory)

                                    if is_final_trajectory:
                                        self.final_trajectories.append(trajectory)

                                    if is_win_trajectory:
                                        self.win_trajectories.append(trajectory)    

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

                            # use max_frames_per_episode to end the episode
                            if self.max_episodes and total_episodes >= self.max_episodes:
                                print("Beyond the max_episodes, return!")
                                raise Exception

        except Exception as e:
            # print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())
            pass

        finally:
            print("results: ", results) if debug else None
            print("win rate: ", results[2] / (1e-9 + sum(results))) if debug else None

            total_time = time() - training_start_time
            #print('agent_', self.idx, "total_time: ", total_time / 60.0, "min") if debug else None

            self.is_running = False

    # create env function
    def create_env_one_player(self, player=None, game_steps_per_episode=GAME_STEPS_PER_EPISODE, 
                              step_mul=STEP_MUL, version=VERSION, 
                              map_name=MAP_NAME, random_seed=RANDOM_SEED):

        player_aif = AgentInterfaceFormat(**AAIFP._asdict())
        agent_interface_format = [player_aif]

        sc2_computer = Bot([Race.terran],
                           Difficulty(DIFFICULTY),
                           [BotBuild.random])

        env = SC2Env(map_name=map_name,
                     players=[Agent(Race.protoss, "test1"),
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

    points = float(collected_points + used_points + 2 * killed_points)

    #points = float(killed_points)

    points = points * REWARD_SCALE

    return points


def test(on_server=False, replay_path=None):
    learners = []
    actors = []

    with Manager() as manager:

        trajectories = manager.list()
        final_trajectories = manager.list()
        win_trajectories = manager.list()

        for idx in range(1):

            buffer_lock = Lock()
            learner = LearnerProcess(max_time_for_training=60 * 60 * 24 * 7, lr=LR, is_training=IS_TRAINING, 
                                     use_opponent_state=USE_OPPONENT_STATE,
                                     no_replay_learn=NO_REPLAY_LEARN, num_epochs=NUM_EPOCHS,
                                     count_of_batches=COUNT_OF_BATCHES, buffer_size=BUFFER_SIZE,
                                     use_random_sample=USE_RANDOM_SAMPLE, only_update_baseline=ONLY_UPDATE_BASELINE,
                                     trajectories=trajectories, final_trajectories=final_trajectories,
                                     win_trajectories=win_trajectories)
            learners.append(learner)

            for z in range(ACTOR_NUMS):
                actor = ActorVSComputerProcess(z + 1, 
                                               trajectories, final_trajectories,
                                               win_trajectories)
                actors.append(actor)

        processes = []
        for l in learners:
            l.start()
            processes.append(l.process)
            sleep(1)
        for a in actors:
            a.start()
            processes.append(a.process)
            sleep(1)

        try: 
            # Wait for training to finish.
            for t in processes:
                t.join()

            coordinator.write_eval_results()

        except Exception as e: 
            print("Exception Handled in Main, Detials of the Exception:", e)
