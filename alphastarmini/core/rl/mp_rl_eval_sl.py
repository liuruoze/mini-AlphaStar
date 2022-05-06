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
    raise NotImplementedError

SAVE_STATISTIC = True
SAVE_REPLAY = True

# 24000
GAME_STEPS_PER_EPISODE = 18000  
MAX_EPISODES = 5
ACTOR_NUMS = 2
STEP_MUL = 16
DIFFICULTY = 1

# model path
MODEL_TYPE = "sl"
MODEL_PATH = "./model/"

IS_TRAINING = False
MAP_NAME = SCHP.map_name  # P.map_name "Simple64" "AbyssalReef"
USE_PREDICT_STEP_MUL = AHP.use_predict_step_mul
WIN_THRESHOLD = 4000


RANDOM_SEED = 1
VERSION = SCHP.game_version

RESTORE = True
OUTPUT_FILE = './outputs/mp_eval_sl.txt'

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

    def __init__(self, player, coordinator, idx, max_time_for_training=60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 4,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING,
                 replay_dir="./added_simple64_replays/"):
        self.player = player

        print('initialed player')
        self.player.add_actor(self)
        self.player.agent.set_rl_training(is_training)

        #model.load_state_dict(torch.load(model_path, map_location=device), strict=False) 
        if ON_GPU:
            self.player.agent.agent_nn.to(DEVICE)

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

        self.idx = idx

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
                                player_step = self.player.agent.step_from_state(state, player_memory, obs=home_obs.observation)

                                player_function_call, player_action, player_logits, \
                                    player_new_memory, player_select_units_num, entity_num = self.player.agent.step_from_state(state, 
                                                                                                                               player_memory, 
                                                                                                                               obs=home_obs.observation)

                            print("player_function_call:", player_function_call) if not SAVE_STATISTIC else None
                            print("player_action:", player_action) if debug else None
                            print("player_action.delay:", player_action.delay) if debug else None
                            print("player_select_units_num:", player_select_units_num) if debug else None

                            expected_delay = player_action.delay.item()
                            step_mul = max(1, expected_delay)
                            print("step_mul:", step_mul) if debug else None

                            env_actions = [player_function_call]

                            if USE_PREDICT_STEP_MUL:
                                timesteps = env.step(env_actions, step_mul=step_mul)  # STEP_MUL step_mul
                            else:
                                timesteps = env.step(env_actions, step_mul=STEP_MUL)

                            [home_next_obs] = timesteps
                            reward = home_next_obs.reward
                            print("reward: ", reward) if debug else None

                            is_final = home_next_obs.last()

                            # calculate the build order
                            player_bo = L.calculate_build_order(player_bo, home_obs.observation, home_next_obs.observation)
                            print("player build order:", player_bo) if debug else None

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

                                    if killed_points > WIN_THRESHOLD:
                                        outcome = 1

                                    food_used_list.append(food_used)
                                    army_count_list.append(army_count)
                                    collected_points_list.append(collected_points)
                                    used_points_list.append(used_points)
                                    killed_points_list.append(killed_points)
                                    steps_list.append(game_loop)

                                    end_episode_time = time()  # in seconds.
                                    end_episode_time = strftime("%Y-%m-%d %H:%M:%S", localtime(end_episode_time))

                                    statistic = 'Agent ID: {} | Bot Difficulty: {} | Episode: [{}/{}] | food_used: {:.1f} | army_count: {:.1f} | collected_points: {:.1f} | used_points: {:.1f} | killed_points: {:.1f} | steps: {:.3f}s \n'.format(
                                        self.idx, DIFFICULTY, total_episodes, MAX_EPISODES, food_used, army_count, collected_points, used_points, killed_points, game_loop)

                                    statistic = end_episode_time + " " + statistic

                                    with open(OUTPUT_FILE, 'a') as file:
                                        file.write(statistic)

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

                        self.coordinator.only_send_outcome(self.player, outcome)

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")
                            raise Exception

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            print("results: ", results) if debug else None
            win_rate = results[2] / (1e-9 + sum(results))
            print("win rate: ", win_rate) if debug else None

            total_time = time() - training_start_time

            if SAVE_STATISTIC: 
                self.coordinator.send_eval_results(self.player, DIFFICULTY, food_used_list, army_count_list, collected_points_list, 
                                                   used_points_list, killed_points_list, steps_list, total_time)

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

    rank = 0

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = None  # Learner(player, rank, v_steps, device, max_time_for_training=60 * 60 * 24, is_training=IS_TRAINING)
        learners.append(learner)
        actors.extend([ActorEval(player, coordinator, j + 1) for j in range(ACTOR_NUMS)])

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

        coordinator.write_eval_results()

    except Exception as e: 
        print("Exception Handled in Main, Detials of the Exception:", e)
