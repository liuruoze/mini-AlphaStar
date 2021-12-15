#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by fighting against built-in AI (computer), using no replays"

from time import time, sleep, strftime, localtime

import os
import random
import traceback
import threading
import datetime

import numpy as np

import torch

from tensorboardX import SummaryWriter

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

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


MAX_EPISODES = 3
IS_TRAINING = True
MAP_NAME = P.map_name  # "Simple64" "AbyssalReef"
STEP_MUL = 8
GAME_STEPS_PER_EPISODE = 9000    # 9000

DIFFICULTY = 1
RANDOM_SEED = 1
VERSION = '3.16.1'

RESTORE = True
SAVE_REPLAY = False
DEFINED_REWARD = True

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
if torch.backends.cudnn.is_available():
    print('cudnn available')
    print('cudnn version', torch.backends.cudnn.version())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# TODO: fix the bug ValueError: The game didn't advance to the expected game loop. Expected: 2512, got: 2507

class ActorVSComputer:
    """A single actor loop that generates trajectories by playing with built-in AI (computer).

    We don't use batched inference here, but it was used in practice.
    """

    def __init__(self, player, coordinator, idx, max_time_for_training=60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 4,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING,
                 replay_dir="./added_simple64_replays/"):
        self.player = player
        self.idx = idx

        print('initialed player')
        self.player.add_actor(self)
        self.player.agent.set_rl_training(is_training)

        #model.load_state_dict(torch.load(model_path, map_location=device), strict=False) 
        if ON_GPU:
            self.player.agent.agent_nn.to(DEVICE)

        self.teacher = get_supervised_agent(player.race, model_type="sl", restore=RESTORE)
        print('initialed teacher')

        if ON_GPU:
            self.teacher.agent_nn.to(DEVICE)

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

        if self.player.learner is not None:
            if DEFINED_REWARD:
                self.writer = self.player.learner.writer

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

            start_time = time()
            print("start_time before training:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)))

            # use max_episodes to end the loop
            while time() - start_time < self.max_time_for_training:
                agents = [self.player]

                with self.create_env_one_player(self.player) as env:

                    # set the obs and action spec
                    observation_spec = env.observation_spec()
                    action_spec = env.action_spec()

                    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                        agent.setup(obs_spec, act_spec)

                    self.teacher.setup(self.player.agent.obs_spec, self.player.agent.action_spec)

                    print('player:', self.player) if debug else None
                    print('opponent:', "Computer bot") if debug else None

                    trajectory = []
                    start_time = time()  # in seconds.
                    print("start_time before reset:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)))

                    # one opponent match (may include several games) defaultly lasts for no more than 2 hour
                    while time() - start_time < self.max_time_per_one_opponent:

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
                        teacher_memory = self.teacher.initial_state()

                        # initial build order
                        player_bo = []

                        episode_frames = 0
                        # default outcome is 0 (means draw)
                        outcome = 0

                        # initial last list
                        last_list = [0, 0, 0]

                        # points for defined reward
                        points, last_points = 0, 0

                        # in one episode (game)
                        start_episode_time = time()  # in seconds.
                        print("start_episode_time before is_final:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_episode_time)))

                        while not is_final:
                            total_frames += 1
                            episode_frames += 1

                            t = time()

                            state = self.player.agent.agent_nn.preprocess_state_all(home_obs.observation, 
                                                                                    build_order=player_bo, 
                                                                                    last_list=last_list)
                            # TODO, implement baseline process in preprocess_state_all_plus_baseline (a new function)
                            baseline_state = self.player.agent.agent_nn.get_scalar_list(home_obs.observation, 
                                                                                        build_order=player_bo)

                            player_step = self.player.agent.step_from_state(state, player_memory)
                            player_function_call, player_action, player_logits, player_new_memory, player_select_units_num = player_step

                            print("player_function_call:", player_function_call) if debug else None
                            print("player_action:", player_action) if debug else None
                            print("player_action.delay:", player_action.delay) if debug else None
                            print("player_select_units_num:", player_select_units_num) if debug else None

                            if False:
                                show_sth(home_obs, player_action)

                            expected_delay = player_action.delay.item()
                            step_mul = max(1, expected_delay)
                            print("step_mul:", step_mul) if debug else None

                            print('run_loop, t1', time() - t) if speed else None
                            t = time()

                            if True:
                                teacher_step = self.teacher.step_based_on_actions(state, teacher_memory, player_action, player_select_units_num)
                                teacher_logits, teacher_select_units_num, teacher_new_memory = teacher_step
                                print("teacher_logits:", teacher_logits) if debug else None
                            else:
                                teacher_logits = None

                            print('run_loop, t2', time() - t) if speed else None
                            t = time()

                            env_actions = [player_function_call]

                            player_action_spec = action_spec[0]
                            action_masks = RU.get_mask(player_action, player_action_spec)
                            z = None

                            print('run_loop, t3', time() - t) if speed else None
                            t = time()

                            timesteps = env.step(env_actions, step_mul=STEP_MUL)  # STEP_MUL step_mul
                            [home_next_obs] = timesteps
                            reward = home_next_obs.reward
                            print("reward: ", reward) if 0 else None

                            print('run_loop, t4', time() - t) if speed else None
                            t = time()

                            is_final = home_next_obs.last()

                            # calculate the build order
                            player_bo = L.calculate_build_order(player_bo, home_obs.observation, home_next_obs.observation)
                            print("player build order:", player_bo) if debug else None

                            print('run_loop, t5', time() - t) if speed else None
                            t = time()

                            # calculate the unit counts of bag
                            player_ucb = L.calculate_unit_counts_bow(home_obs.observation).reshape(-1).numpy().tolist()
                            print("player unit count of bow:", sum(player_ucb)) if debug else None

                            print('run_loop, t6', time() - t) if speed else None
                            t = time()

                            game_loop = home_obs.observation.game_loop[0]
                            print("game_loop", game_loop) if debug else None

                            if DEFINED_REWARD:
                                points = get_points(home_next_obs)

                                if last_points != 0:
                                    reward = points - last_points
                                else:
                                    reward = 0
                                print("{:d} get defined reward".format(self.idx), reward) if debug else None

                                last_points = points

                            # note, original AlphaStar pseudo-code has some mistakes, we modified 
                            # them here
                            traj_step = Trajectory(
                                state=state,
                                baseline_state=baseline_state,
                                baseline_state_op=baseline_state,  # when fighting with computer, we don't use opponent state
                                memory=player_memory,
                                z=z,
                                masks=action_masks,
                                action=player_action,
                                behavior_logits=player_logits,
                                teacher_logits=teacher_logits,      
                                is_final=is_final,                                          
                                reward=reward,

                                build_order=player_bo,
                                z_build_order=player_bo,  # we change it to the sampled build order
                                unit_counts=player_ucb,  # player_ucb,
                                z_unit_counts=player_ucb,  # player_ucb,  # we change it to the sampled unit counts
                                game_loop=game_loop,
                                last_list=last_list,
                            )
                            if self.is_training:
                                trajectory.append(traj_step)

                            player_memory = tuple(h.detach() for h in player_new_memory)
                            home_obs = home_next_obs
                            last_delay = expected_delay
                            last_action_type = player_action.action_type.item()
                            last_repeat_queued = player_action.queue.item()
                            last_list = [last_delay, last_action_type, last_repeat_queued]

                            if is_final:
                                outcome = home_next_obs.reward
                                print("outcome: ", outcome) if debug else None
                                results[outcome + 1] += 1

                                if SAVE_REPLAY:
                                    env.save_replay(self.replay_dir)

                                if DEFINED_REWARD:
                                    final_points = get_points(home_next_obs)

                                    self.writer.add_scalar('agent_' + str(self.idx) + '/defined_reward', final_points, total_episodes)

                            if self.is_training and len(trajectory) >= AHP.sequence_length:                    
                                trajectories = RU.stack_namedtuple(trajectory)

                                if self.player.learner is not None:
                                    if self.player.learner.is_running:

                                        print("Learner send_trajectory!") if debug else None

                                        self.player.learner.send_trajectory(trajectories)
                                        trajectory = []
                                    else:
                                        print("Learner stops!")

                                        print("Actor also stops!")
                                        return

                            # use max_frames to end the loop
                            # whether to stop the run
                            if self.max_frames and total_frames >= self.max_frames:
                                print("Beyond the max_frames, return!")
                                return

                            # use max_frames_per_episode to end the episode
                            if self.max_frames_per_episode and episode_frames >= self.max_frames_per_episode:
                                print("Beyond the max_frames_per_episode, break!")
                                break

                        #self.coordinator.send_outcome(self.player, self.opponent, outcome)

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")
                            return

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            print("results: ", results) if debug else None
            print("win rate: ", results[2] / (1e-9 + sum(results))) if debug else None

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

    points = collected_points + used_points + killed_points

    return points


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

    print('unit_type_list', unit_type_list) if debug else None     


def injected_function_call(home_obs, env, function_call):
    # for testing

    obs = home_obs.observation
    raw_units = obs["raw_units"]

    function = function_call.function
    # note, this "function" is a IntEnum
    # change it to int by "int(function)" or "function.value"
    # or show the string by "function.name"
    func_name = function.name

    [select, target, max_num] = RAMP.SMALL_MAPPING.get(func_name, [None, None, 1])
    print('select, target, max_num', select, target, max_num) if debug else None

    # select, target, min_num = RAMP.select_and_target_unit_type_for_protoss_actions(function_call)
    # print('select, target, min_num', select, target, min_num) if debug else None

    select_candidate = []
    target_candidate = []

    nexus_u = None

    for u in raw_units:
        if u.alliance == 1:
            if u.unit_type == 59:  # Nexus
                nexus_u = u
                print('nexus_u', nexus_u.x, nexus_u.y) if debug else None
        if select is not None:
            if not isinstance(select, list):
                select = [select]
            if u.alliance == 1:
                if u.unit_type in select:
                    select_candidate.append(u.tag)

        if target is not None:
            if not isinstance(target, list):
                target = [target]
            if u.unit_type in target:
                if u.display_type == 1:  # visible
                    target_candidate.append(u.tag)

    unit_tags = []
    if len(select_candidate) > 0:
        print('select_candidate', select_candidate)
        if max_num == 1:
            unit_tags = [random.choice(select_candidate)]
        elif max_num > 1:
            unit_tags = select_candidate[0:max_num - 1]

    if len(target_candidate) > 0:   
        print('target_candidate', target_candidate) 
        target_tag = random.choice(target_candidate)

    sc2_action = env._features[0].transform_action(obs, function_call)  
    print("sc2_action before transformed:", sc2_action) if debug else None

    if sc2_action.HasField("action_raw"):
        raw_act = sc2_action.action_raw
        if raw_act.HasField("unit_command"):
            uc = raw_act.unit_command
            # to judge a repteated field whether has 
            # use the following way
            if len(uc.unit_tags) != 0:
                # can not assign, must use unit_tags[:]=[xx tag]
                uc.unit_tags[:] = unit_tags

            # we use fixed target unit tag only for Harvest_Gather_unit action
            if uc.HasField("target_unit_tag"):
                if len(target_candidate) > 0:    
                    uc.target_unit_tag = target_tag

            if uc.HasField("target_world_space_pos"):
                twsp = uc.target_world_space_pos
                rand_x = random.randint(-10, 10)
                rand_y = random.randint(-10, 10)

                if func_name != "Attack_pt":  # build buildings
                    print('nexus_u', nexus_u.x, nexus_u.y) if debug else None
                    # these value are considered in minimap unit
                    twsp.x = (35 + rand_x * 1)
                    twsp.y = (55 + rand_y * 1)  
                    # AbysaalReef is [152 x 136]
                else:                        # attack point
                    twsp.x = 50
                    twsp.y = 22                 

    print("sc2_action after transformed:", sc2_action) if debug else None

    return sc2_action


def test(on_server=False, replay_path=None):
    # model path
    MODEL_TYPE = "sl"
    MODEL_PATH = "./model/"
    ACTOR_NUMS = 1

    league = League(
        initial_agents={
            race: get_supervised_agent(race, path=MODEL_PATH, model_type=MODEL_TYPE, restore=RESTORE)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=0,
        league_exploiters=0)

    coordinator = Coordinator(league)
    learners = []
    actors = []

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = Learner(player, max_time_for_training=60 * 60 * 24, is_training=IS_TRAINING)
        learners.append(learner)
        actors.extend([ActorVSComputer(player, coordinator, (idx + 1)) for _ in range(ACTOR_NUMS)])

    threads = []
    for l in learners:
        l.start()
        threads.append(l.thread)
        sleep(1)
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
