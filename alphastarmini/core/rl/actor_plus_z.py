#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the actor using Z (the reward for imititing the expert in replays) in the actor-learner mode in the IMPALA architecture "

# modified from AlphaStar pseudo-code
import traceback
from time import time, sleep, strftime, localtime
import threading

import torch
from torch.optim import Adam

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as com_pb

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent, Race

from alphastarmini.core.rl.rl_utils import Trajectory, get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl import rl_utils as U

from alphastarmini.lib import utils as L

# below packages are for test
from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Training_Races as TR
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

# for replay reward
import os
import random
from pysc2.lib import point
from pysc2.lib import features as F
from pysc2 import run_configs


__author__ = "Ruo-Ze Liu"

debug = False

STEP_MUL = 8   # 1
GAME_STEPS_PER_EPISODE = 18000    # 9000
MAX_EPISODES = 1000      # 100   

RANDOM_SEED = 2
VERSION = '4.10.0'
REPLAY_VERIOSN = '3.16.1'
REPLAY_PATH = "data/Replays/filtered_replays_1/"

RESTORE = False

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
if torch.backends.cudnn.is_available():
    print('cudnn available')
    print('cudnn version', torch.backends.cudnn.version())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


class ActorLoopPlusZ:
    """A single actor loop that generates trajectories.

    We don't use batched inference here, but it was used in practice.
    TODO: implement the batched version
    """

    def __init__(self, player, coordinator, max_time_for_training = 60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 2,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, use_replay_expert_reward=True,
                 replay_path=REPLAY_PATH, replay_version=REPLAY_VERIOSN):

        self.player = player
        self.player.add_actor(self)
        if ON_GPU:
            self.player.agent.agent_nn.to(DEVICE)

        self.teacher = get_supervised_agent(player.race, model_type="sl", restore=RESTORE)
        if ON_GPU:
            self.teacher.agent_nn.to(DEVICE)

        # below code is not used because we only can create the env when we know the opponnet information (e.g., race)
        # AlphaStar: self.environment = SC2Environment()

        self.coordinator = coordinator
        self.max_time_for_training = max_time_for_training
        self.max_time_per_one_opponent = max_time_per_one_opponent
        self.max_frames_per_episode = max_frames_per_episode
        self.max_frames = max_frames
        self.max_episodes = max_episodes

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True                            # Daemonize thread

        self.is_running = True
        self.is_start = False

        self.use_replay_expert_reward = use_replay_expert_reward
        self.replay_path = replay_path
        self.replay_version = replay_version

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
            results = [0, 0, 0]

            start_time = time()
            print("start_time before training:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)))

            while time() - start_time < self.max_time_for_training:
                self.opponent, _ = self.player.get_match()
                agents = [self.player, self.opponent]

                # if self.use_replay_expert_reward:
                run_config = run_configs.get(version=self.replay_version)  # the replays released by blizzard are all 3.16.1 version

                with self.create_env(self.player, self.opponent) as env:

                    # set the obs and action spec
                    observation_spec = env.observation_spec()
                    action_spec = env.action_spec()

                    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                        agent.setup(obs_spec, act_spec)

                    self.teacher.setup(self.player.agent.obs_spec, self.player.agent.action_spec)

                    print('player:', self.player) if debug else None
                    print('opponent:', self.opponent) if debug else None
                    print('teacher:', self.teacher) if debug else None

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

                        # check the condition that the replay is over but the game is not
                        with run_config.start(full_screen=False) as controller:
                            # here we must use the with ... as ... statement, or it will cause an error
                            #controller = run_config.start(full_screen=False)

                            # start replay reward
                            raw_affects_selection = False
                            raw_crop_to_playable_area = False
                            screen_resolution = point.Point(64, 64)
                            minimap_resolution = point.Point(64, 64)
                            camera_width = 24

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

                            replay_files = os.listdir(self.replay_path)

                            # random select a replay file from the candidate replays
                            random.shuffle(replay_files)

                            replay_path = self.replay_path + replay_files[0]
                            print('replay_path:', replay_path)
                            replay_data = run_config.replay_data(replay_path)
                            replay_info = controller.replay_info(replay_data)
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

                            win_observe_id = 0

                            for i, result in enumerate(observe_result_list):
                                if result == sc_pb.Victory:
                                    win_observe_id = observe_id_list[i]
                                    break

                            start_replay = sc_pb.RequestStartReplay(
                                replay_data=replay_data,
                                options=interface,
                                disable_fog=False,  # FLAGS.disable_fog
                                observed_player_id=win_observe_id,  # FLAGS.observed_player
                                map_data=None,
                                realtime=False
                            )

                            controller.start_replay(start_replay)
                            feat = F.features_from_game_info(game_info=controller.game_info(), 
                                                             raw_resolution=AAIFP.raw_resolution, 
                                                             hide_specific_actions=AAIFP.hide_specific_actions,
                                                             use_feature_units=True, use_raw_units=True,
                                                             use_unit_counts=True, use_raw_actions=True,
                                                             show_cloaked=True, show_burrowed_shadows=True, 
                                                             show_placeholders=True) 
                            replay_obs = None
                            replay_bo = []

                            replay_o = controller.observe()
                            replay_obs = feat.transform_obs(replay_o)
                            # end replay reward

                            [home_obs, away_obs] = timesteps
                            is_final = home_obs.last()

                            player_memory = self.player.agent.initial_state()
                            opponent_memory = self.opponent.agent.initial_state()
                            teacher_memory = self.teacher.initial_state()

                            # initial build order
                            player_bo = []

                            episode_frames = 0
                            # default outcome is 0 (means draw)
                            outcome = 0

                            # in one episode (game)
                            # 
                            start_episode_time = time()  # in seconds.
                            print("start_episode_time before is_final:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_episode_time)))

                            while not is_final:
                                total_frames += 1
                                episode_frames += 1

                                state = self.player.agent.agent_nn.preprocess_state_all(home_obs.observation, build_order=player_bo)
                                state_op = self.player.agent.agent_nn.preprocess_state_all(away_obs.observation)

                                # baseline_state = self.player.agent.agent_nn.get_scalar_list(home_obs.observation, build_order=player_bo)
                                # baseline_state_op = self.player.agent.agent_nn.get_scalar_list(away_obs.observation)

                                baseline_state = self.player.agent.agent_nn.get_baseline_state_from_multi_source_state(state)
                                baseline_state_op = self.player.agent.agent_nn.get_baseline_state_from_multi_source_state(state_op)

                                player_step = self.player.agent.step_from_state(state, player_memory)
                                player_function_call, player_action, player_logits, player_new_memory = player_step
                                print("player_function_call:", player_function_call) if debug else None

                                opponent_step = self.opponent.agent.step_from_state(state_op, opponent_memory)
                                opponent_function_call, opponent_action, opponent_logits, opponent_new_memory = opponent_step

                                # Q: how to do it ?
                                # teacher_logits = self.teacher(home_obs, player_action, teacher_memory)
                                # may change implemention of teacher_logits
                                teacher_step = self.teacher.step_from_state(state, teacher_memory)
                                teacher_function_call, teacher_action, teacher_logits, teacher_new_memory = teacher_step
                                print("teacher_function_call:", teacher_function_call) if debug else None

                                env_actions = [player_function_call, opponent_function_call]

                                player_action_spec = action_spec[0]
                                action_masks = U.get_mask(player_action, player_action_spec)
                                z = None

                                timesteps = env.step(env_actions)
                                [home_next_obs, away_next_obs] = timesteps

                                # print the observation of the agent
                                # print("home_obs.observation:", home_obs.observation)

                                reward = home_next_obs.reward
                                print("reward: ", reward) if debug else None
                                is_final = home_next_obs.last()

                                # calculate the build order
                                player_bo = L.calculate_build_order(player_bo, home_obs.observation, home_next_obs.observation)
                                print("player build order:", player_bo) if debug else None

                                # calculate the unit counts of bag
                                player_ucb = L.calculate_unit_counts_bow(home_obs.observation).reshape(-1).numpy().tolist()
                                print("player unit count of bow:", sum(player_ucb)) if debug else None

                                # start replay_reward
                                # note the controller should step the same steps as with the rl actor (keep the time as the same)
                                controller.step(STEP_MUL)

                                replay_next_o = controller.observe()
                                replay_next_obs = feat.transform_obs(replay_next_o)

                                # calculate the build order for replay
                                replay_bo = L.calculate_build_order(replay_bo, replay_obs, replay_next_obs)
                                print("replay build order:", player_bo) if debug else None

                                # calculate the unit counts of bag for replay
                                replay_ucb = L.calculate_unit_counts_bow(replay_obs).reshape(-1).numpy().tolist()
                                print("replay unit count of bow:", sum(replay_ucb)) if debug else None            
                                # end replay_reward

                                game_loop = home_obs.observation.game_loop[0]
                                print("game_loop", game_loop) if debug else None 

                                # note, original AlphaStar pseudo-code has some mistakes, we modified 
                                # them here
                                traj_step = Trajectory(
                                    state=state,
                                    baseline_state=baseline_state,
                                    baseline_state_op=baseline_state_op,  
                                    memory=player_memory,
                                    z=z,
                                    masks=action_masks,
                                    action=player_action,
                                    behavior_logits=player_logits,
                                    teacher_logits=teacher_logits,      
                                    is_final=is_final,                                          
                                    reward=reward,
                                    build_order=player_bo,
                                    z_build_order=replay_bo,  # we change it to the sampled build order
                                    unit_counts=player_ucb,
                                    z_unit_counts=replay_ucb,  # we change it to the sampled unit counts
                                    game_loop=game_loop,
                                )
                                trajectory.append(traj_step)

                                player_memory = tuple(h.detach() for h in player_new_memory)
                                opponent_memory = tuple(h.detach() for h in opponent_new_memory)

                                teacher_memory = tuple(h.detach() for h in teacher_new_memory)

                                home_obs = home_next_obs
                                away_obs = away_next_obs

                                # for replay reward
                                replay_obs = replay_next_obs
                                replay_o = replay_next_o

                                if is_final:
                                    outcome = reward
                                    print("outcome: ", outcome) if debug else None
                                    results[outcome + 1] += 1

                                if len(trajectory) >= AHP.sequence_length:                    
                                    trajectories = U.stack_namedtuple(trajectory)

                                    if self.player.learner is not None:
                                        if self.player.learner.is_running:
                                            print("Learner send_trajectory!")
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

                                # end of replay
                                if replay_o.player_result:  
                                    print(replay_o.player_result)
                                    break

                            self.coordinator.send_outcome(self.player, self.opponent, outcome)

                            # use max_frames_per_episode to end the episode
                            if self.max_episodes and total_episodes >= self.max_episodes:
                                print("Beyond the max_episodes, return!")
                                print("results: ", results) if debug else None
                                print("win rate: ", results[2] / (1e-8 + sum(results))) if debug else None
                                return

                    # close the replays

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            self.is_running = False

    # create env function
    def create_env(self, player, opponent, game_steps_per_episode=GAME_STEPS_PER_EPISODE, 
                   step_mul=STEP_MUL, version=VERSION, 
                   # the map should be the same as in the expert replay
                   map_name="AbyssalReef", random_seed=RANDOM_SEED):

        player_aif = AgentInterfaceFormat(**AAIFP._asdict())
        opponent_aif = AgentInterfaceFormat(**AAIFP._asdict())
        agent_interface_format = [player_aif, opponent_aif]

        # create env
        print('map name:', map_name) 
        print('player.name:', player.name)
        print('opponent.name:', opponent.name)
        print('player.race:', player.race)
        print('opponent.race:', opponent.race)

        env = SC2Env(map_name=map_name,
                     players=[Agent(player.race, player.name),
                              Agent(opponent.race, opponent.name)],
                     step_mul=step_mul,
                     game_steps_per_episode=game_steps_per_episode,
                     agent_interface_format=agent_interface_format,
                     version=version,
                     random_seed=random_seed)

        return env
