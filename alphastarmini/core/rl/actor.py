#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the actor (loop) in the actor-learner mode in the IMPALA architecture "

# modified from AlphaStar pseudo-code
import traceback
from time import time, sleep, strftime, localtime
import threading

import torch
from torch.optim import Adam

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

__author__ = "Ruo-Ze Liu"

debug = False

STEP_MUL = 8   # 1
GAME_STEPS_PER_EPISODE = 1800    # 9000
MAX_EPISODES = 5      # 100   
MAIN_PLAYER_NUMS = 1


class ActorLoop:
    """A single actor loop that generates trajectories.

    We don't use batched inference here, but it was used in practice.
    Note: this code should check the teacher implemention
    """

    def __init__(self, player, coordinator, max_time_for_training = 60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 2,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES):

        self.player = player
        self.player.add_actor(self)

        self.teacher = get_supervised_agent(player.race, model_type="sl")

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

                with self.create_env(self.player, self.opponent) as env:

                    # set the obs and action spec
                    observation_spec = env.observation_spec()
                    action_spec = env.action_spec()

                    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                        agent.setup(obs_spec, act_spec)

                    print('player:', self.player) if debug else None
                    print('opponent:', self.opponent) if debug else None

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

                            # run_loop: actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
                            player_step = self.player.agent.step_logits(home_obs, player_memory)
                            player_function_call, player_action, player_logits, player_new_memory = player_step

                            print("player_function_call:", player_function_call) if 0 else None

                            opponent_step = self.opponent.agent.step_logits(away_obs, opponent_memory)
                            opponent_function_call, opponent_action, opponent_logits, opponent_new_memory = opponent_step

                            # Q: how to do it ?
                            # teacher_logits = self.teacher(home_obs, player_action, teacher_memory)
                            # We should add the right implemention of teacher_logits, see actor_plus_z.py
                            teacher_logits = player_logits

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
                            print("player unit count of bow:", player_ucb) if debug else None

                            # note, original AlphaStar pseudo-code has some mistakes, we modified 
                            # them here
                            traj_step = Trajectory(
                                observation=home_obs.observation,
                                opponent_observation=away_obs.observation,
                                memory=player_memory,
                                z=z,
                                masks=action_masks,
                                action=player_action,
                                behavior_logits=player_logits,
                                teacher_logits=teacher_logits,      
                                is_final=is_final,                                          
                                reward=reward,
                                build_order=player_bo,
                                z_build_order=player_bo,  # change it to the sampled build order
                                unit_counts=player_ucb,
                                z_unit_counts=player_ucb,  # change it to the sampled unit counts
                            )
                            trajectory.append(traj_step)

                            player_memory = tuple(h.detach() for h in player_new_memory)
                            opponent_memory = tuple(h.detach() for h in opponent_new_memory)

                            home_obs = home_next_obs
                            away_obs = away_next_obs

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

                        self.coordinator.send_outcome(self.player, self.opponent, outcome)

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")
                            print("results: ", results) if debug else None
                            print("win rate: ", results[2] / (1e-8 + sum(results))) if debug else None
                            return

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            self.is_running = False

    # create env function
    def create_env(self, player, opponent, game_steps_per_episode=GAME_STEPS_PER_EPISODE, 
                   step_mul=STEP_MUL, version=None, 
                   map_name="Simple64", random_seed=1):

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
