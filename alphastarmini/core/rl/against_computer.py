#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the actor (only play against the inner computer) in the actor-learner mode in the IMPALA architecture "

# modified from AlphaStar pseudo-code
import traceback
from time import time, sleep, strftime, localtime
import threading

import os

USED_DEVICES = "0"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.optim import Adam


from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent, Race, Bot, Difficulty, BotBuild

from alphastarmini.core.rl.env_utils import SC2Environment, get_env_outcome
from alphastarmini.core.rl.utils import Trajectory, get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl import utils as U

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

# below packages are for test
from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Training_Races as TR
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

__author__ = "Ruo-Ze Liu"

debug = False

STEP_MUL = 8
GAME_STEPS_PER_EPISODE = 12000    # 9000
MAX_EPISODES = 25      # 100   
MAIN_PLAYER_NUMS = 1
ACTOR_NUMS = 2
IS_TRAINING = True


class ActorLoopVersusComputer:
    """A single actor loop that generates trajectories by playing with built-in AI (computer).

    We don't use batched inference here, but it was used in practice.
    TODO: implement the batched version
    """

    def __init__(self, player, coordinator, max_time_for_training = 60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 4,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, is_training=IS_TRAINING):

        self.player = player
        self.player.add_actor(self)

        # below code is not used because we only can create the env when we know the opponnet information (e.g., race)
        # AlphaStar: self.environment = SC2Environment()

        self.teacher = get_supervised_agent(player.race, model_type="sl")

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

            # use max_episodes to end the loop
            while time() - start_time < self.max_time_for_training:
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
                    start_time = time()  # in seconds.
                    print("start_time before reset:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)))

                    # one opponent match (may include several games) defaultly lasts for no more than 2 hour
                    while time() - start_time < self.max_time_per_one_opponent:

                        # Note: the pysc2 environment don't return z
                        # TODO: check it

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

                            teacher_logits = player_logits

                            env_actions = [player_function_call]

                            player_action_spec = action_spec[0]
                            action_masks = U.get_mask(player_action, player_action_spec)
                            z = None

                            timesteps = env.step(env_actions)
                            [home_next_obs] = timesteps
                            reward = home_next_obs.reward
                            print("reward: ", reward) if 0 else None

                            is_final = home_next_obs.last()

                            # note, original AlphaStar pseudo-code has some mistakes, we modified 
                            # them here
                            traj_step = Trajectory(
                                observation=home_obs.observation,
                                opponent_observation=home_obs.observation,
                                memory=player_memory,
                                z=z,
                                masks=action_masks,
                                action=player_action,
                                behavior_logits=player_logits,
                                teacher_logits=teacher_logits,      
                                is_final=is_final,                                          
                                reward=reward,
                            )
                            trajectory.append(traj_step)

                            player_memory = tuple(h.detach() for h in player_new_memory)

                            home_obs = home_next_obs

                            if is_final:
                                outcome = reward
                                print("outcome: ", outcome) if 1 else None
                                results[outcome + 1] += 1

                            if len(trajectory) >= AHP.sequence_length:                    
                                trajectories = U.stack_namedtuple(trajectory)

                                if self.player.learner is not None:
                                    if self.player.learner.is_running:
                                        if self.is_training:
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
                            print("results: ", results) if 1 else None
                            print("win rate: ", results[2] / (1e-8 + sum(results))) if 1 else None
                            return

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            self.is_running = False

    # create env function
    def create_env_one_player(self, player, game_steps_per_episode=GAME_STEPS_PER_EPISODE, 
                              step_mul=STEP_MUL, version=None, 
                              map_name="Simple64", random_seed=1):

        player_aif = AgentInterfaceFormat(**AAIFP._asdict())
        agent_interface_format = [player_aif]

        # create env
        print('map name:', map_name) 
        print('player.name:', player.name)
        print('player.race:', player.race)

        sc2_computer = Bot([Race.terran],
                           Difficulty.very_hard,
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


def test(on_server=False):
    league = League(
        initial_agents={
            race: get_supervised_agent(race, model_type="rl")
            for race in [Race.protoss]
        },
        main_players=MAIN_PLAYER_NUMS, 
        main_exploiters=0,
        league_exploiters=0)

    coordinator = Coordinator(league)
    learners = []
    actors = []

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = Learner(player, max_time_for_training=60 * 60 * 24)
        learners.append(learner)
        actors.extend([ActorLoopVersusComputer(player, coordinator) for _ in range(ACTOR_NUMS)])

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
