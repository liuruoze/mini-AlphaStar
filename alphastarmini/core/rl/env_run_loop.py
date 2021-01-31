#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the sc2 run_loop, maybe add some functions for the original pysc2 version"

# modified from pysc2 code

import time

__author__ = "Ruo-Ze Liu"

debug = False


def run_loop(agents, env, max_frames=0, max_episodes=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    # max frames for one episode
    # e.g., we don't want a game lasts for more than one hour
    # note that one second equal to 22.4 frames in real time mode
    max_frames_for_one_episode = 0  # 60 * 60 * 22.4

    # set the obs and action spec
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)

    try:
        # if max_episodes=0, we run the game forever
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1

            # reset the environment
            # timesteps are actually obs, each for the each agent
            timesteps = env.reset()

            # also reset the agents
            for a in agents:
                a.reset()

            game_frames = 0
            while True:
                total_frames += 1
                game_frames += 1

                # for each agent and each timestep, the agent.step(timestep) function
                # actually map the timestep(obs) to action
                # due to we have two agents, each timestep maps to each action for each agent
                # the actions combined as a list to pass to the env
                # env.step(actions) will calculate the next timesteps, each for different agent
                actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]

                # if max_frames=0, we run the game forever
                # thus, we can use max_frames or max_episodes to control how long we want the agent run
                if max_frames and total_frames >= max_frames:
                    return

                # if the game ends, the episode ends
                if timesteps[0].last():
                    break

                # if beyond the max_frames_for_one_episode, this episode ends
                if max_frames_for_one_episode and game_frames >= max_frames_for_one_episode:
                    break

                # else, get the next timesteps, each for different agent
                timesteps = env.step(actions)

    except KeyboardInterrupt:
        pass

    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
