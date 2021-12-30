#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Computing the pseudo-reward (based on distance between the agent and a human player)."

# code for computing the pseudo-reward

import time
import random

import numpy as np

import torch

import Levenshtein

from alphastarmini.core.rl import rl_utils as U
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
import alphastarmini.lib.edit_distance as ED


__author__ = "Ruo-Ze Liu"

debug = False


def list2str(l):
    # note: the maximus accept number for chr() is 1114111, else raise a ValueError
    return ''.join([chr(int(i)) for i in l])


def reward_by_build_order(bo, z_bo, gl=0):
    str1 = list2str(bo)
    str2 = list2str(z_bo)

    dist = Levenshtein.distance(str1, str2)
    print('dist:', dist) if debug else None

    # the cost is the squared distance between the 
    # true built entity and the agent's entity
    reward = dist * dist 

    # rescaled to be within [0, 0.8] with the maximum of 0.8 when it is more than two gateways away.
    # Question: how to do it ?
    reward = min(reward, 50) / 50.0 * 0.8
    reward = -reward
    print('reward:', reward) if debug else None

    scale = time_decay_scale(game_loop=gl)
    # note, build_order seems not to use time decay scale

    return reward


def reward_by_unit_counts(ucb, z_ucb, gl=0):
    str1 = list2str(ucb)
    str2 = list2str(z_ucb)

    dist = Levenshtein.hamming(str1, str2)
    print('dist:', dist) if debug else None

    reward = - dist
    print('reward:', reward) if debug else None

    scale = time_decay_scale(game_loop=gl)
    print('scale:', scale) if debug else None

    # hamming distance use time decay scale
    return reward * scale


def time_decay_scale(game_loop):

    time_seconds = int(game_loop / 22.4)
    time_minutes = int(time_seconds / 60)

    scale = 1.
    if time_minutes > 24:
        scale = 0.
    elif time_minutes > 16:
        scale = 0.25
    elif time_minutes > 8:
        scale = 0.5

    return scale


def compute_pseudoreward(trajectories, reward_name, device):
    """Computes the relevant pseudoreward from trajectories.

    See Methods and detailed_architecture.txt for details.

    Paper description:
      These pseudo-rewards measure the edit distance between sampled 
      and executed build orders, and the Hamming distance 
      between sampled and executed cumulative statistics.
    """

    # The build order reward is the negative Levenshtein distance between 
    # the true (human replay) build order and the agent's build order, except 
    # that in the case $lev_{a, b}(i - 1, j - 1) + 1_{\(a != b\)}$, 
    # instead of $1_{\(a != b\)}$ in the case where the types of the units 
    # are the same, the cost is the squared distance between the 
    # true built entity and the agent's entity, rescaled to be within [0, 0.8] 
    # with the maximum of 0.8 when it is more than two gateways away. A reward 
    # is given whenever the agent's build order changes (because 
    # it has built something new). Units that aren't built (like 
    # auto turrets or larva), worker units, and supply buildings are 
    # skipped in the build order. 

    ''' 
    my notes: Actually, there are so many ambiguous descriptions in this place, e.g., what 
    are the "two gateways away"? and why the units aren't built includes "auto turrets" (I think 
    they are built by players). Moreover, is it the supply buildings that mean the building 
    for the supplement, so, do they include "overlords" or "pylons"? The except case in the 
    calculation of Levenshtein distance is also unclear.

    '''

    # The built units reward is the Hamming distance between the entities built 
    # in some human replay and the entities the agent has built. After 8 minutes, 
    # the reward is multiplied by 0.5. After 16 minutes, the reward is multiplied 
    # by an additional 0.5. After 24 minutes, there are no more rewards.

    ''' 
    my notes: First, it seems here don't use the build order, actually, it should be seen that 
    it use the bag-of-words here. One question: is the reward be calculated in every step? Another 
    question: if the reward is calculated in each step, does it means the agent accept a negative (I 
    think the reward will be the negatived distance) reward every time until the agent builds the 
    same unit counts as the human player (which is a very strong assumption, meaning that we want 
    the agent do nearly the same as the human player). However, we can see that the restriction is 
    smaller as the time passed (e.g, after 8 minutes, the reward is halved). However, is the choice 
    of the "8" too experiential? I think here brings too much human knowledge here..

    '''

    print("reward name:", reward_name) if debug else None

    # add to use reward_name to judge to use the win loss reward
    if reward_name == 'winloss_baseline':
        print('trajectories.reward', trajectories.reward) if debug else None

        rewards_tensor = torch.tensor(trajectories.reward, dtype=torch.float32, device=device)
        return rewards_tensor

    # if we don't use replays to generate reward, just return the result reward
    # if trajectories.build_order[0][0] is None and trajectories.z_build_order[0][0] is None:
    #     rewards_tensor = torch.tensor(trajectories.reward, dtype=torch.float32, device=device)
    #     return rewards_tensor

    print("trajectories.build_order", trajectories.build_order) if debug else None
    print("trajectories.z_build_order", trajectories.z_build_order) if debug else None
    print("trajectories.unit_counts", trajectories.unit_counts) if debug else None
    print("trajectories.z_unit_counts", trajectories.z_unit_counts) if debug else None
    print("trajectories.game_loop", trajectories.game_loop) if debug else None

    # mAS: note we use scale to make the reward of leven and hamming not to much
    scale_leven = 0.05
    scale_hamming = 0.05

    weight_leven = 1.0 * scale_leven
    weight_hamming = 1.0 * scale_hamming

    # in order to distinguish between build_order and built_units 
    if reward_name == 'build_order_baseline':
        weight_hamming = 0.0
    if reward_name == 'built_units_baseline':
        weight_leven = 0.0    

    # AlphaStar: The built units reward is the Hamming distance between the effects in some human replay 
    # and the effects the agent has created. After 8 minutes, the reward is multiplied by 0.5. 
    # After 16 minutes, the reward is multiplied by an additional 0.5. After 24 minutes, there are no more rewards.
    if reward_name == 'effects_baseline':
        # now we don't separate the effects from the unit, 
        # so we ignore this reward now
        weight_hamming = 0.0
        weight_leven = 0.0

    # AlphaStar: The built units reward is the Hamming distance between the upgrades in some human replay and the upgrades 
    # the agent has researched. After 8 minutes, the reward is multiplied by 0.5. After 16 minutes, the reward is 
    # multiplied by an additional 0.5. After 24 minutes, there are no more rewards.
    if reward_name == 'upgrades_baseline':
        # now we don't separate the upgrades from the unit, 
        # so we ignore this reward now
        weight_hamming = 0.0
        weight_leven = 0.0

    rewards_traj = []
    for t1, t2, t3, t4, t5 in zip(trajectories.build_order, trajectories.z_build_order,
                                  trajectories.unit_counts, trajectories.z_unit_counts,
                                  trajectories.game_loop):
        # Calculate the Levenshtein distance,
        leven_batch = []
        for b1, b2, b5 in zip(t1, t2, t5):
            r = reward_by_build_order(b1, b2, b5)
            leven_batch.append(r)
        print('leven_batch:', leven_batch) if debug else None

        # Calculate the Hamming distance,
        hamming_batch = []
        for b1, b2, b5 in zip(t3, t4, t5):
            r = reward_by_unit_counts(b1, b2, b5)
            hamming_batch.append(r)
        print('hamming_batch:', hamming_batch) if debug else None

        reward_batch = []
        for r1, r2 in zip(leven_batch, hamming_batch):
            reward_batch.append(weight_leven * r1 + weight_hamming * r2)
        print('reward_batch:', reward_batch) if debug else None

        rewards_traj.append(reward_batch)

    print('rewards_traj:', rewards_traj) if debug else None 

    rewards_numpy = np.array(rewards_traj)
    rewards_tensor = torch.tensor(rewards_numpy, dtype=torch.float32, device=device)

    return rewards_tensor


def test():
    levenshtein = ED.levenshtein_recur
    hammingDist = ED.hammingDist

    Start = 0
    Stop = SCHP.max_unit_type

    print("Stop is :", Stop) if debug else None

    limit = 5

    l_1 = []
    l_2 = []

    # random
    #[l_1.append(random.randrange(Start, Stop)) for i in range(limit)]
    #[l_2.append(random.randrange(Start, Stop)) for i in range(limit)]

    # specfic
    l_1 = [13, 23, 45, 1114111]
    l_2 = [13, 45, 1114110]

    s_1 = list2str(l_1)
    s_2 = list2str(l_2)

    print("edit distance between 'l_1', 'l_2'", Levenshtein.distance(s_1, s_2))

    # note: hamming distance need the two lists have equal length
    l_1 = [13, 23, 45, 1114111]
    l_2 = [13, 21, 45, 1114110]

    s_1 = list2str(l_1)
    s_2 = list2str(l_2)

    print("hamming distance between 'l_1', 'l_2'", Levenshtein.hamming(s_1, s_2))  

    return


if __name__ == '__main__':
    test()
