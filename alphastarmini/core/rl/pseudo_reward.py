#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Computing the pseudo-reward (based on distance between the agent and a human player)."

# code for computing the pseudo-reward

import time
import random

import numpy as np

import torch

import Levenshtein

from alphastarmini.core.rl import utils as U

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
    reward = - dist * dist 
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
