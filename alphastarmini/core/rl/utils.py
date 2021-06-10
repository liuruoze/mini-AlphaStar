#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Help code for rl training "

# modified from AlphaStar pseudo-code
import os
import torch
import traceback
import collections

from alphastarmini.lib.utils import load_latest_model
from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent

__author__ = "Ruo-Ze Liu"

debug = False

TRAJECTORY_FIELDS = [
    'observation',  # Player observation.
    'opponent_observation',  # Opponent observation, used for value network.
    'memory',  # State of the agent (used for initial LSTM state).
    'z',  # Conditioning information for the policy.
    'is_final',  # If this is the last step.
    # namedtuple of masks for each action component. 0/False if final_step of
    # trajectory, or the argument is not used; else 1/True.
    'masks',
    'action',  # Action taken by the agent.
    'behavior_logits',  # namedtuple of logits of the behavior policy.
    'teacher_logits',  # namedtuple of logits of the supervised policy.
    'reward',  # Reward for the agent after taking the step.
    'build_order',  # build_order
    'z_build_order',  # the build_order for the sampled replay
    'unit_counts',  # unit_counts
    'z_unit_counts',  # the unit_counts for the sampled replay
]

Trajectory = collections.namedtuple('Trajectory', TRAJECTORY_FIELDS)


def get_supervised_agent(race, path="./model/", model_type="sl", restore=True):
    agent = AlphaStarAgent(name='supervised', race=race, initial_weights=None)

    if restore:
        agent.model = load_latest_model(model_type=model_type, path=path)

    return agent


def namedtuple_one_list(trajectory_list):
    try:           
        d = [list(itertools.chain(*l)) for l in zip(*trajectory_list)]
        #print('len(d):', len(d))
        new = Trajectory._make(d)

    except Exception as e:
        print("stack_namedtuple Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())

        return None

    return new


def stack_namedtuple(trajectory):
    try:           
        d = list(zip(*trajectory))
        #print('len(d):', len(d))
        new = Trajectory._make(d)

    except Exception as e:
        print("stack_namedtuple Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())

        return None

    return new


def namedtuple_zip(trajectory):
    try: 
        d = [list(zip(*z)) for z in trajectory]
        new = Trajectory._make(d)

    except Exception as e:
        print("namedtuple_zip Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())

        return None

    return new


def get_mask(action, action_spec):
    function_id = action.action_type.item()
    need_args = action_spec.functions[function_id].args

    # action type and delay is always enable
    action_mask = [1, 1, 0, 0, 0, 0]

    for arg in need_args:
        print("arg:", arg) if debug else None
        if arg.name == 'queued':
            action_mask[2] = 1
        elif arg.name == 'unit_tags':
            action_mask[3] = 1
        elif arg.name == 'target_unit_tag':
            action_mask[4] = 1
        elif arg.name == 'world':
            action_mask[5] = 1                       

    print('action_mask:', action_mask) if debug else None

    return action_mask
