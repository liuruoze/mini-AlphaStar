#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Help code for rl training "

# modified from AlphaStar pseudo-code
import os
import traceback
import collections
import copy

import numpy as np

import torch

from pysc2.lib.features import FeatureUnit

from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.utils import load_latest_model, initial_model_state_dict, get_batch_unit_type_mask

from alphastarmini.third import action_dict as AD

__author__ = "Ruo-Ze Liu"

debug = False

TRAJECTORY_FIELDS = [
    'state',  # Player observation.
    'baseline_state',  # Opponent observation, used for value network.
    'baseline_state_op',
    'memory',  # State of the agent (used for initial LSTM state).
    'z',  # Conditioning information for the policy.
    'is_final',  # If this is the last step.
    # namedtuple of masks for each action component. 0/False if final_step of
    # trajectory, or the argument is not used; else 1/True.
    'masks',
    'unit_type_entity_mask',
    'action',  # Action taken by the agent.
    'behavior_logits',  # namedtuple of logits of the behavior policy.
    'teacher_logits',  # namedtuple of logits of the supervised policy.
    'reward',  # Reward for the agent after taking the step.
    'player_select_units_num',  # action select units num
    'entity_num',  # the entity nums in this state
    'build_order',  # build_order
    'z_build_order',  # the build_order for the sampled replay
    'unit_counts',  # unit_counts
    'z_unit_counts',  # the unit_counts for the sampled replay
    'game_loop',  # seconds = int(game_loop / 22.4) 
    'last_list',  # [last_delay, last_action_type, last_repeat_queued]
]

Trajectory = collections.namedtuple('Trajectory', TRAJECTORY_FIELDS)


def get_supervised_agent(race, device, path="./model/", model_type="sl", restore=True):
    as_agent = AlphaStarAgent(name='supervised', race=race, initial_weights=None)

    if restore:
        # sl_model = load_latest_model(model_type=model_type, path=path)
        # if sl_model is not None:
        #     as_agent.agent_nn.model = sl_model
        initial_model_state_dict(model_type=model_type, path=path, model=as_agent.agent_nn.model, device=device)

    return as_agent


def namedtuple_one_list(trajectory_list):
    try:           
        d = [list(itertools.chain(*l)) for l in zip(*trajectory_list)]
        #print('len(d):', len(d))
        new = Trajectory._make(d)
        del d

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
        del d

    except Exception as e:
        print("stack_namedtuple Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())

        return None

    return new


def namedtuple_zip(trajectory):
    try: 
        d = [list(zip(*z)) for z in trajectory]
        new = Trajectory._make(d)
        del d

    except Exception as e:
        print("namedtuple_zip Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())

        return None

    return new


def namedtuple_deepcopy(trajectories):
    try:
        # shape before: [batch_size x dict_name x seq_size]
        # 
        #trajectory = stack_namedtuple(trajectories)
        batch_list = [] 
        for trajectory in trajectories:
            d = [copy.deepcopy(z) for z in trajectory]
            new = Trajectory._make(d)
            batch_list.append(new)

        # batch_list = [] 
        # for batch_item in batch_seq_trajectory:
        #     seq_list = []
        #     for seq_item in batch_item:
        #         d = [copy.deepcopy(z) for z in seq_item]
        #         d = Trajectory._make(d)
        #         seq_list.append(d)
        #         del seq_item
        #     del batch_item
        #     batch_list.append(seq_list)
        # del batch_seq_trajectory

    except Exception as e:
        print("namedtuple_zip Exception cause return, Detials of the Exception:", e)
        print(traceback.format_exc())

        return None

    return batch_list


def get_unit_type_mask(action, obs):
    function_id = action.action_type
    obs_list = [obs]
    action_types = [function_id]

    unit_type_masks = get_batch_unit_type_mask(action_types, obs_list)
    unit_type_mask = unit_type_masks[0].reshape(-1)
    del unit_type_masks, obs_list, action_types, function_id

    return unit_type_mask


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
    del action, function_id, need_args

    return action_mask
