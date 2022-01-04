#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for RL losses."

# modified from AlphaStar pseudo-code
import traceback
import collections
import itertools

import numpy as np

import torch
import torch.nn as nn

from alphastarmini.core.rl.rl_utils import Trajectory
from alphastarmini.core.rl.action import ArgsActionLogits
from alphastarmini.core.rl.action import ArgsAction

from alphastarmini.core.rl import rl_algo as RA
from alphastarmini.core.rl import rl_utils as RU
from alphastarmini.core.rl import pseudo_reward as PR

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS


__author__ = "Ruo-Ze Liu"

debug = False


ACTION_FIELDS = [
    'action_type',  
    'delay',
    'queue',
    'units',
    'target_unit',
    'target_location',
]

SELECTED_UNITS_PLUS_ONE = False

# below now only consider action_type now
# baseline are also all zeros
# TODO, change to a more right implementation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def filter_by_for_lists(action_fields, target_list):
    return torch.cat([getattr(b, action_fields) for a in target_list for b in a], dim=0)


def filter_by_for_masks(action_fields, target_mask):
    index = ACTION_FIELDS.index(action_fields)
    mask = torch.tensor(target_mask, device=device)
    mask = mask[:, :, index]
    return mask


def compute_over_actions(f, *args):
    """Runs f over all elements in the lists composing *args.
    """
    return sum(f(*a) for a in zip(*args))


def mergeArgsActionLogits(list_args_action_logits):
    l = [i.toList() for i in list_args_action_logits]
    a = [torch.cat(z, dim=0) for z in zip(*l)]
    b = [t.reshape(t.shape[0], -1) for t in a]

    return ArgsActionLogits(*b)


def entropy_loss_for_all_arguments(target_logits, trajectories, target_select_units_num, target_entity_num):
    """Computes the entropy loss for a set of logits.

    Args:
      target_logits: [batch_size, seq_size, -1]
      target_select_units_num: [batch_size, seq_size, 1]
      trajectories: [seq_size, batch_size, -1]
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """
    device = target_logits.action_type.device

    target_select_units_num = target_select_units_num.reshape(AHP.sequence_length * AHP.batch_size, -1)
    if SELECTED_UNITS_PLUS_ONE:
        target_select_units_num = target_select_units_num + 1

    target_selected_mask = torch.arange(AHP.max_selected, device=device).float()
    target_selected_mask = target_selected_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    target_selected_mask = target_selected_mask < target_select_units_num
    assert target_selected_mask.dtype == torch.bool

    player_select_units_num = torch.tensor(trajectories.player_select_units_num, device=device).reshape(-1)
    if SELECTED_UNITS_PLUS_ONE:
        player_select_units_num = player_select_units_num + 1

    player_selected_mask = torch.arange(AHP.max_selected, device=device).float()
    player_selected_mask = player_selected_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    player_selected_mask = player_selected_mask < player_select_units_num.unsqueeze(dim=1)
    assert player_selected_mask.dtype == torch.bool

    entity_mask = torch.arange(AHP.max_entities, device=device).float()
    entity_mask = entity_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    entity_mask = entity_mask < target_entity_num.reshape(-1).unsqueeze(dim=-1)

    masks = torch.tensor(trajectories.masks, device=device)
    entropy_loss = 0  # torch.zeros(1, dtype=torch.float32, device=device)

    for i, field in enumerate(ACTION_FIELDS):    
        logits = getattr(target_logits, field) 
        print("field name:", field) if debug else None
        print("logits.shape:", logits.shape) if debug else None

        # logits = logits.reshape(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        # logits = logits.transpose(0, 1)
        logits = logits.reshape(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))

        all_logits = [logits]

        if field == "units":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size, AHP.max_selected, AHP.max_entities) for logit in all_logits]
        elif field == "target_unit":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size * 1, AHP.max_entities) for logit in all_logits]     
        elif field == "target_location":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]

        [logits] = all_logits

        if field == "units":
            selected_mask = target_selected_mask * player_selected_mask
            entropy = RA.entropy(logits, selected_mask=selected_mask, entity_mask=entity_mask.unsqueeze(-2), debug=False)
            entropy = entropy.sum(dim=-2)
        else:
            if field == "target_unit":
                entropy = RA.entropy(logits, entity_mask=entity_mask, debug=False)
            else:
                entropy = RA.entropy(logits)
        entropy = entropy.sum(dim=-1)

        # Normalize by actions available.
        entropy = entropy / torch.log(torch.tensor(logits.shape[-1], 
                                                   dtype=torch.float32, 
                                                   device=device))

        mask = masks[:, :, i]
        mask = mask.reshape(-1, 1)
        entropy = entropy * mask

        entropy_loss += entropy.mean()

    return entropy_loss


def human_policy_kl_loss_all(target_logits, trajectories, target_select_units_num, target_entity_num):
    """Computes the KL loss to the human policy.

    Args:
      target_logits: [batch_size, seq_size, -1]
      target_select_units_num: [batch_size, seq_size, 1]
      trajectories: [seq_size, batch_size, -1]
    Returns:

    """
    device = target_logits.action_type.device
    teacher_logits = trajectories.teacher_logits

    target_select_units_num = target_select_units_num.reshape(AHP.sequence_length * AHP.batch_size, -1)
    if SELECTED_UNITS_PLUS_ONE:
        target_select_units_num = target_select_units_num + 1

    target_selected_mask = torch.arange(AHP.max_selected, device=device).float()
    target_selected_mask = target_selected_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    target_selected_mask = target_selected_mask < target_select_units_num
    assert target_selected_mask.dtype == torch.bool

    player_select_units_num = torch.tensor(trajectories.player_select_units_num, device=device).reshape(-1)
    if SELECTED_UNITS_PLUS_ONE:
        player_select_units_num = player_select_units_num + 1

    player_selected_mask = torch.arange(AHP.max_selected, device=device).float()
    player_selected_mask = player_selected_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    player_selected_mask = player_selected_mask < player_select_units_num.unsqueeze(dim=1)
    assert player_selected_mask.dtype == torch.bool

    entity_mask = torch.arange(AHP.max_entities, device=device).float()
    entity_mask = entity_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    entity_mask = entity_mask < target_entity_num.reshape(-1).unsqueeze(dim=-1)

    masks = torch.tensor(trajectories.masks, device=device)
    kl_loss = 0  # torch.zeros(1, dtype=torch.float32, device=device)

    for i, field in enumerate(ACTION_FIELDS):    

        logits = getattr(target_logits, field) 
        print("field name:", field) if debug else None
        print("logits.shape:", logits.shape) if debug else None

        # logits = logits.reshape(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        # logits = logits.transpose(0, 1)
        logits = logits.reshape(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))

        t_logits = filter_by_for_lists(field, teacher_logits) 
        print("t_logits.shape:", t_logits.shape) if debug else None

        all_logits = [logits, t_logits]

        if field == "units":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size, AHP.max_selected, AHP.max_entities) for logit in all_logits]
        elif field == "target_unit":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size * 1, AHP.max_entities) for logit in all_logits]     
        elif field == "target_location":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]

        [logits, t_logits] = all_logits

        if field == "units":
            selected_mask = target_selected_mask * player_selected_mask
            kl = RA.kl(logits, t_logits, selected_mask=selected_mask, entity_mask=entity_mask.unsqueeze(-2), 
                       debug=False, target_entity_num=target_entity_num.reshape(-1), target_select_units_num=target_select_units_num.reshape(-1),
                       player_select_units_num=player_select_units_num.reshape(-1))
            kl = kl.sum(dim=-2)
        else:
            if field == "target_unit":
                kl = RA.kl(logits, t_logits, entity_mask=entity_mask, debug=False)
            else:
                kl = RA.kl(logits, t_logits)
        kl = kl.sum(dim=-1)

        mask = masks[:, :, i]
        mask = mask.reshape(-1, 1)

        kl = kl * mask
        kl = kl.mean()
        print("kl:", kl) if debug else None

        kl_loss += kl

    return kl_loss


def human_policy_kl_loss_action(target_logits, trajectories):
    """Computes the KL loss to the human policy.

    Args:
      target_logits: [batch_size, seq_size, -1]
      trajectories: [seq_size, batch_size, -1]
    Returns:

    """
    device = target_logits.action_type.device
    teacher_logits = trajectories.teacher_logits

    game_loop = torch.tensor(trajectories.game_loop, device=device).reshape(AHP.sequence_length * AHP.batch_size)
    seconds = game_loop / 22.4
    flag = seconds < 4 * 60

    masks = torch.tensor(trajectories.masks, device=device)
    kl_loss = 0  # torch.zeros(1, dtype=torch.float32, device=device)

    for i, field in enumerate(ACTION_FIELDS):
        if i != 0:
            break

        logits = getattr(target_logits, field) 
        print("field name:", field) if debug else None
        print("logits.shape:", logits.shape) if debug else None

        # logits = logits.reshape(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        # logits = logits.transpose(0, 1)
        logits = logits.reshape(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))

        t_logits = filter_by_for_lists(field, teacher_logits) 
        all_logits = [logits, t_logits]

        if field == "units":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size, AHP.max_selected, AHP.max_entities) for logit in all_logits]
        elif field == "target_unit":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size * 1, AHP.max_entities) for logit in all_logits]     
        elif field == "target_location":
            all_logits = [logit.reshape(AHP.sequence_length * AHP.batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]

        [logits, t_logits] = all_logits

        if field == "units":
            kl = RA.kl(logits, t_logits, target_selected_mask * player_selected_mask)
            kl = kl.sum(dim=-2)
        else:
            kl = RA.kl(logits, t_logits)
        kl = kl.sum(dim=-1)

        if i == 0:
            print("kl.shape:", kl.shape) if debug else None
            print("flag.shape:", flag.shape) if debug else None
            kl = kl * flag

        mask = masks[:, :, i]
        mask = mask.reshape(-1, 1)

        kl = kl * mask
        kl = kl.mean()
        print("kl:", kl) if debug else None

        kl_loss += kl

    return kl_loss


def td_lambda_loss(baselines, rewards, trajectories): 
    # note, use '~' must ensure the type is bool
    discounts = ~np.array(trajectories.is_final[:-1], dtype=np.bool)
    discounts = torch.tensor(discounts, device=device)

    baselines = baselines
    # rewards should be T_0 -> T_{n-1}
    rewards = rewards[:-1]

    # The baseline is then updated using TDLambda, with relative weighting 10.0 and lambda 0.8.
    returns = RA.lambda_returns(baselines[1:], rewards, discounts, lambdas=0.8)

    # returns = stop_gradient(returns)
    returns = returns.detach()
    print("returns:", returns) if debug else None

    result = returns - baselines[:-1]
    print("result:", result) if debug else None

    # change to pytorch version
    return 0.5 * torch.mean(torch.square(result))


def policy_gradient_loss(logits, actions, advantages, mask, selected_mask=None,
                         entity_mask=None, debug=False, outlier_remove=False):
    """Helper function for computing policy gradient loss for UPGO and v-trace."""

    # logits: shape [BATCH_SIZE, CLASS_SIZE]
    # actions: shape [BATCH_SIZE]
    # advantages: shape [BATCH_SIZE]

    # selected_mask: shape [BATCH_SIZE, MAX_SELECT]
    # entity_mask: shape [BATCH_SIZE, MAX_ENTITY]

    if entity_mask is not None:
        # if it is not None, it is the units head
        entity_mask = entity_mask.unsqueeze(dim=1)

        print("logits.shape:", logits.shape) if 1 else None
        print("actions.shape:", actions.shape) if 1 else None
        print("entity_mask.shape:", entity_mask.shape) if 1 else None

        # logits: shape [BATCH_SIZE, MAX_SELECT, MAX_ENTITY]
        # entity_mask: shape [BATCH_SIZE, 1, MAX_ENTITY]
        logits = logits * entity_mask

        if True:
            outlier_mask = (torch.abs(logits) == 1e8)
            if outlier_mask.any() > 0:
                print("outlier_mask:", outlier_mask.nonzero(as_tuple=True)) if 1 else None
                index = outlier_mask.nonzero(as_tuple=True)
                print("logits[index]:", logits[index]) if 1 else None
                print("actions[index[0:2]]:", actions[index[0], 0]) if 1 else None
                print("actions[index[0:2]]:", actions[index[0], 1]) if 1 else None
                print("mask[index[0]]:", mask[index[0]]) if 1 else None

        if len(logits.shape) == 3:
            select_size = logits.shape[1]
            logits = logits.reshape(-1, logits.shape[-1])
        if len(actions.shape) == 3:
            actions = actions.reshape(-1, actions.shape[-1])

        print("actions.shape:", actions.shape) if debug else None
        print("logits.shape:", logits.shape) if debug else None

    action_log_prob = RA.log_prob(actions, logits, reduction="none")
    print("action_log_prob:", action_log_prob) if debug else None
    print("action_log_prob.shape:", action_log_prob.shape) if debug else None

    advantages = advantages.clone().detach()
    print("advantages:", advantages.mean()) if debug else None
    print("advantages.shape:", advantages.shape) if debug else None

    if selected_mask is not None:
        # if it is not None, it is the units head

        # action_log_prob: shape [BATCH_SIZE, MAX_SELECT]
        # selected_mask: shape [BATCH_SIZE, MAX_SELECT]       

        if len(action_log_prob.shape) == 1:
            action_log_prob = action_log_prob.reshape(-1, select_size)
        print("action_log_prob.shape:", action_log_prob.shape) if debug else None

        action_log_prob = action_log_prob * selected_mask
        action_log_prob = torch.sum(action_log_prob, dim=-1)

        # action_log_prob: shape [BATCH_SIZE]
        print("action_log_prob.shape:", action_log_prob.shape) if debug else None

    print("mask:", mask) if debug else None
    print("mask.shape:", mask.shape) if debug else None

    results = mask * advantages * action_log_prob
    print("results:", results) if debug else None
    print("results.shape:", results.shape) if debug else None

    if outlier_remove:
        outlier_mask = (torch.abs(results) >= 1e6)
        results = results * ~outlier_mask
    else:
        outlier_mask = (torch.abs(results) >= 1e6)
        if outlier_mask.any() > 0:
            print("outlier_mask:", outlier_mask.nonzero(as_tuple=True)) if 1 else None

            index = outlier_mask.nonzero(as_tuple=True)

            print("action_log_prob[index]:", action_log_prob[index]) if 1 else None
            print("advantages[index]:", advantages[index]) if 1 else None
            print("mask[index]:", mask[index]) if 1 else None

            # stop()

    # note, we should do policy ascent on the results
    # which means if we use policy descent, we should add a "-" sign for results
    loss = -results.mean()

    return loss


def vtrace_pg_loss(target_logits, target_actions, baselines, rewards, trajectories,
                   action_fields, target_select_units_num, target_entity_num):
    print('action_fields', action_fields) if debug else None

    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories))

    rewards = rewards[:-1]
    values = baselines[:-1]

    sequence_length = rewards.shape[0]
    batch_size = rewards.shape[1]

    device = target_logits.device

    target_logits = getattr(target_logits, action_fields)
    target_actions = getattr(target_actions, action_fields)

    target_actions = target_actions[:-1]
    target_logits = target_logits[:-1]
    target_select_units_num = target_select_units_num[:-1].reshape(-1)
    target_entity_num = target_entity_num[:-1].reshape(-1)

    seqbatch_shape = sequence_length * batch_size
    target_logits = target_logits.view(seqbatch_shape, *tuple(target_logits.shape[2:]))

    behavior_logits = filter_by_for_lists(action_fields, trajectories.behavior_logits)
    actions = filter_by_for_lists(action_fields, trajectories.action)

    selected_mask = None
    if action_fields == 'units':
        seqbatch_unit_shape = target_logits.shape[0:2]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        target_actions = target_actions.reshape(-1, target_actions.shape[-1])

        player_select_units_num = torch.tensor(trajectories.player_select_units_num, device=device).reshape(-1)
        selected_mask = torch.arange(AHP.max_selected, device=device).float()
        selected_mask = selected_mask.repeat(seqbatch_shape, 1)
        selected_mask = selected_mask < player_select_units_num.unsqueeze(dim=-1)

        target_selected_mask = torch.arange(AHP.max_selected, device=device).float()
        target_selected_mask = target_selected_mask.repeat(seqbatch_shape, 1)
        target_selected_mask = target_selected_mask < target_select_units_num.unsqueeze(-1)

        #entity_num = torch.tensor(trajectories.entity_num, device=target_logits.device).reshape(-1)
        entity_mask = torch.arange(AHP.max_entities, device=device).float()
        entity_mask = entity_mask.repeat(seqbatch_shape, 1)
        entity_mask = entity_mask < target_entity_num.unsqueeze(dim=-1)

    elif action_fields == 'target_unit':
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        #entity_num = torch.tensor(trajectories.entity_num, device=target_logits.device).reshape(-1)
        entity_mask = torch.arange(AHP.max_entities, device=device).float()
        entity_mask = entity_mask.repeat(seqbatch_shape, 1)
        entity_mask = entity_mask < target_entity_num.unsqueeze(dim=-1)

    elif action_fields == 'target_location':
        target_logits = target_logits.reshape(target_logits.shape[0], -1)
        behavior_logits = behavior_logits.reshape(behavior_logits.shape[0], -1)

        actions_tmp = torch.zeros(behavior_logits.shape[0], 1, dtype=torch.int64, device=device)
        for i, pos in enumerate(actions):
            [x, y] = pos
            index = SCHP.world_size * y + x
            actions_tmp[i][0] = index
        actions = actions_tmp

    clipped_rhos = RA.compute_importance_weights(behavior_logits, target_logits, actions)
    print("clipped_rhos.shape", clipped_rhos.shape) if debug else None

    if action_fields == 'units' or action_fields == 'target_unit':
        clipped_rhos = clipped_rhos.reshape(seqbatch_shape, -1)
        clipped_rhos = torch.sum(clipped_rhos, dim=-1)
        print('clipped_rhos.shape', clipped_rhos.shape) if debug else None

    clipped_rhos = clipped_rhos.reshape(rewards.shape)
    print("clipped_rhos", clipped_rhos) if debug else None
    print("clipped_rhos.shape", clipped_rhos.shape) if debug else None

    discounts = ~np.array(trajectories.is_final, dtype=np.bool)
    discounts = torch.tensor(discounts, dtype=torch.float32, device=device)
    weighted_advantage = RA.vtrace_advantages(clipped_rhos, rewards,
                                              discounts, values,
                                              baselines[-1])[1].reshape(-1)
    masks = filter_by_for_masks(action_fields, trajectories.masks).reshape(-1)

    if action_fields == 'units':
        # print("target_logits.shape", target_logits.shape) if 1 else None
        # print("actions.shape", actions.shape) if 1 else None
        # print("weighted_advantage.shape", weighted_advantage.shape) if 1 else None
        # print("masks.shape", masks.shape) if 1 else None
        # print("selected_mask.shape", selected_mask.shape) if 1 else None
        # print("target_selected_mask.shape", target_selected_mask.shape) if 1 else None
        # print("entity_mask.shape", entity_mask.shape) if 1 else None

        target_logits = target_logits.reshape(sequence_length * batch_size, AHP.max_selected, -1)
        actions = actions.reshape(sequence_length * batch_size, AHP.max_selected, -1)
        target_actions = target_actions.reshape(sequence_length * batch_size, AHP.max_selected, -1)
        selected_mask = selected_mask * target_selected_mask

        result = policy_gradient_loss(target_logits, actions, weighted_advantage, masks, selected_mask, entity_mask)

    elif action_fields == 'target_unit':
        target_logits = target_logits.reshape(sequence_length * batch_size, 1, -1)
        actions = actions.reshape(sequence_length * batch_size, 1, -1)

        result = policy_gradient_loss(target_logits, actions, weighted_advantage, masks, entity_mask=entity_mask)
    else:
        target_logits = target_logits.reshape(seqbatch_shape, -1)
        actions = actions.reshape(seqbatch_shape, -1)

        result = policy_gradient_loss(target_logits, actions, weighted_advantage, masks)

    return result


def split_upgo_loss(target_logits, baselines, trajectories, target_select_units_num, target_entity_num):
    """Computes split UPGO policy gradient loss.

    See Methods for details on UPGO.
    """
    # Remove last timestep from trajectories and baselines.
    trajectories = Trajectory(*tuple(t[:-1] for t in trajectories))
    print("trajectories.reward", trajectories.reward) if debug else None

    values = baselines[:-1]
    # shape: list of [seq_size x batch_size]
    print("values", values) if debug else None
    print("values.shape", values.shape) if debug else None

    # we change it to pytorch version
    # returns = upgo_returns(values.detach().numpy(), np.array(trajectories.reward), ~np.array(trajectories.is_final), 
    # baselines[-1].detach().numpy())
    reward_tensor = torch.tensor(np.array(trajectories.reward), dtype=torch.float32, device=device)
    discounts = torch.tensor(~np.array(trajectories.is_final, dtype=np.bool), dtype=torch.float32, device=device)
    returns = RA.upgo_returns(values, reward_tensor, discounts, baselines[-1])

    # shape: list of [seq_size x batch_size]
    print("returns", returns) if debug else None
    print("returns.shape", returns.shape) if debug else None

    # Compute the UPGO loss for each action subset.
    # action_type, delay, and other arguments are also similarly separately 
    # updated using UPGO, in the same way as the VTrace Actor-Critic loss, 
    # with relative weight 1.0.
    # We make upgo also contains all the arguments
    loss = sum_upgo_loss(target_logits, values, trajectories, returns, target_select_units_num, target_entity_num)

    return loss


def sum_upgo_loss(target_logits, values, trajectories, returns, target_select_units_num, target_entity_num):
    """Computes the split upgo policy gradient loss.
    """
    loss = 0.
    print('sum_upgo_loss') if 1 else None

    for i, field in enumerate(ACTION_FIELDS):
        loss_field = upgo_loss(target_logits, values, trajectories, returns, field,
                               target_select_units_num, target_entity_num)
        loss_val = loss_field.item()
        print('field', field, 'loss_val', loss_val) if 1 else None

    loss += loss_field

    return loss


def sum_vtrace_pg_loss(target_logits, target_actions, target_select_units_num, target_entity_num, baselines, rewards, trajectories):
    """Computes the split v-trace policy gradient loss.
    """
    loss = 0.
    print('sum_vtrace_pg_loss') if 1 else None

    for i, field in enumerate(ACTION_FIELDS):
        loss_field = vtrace_pg_loss(target_logits, target_actions, baselines, rewards, trajectories, field,
                                    target_select_units_num, target_entity_num)
        loss_val = loss_field.item()
        print('field', field, 'loss_val', loss_val) if 1 else None

        loss += loss_field

    return loss


def upgo_loss(target_logits, values, trajectories, returns, action_fields, target_select_units_num, target_entity_num):
    print('action_fields', action_fields) if debug else None

    sequence_length = returns.shape[0]
    batch_size = returns.shape[1]

    target_logits = getattr(target_logits, action_fields)
    device = target_logits.device

    target_logits = target_logits[:-1]
    target_select_units_num = target_select_units_num[:-1].reshape(-1)
    target_entity_num = target_entity_num[:-1].reshape(-1)

    seqbatch_shape = sequence_length * batch_size
    target_logits = target_logits.view(seqbatch_shape, *tuple(target_logits.shape[2:]))

    behavior_logits = filter_by_for_lists(action_fields, trajectories.behavior_logits)
    actions = filter_by_for_lists(action_fields, trajectories.action)

    selected_mask = None
    if action_fields == 'units':
        seqbatch_unit_shape = target_logits.shape[0:2]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        player_select_units_num = torch.tensor(trajectories.player_select_units_num, device=device).reshape(-1)
        selected_mask = torch.arange(AHP.max_selected, device=device).float()
        selected_mask = selected_mask.repeat(seqbatch_shape, 1)
        selected_mask = selected_mask < player_select_units_num.unsqueeze(dim=-1)

        target_selected_mask = torch.arange(AHP.max_selected, device=device).float()
        target_selected_mask = target_selected_mask.repeat(seqbatch_shape, 1)
        target_selected_mask = target_selected_mask < target_select_units_num.unsqueeze(-1)

        #entity_num = torch.tensor(trajectories.entity_num, device=target_logits.device).reshape(-1)
        entity_mask = torch.arange(AHP.max_entities, device=device).float()
        entity_mask = entity_mask.repeat(seqbatch_shape, 1)
        entity_mask = entity_mask < target_entity_num.unsqueeze(dim=-1)

    elif action_fields == 'target_unit':
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        #entity_num = torch.tensor(trajectories.entity_num, device=target_logits.device).reshape(-1)
        entity_mask = torch.arange(AHP.max_entities, device=device).float()
        entity_mask = entity_mask.repeat(seqbatch_shape, 1)
        entity_mask = entity_mask < target_entity_num.unsqueeze(dim=-1)

    elif action_fields == 'target_location':
        target_logits = target_logits.reshape(target_logits.shape[0], -1)
        behavior_logits = behavior_logits.reshape(behavior_logits.shape[0], -1)

        actions_tmp = torch.zeros(behavior_logits.shape[0], 1, dtype=torch.int64, device=device)
        for i, pos in enumerate(actions):
            [x, y] = pos
            index = SCHP.world_size * y + x
            actions_tmp[i][0] = index
        actions = actions_tmp

    clipped_rhos = RA.compute_importance_weights(behavior_logits, target_logits, actions)
    print("clipped_rhos.shape", clipped_rhos.shape) if debug else None

    if action_fields == 'units' or action_fields == 'target_unit':
        clipped_rhos = clipped_rhos.reshape(seqbatch_shape, -1)
        clipped_rhos = torch.sum(clipped_rhos, dim=-1)
        print('clipped_rhos.shape', clipped_rhos.shape) if debug else None

    clipped_rhos = clipped_rhos.reshape(sequence_length, batch_size)
    print("clipped_rhos", clipped_rhos) if debug else None
    print("clipped_rhos.shape", clipped_rhos.shape) if debug else None

    discounts = ~np.array(trajectories.is_final, dtype=np.bool)
    discounts = torch.tensor(discounts, dtype=torch.float32, device=device)

    weighted_advantage = (returns - values) * clipped_rhos
    weighted_advantage = weighted_advantage.reshape(-1)

    masks = filter_by_for_masks(action_fields, trajectories.masks).reshape(-1)

    if action_fields == 'units':
        target_logits = target_logits.reshape(sequence_length * batch_size, AHP.max_selected, -1)
        actions = actions.reshape(sequence_length * batch_size, AHP.max_selected, -1)
        selected_mask = selected_mask * target_selected_mask

        result = policy_gradient_loss(target_logits, actions, weighted_advantage, masks, selected_mask, entity_mask)

    elif action_fields == 'target_unit':
        target_logits = target_logits.reshape(sequence_length * batch_size, 1, -1)
        actions = actions.reshape(sequence_length * batch_size, 1, -1)

        result = policy_gradient_loss(target_logits, actions, weighted_advantage, masks, entity_mask=entity_mask)
    else:
        target_logits = target_logits.reshape(seqbatch_shape, -1)
        actions = actions.reshape(seqbatch_shape, -1)

        result = policy_gradient_loss(target_logits, actions, weighted_advantage, masks)

    return result


def get_baseline_hyperparameters():
    # Accroding to detailed_architecture.txt, baselines contains the following 5 items:
    # Winloss Baseline;
    # Build Order Baseline;
    # Built Units Baseline;
    # Upgrades Baseline;
    # Effects Baseline.
    # Thus BASELINE_COSTS_AND_REWARDS also have 5 entry
    winloss_baseline_costs = (1.0, 10.0, "winloss_baseline")

    # AlphaStar: The updates are computed similar to Winloss, except without UPGO, applied using Build Order baseline, and 
    # with relative weightings 4.0 for the policy and 1.0 for the baseline.
    build_order_baseline_costs = (4.0, 1.0, "build_order_baseline")

    # AlphaStar: The updates are computed similar to Winloss, except without UPGO, applied using Built Units baseline, 
    # and with relative weightings 6.0 for the policy and 1.0 for the baseline.
    built_units_baseline_costs = (6.0, 1.0, "built_units_baseline")

    # AlphaStar: The updates are computed similar to Winloss, except without UpGO, applied using Upgrades baseline, and 
    # with relative weightings 6.0 for the policy and 1.0 for the baseline.
    upgrades_baseline_costs = (6.0, 1.0, "upgrades_baseline")

    # AlphaStar: The updates are computed similar to Winloss, except without UPGO, applied using Effects baseline, and 
    # with relative weightings 6.0 for the policy and 1.0 for the baseline.
    effects_baseline_costs = (6.0, 1.0, "effects_baseline")

    BASELINE_COSTS_AND_REWARDS = [winloss_baseline_costs, build_order_baseline_costs, built_units_baseline_costs,
                                  upgrades_baseline_costs, effects_baseline_costs]

    return BASELINE_COSTS_AND_REWARDS


def transpose_target_logits(target_logits):

    # change to [seq_size x batch_size x -1] format for computing RL loss
    for i, field in enumerate(ACTION_FIELDS):  

        logits = getattr(target_logits, field) 
        print("field name:", field) if debug else None
        print("logits.shape:", logits.shape) if debug else None

        logits = logits.view(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        logits = logits.transpose(0, 1).contiguous()
        logits = logits.view(AHP.sequence_length, AHP.batch_size, *tuple(logits.shape[2:]))
        print("logits.shape:", logits.shape) if debug else None

        setattr(target_logits, field, logits)

    return target_logits


def transpose_baselines(baseline_list):
    baselines = [baseline.reshape(AHP.batch_size, AHP.sequence_length).transpose(0, 1).contiguous() for baseline in baseline_list]
    return baselines


def transpose_sth(x):
    x = x.view(AHP.batch_size, AHP.sequence_length)
    x = x.transpose(0, 1).contiguous()
    return x


def loss_function(agent, trajectories, use_opponent_state=True):
    """Computes the loss of trajectories given weights."""

    # target_logits: ArgsActionLogits
    target_logits, target_actions, baselines, target_select_units_num, target_entity_num = agent.unroll(trajectories, use_opponent_state)
    target_logits = transpose_target_logits(target_logits)
    baselines = transpose_baselines(baselines)
    target_select_units_num = transpose_sth(target_select_units_num)
    target_entity_num = transpose_sth(target_entity_num)
    target_actions = transpose_target_logits(target_actions)

    print("target_entity_num:", target_entity_num) if 1 else None
    print("target_entity_num.shape:", target_entity_num.shape) if 1 else None

    # note, we change the structure of the trajectories
    trajectories = RU.stack_namedtuple(trajectories) 

    # shape before: [dict_name x batch_size x seq_size]
    # shape after: [dict_name x seq_size x batch_size]
    trajectories = RU.namedtuple_zip(trajectories) 

    entity_num = torch.tensor(trajectories.entity_num, device=device)
    print("entity_num:", entity_num) if 1 else None
    print("entity_num.shape:", entity_num.shape) if 1 else None

    assert torch.equal(target_entity_num, entity_num)

    loss_actor_critic = 0.

    # We use a number of actor-critic losses - one for the winloss baseline, which
    # outputs the probability of victory, and one for each pseudo-reward
    # associated with following the human strategy statistic z.
    # See the paper methods and detailed_architecture.txt for more details.
    BASELINE_COSTS_AND_REWARDS = get_baseline_hyperparameters()

    reward_index = 0
    for baseline, costs_and_rewards in zip(baselines, BASELINE_COSTS_AND_REWARDS):

        # baseline is for caluculation in td_lambda and vtrace_pg
        # costs_and_rewards are only weight for loss
        if reward_index == 0:

            pg_cost, baseline_cost, reward_name = costs_and_rewards
            print("reward_name:", reward_name) if 1 else None

            rewards = PR.compute_pseudoreward(trajectories, reward_name, device=target_logits.action_type.device)
            print("rewards:", rewards) if 0 else None

            # The action_type argument, delay, and all other arguments are separately updated 
            # using a separate ("split") VTrace Actor-Critic losses. The weighting of these 
            # updates will be considered 1.0. action_type, delay, and other arguments are 
            # also similarly separately updated using UPGO, in the same way as the VTrace 
            # Actor-Critic loss, with relative weight 1.0. 
            lambda_loss = td_lambda_loss(baseline, rewards, trajectories)
            print("lambda_loss:", lambda_loss) if debug else None

            loss_actor_critic += (baseline_cost * lambda_loss)

            # we add vtrace loss
            pg_loss = sum_vtrace_pg_loss(target_logits, target_actions, target_select_units_num, target_entity_num, 
                                         baseline, rewards, trajectories)
            print("pg_loss:", pg_loss) if debug else None

            loss_actor_critic += (pg_cost * pg_loss)

        reward_index += 1

    # Note: upgo_loss has only one baseline which is just for winloss 
    # AlphaStar: loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines.winloss_baseline, trajectories)
    UPGO_WEIGHT = 1.0
    loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines[0], trajectories, target_select_units_num, target_entity_num)

    # Distillation Loss:
    # There is an distillation loss with weight 2e-3 on all action arguments, 
    # to match the output logits of the fine-tuned supervised policy 
    # which has been given the same observation.
    # If the trajectory was conditioned on `cumulative_statistics`, there is an additional 
    # distillation loss of weight 1e-1 on the action type logits 
    # for the first four minutes of the game.
    # Thus ALL_KL_COST = 2e-3
    # and ACTION_TYPE_KL_COST = 1e-1
    ALL_KL_COST = 2e-3
    ACTION_TYPE_KL_COST = 1e-1

    # for all arguments
    all_kl_loss = human_policy_kl_loss_all(target_logits, trajectories, target_select_units_num, target_entity_num)
    action_type_kl_loss = human_policy_kl_loss_action(target_logits, trajectories)

    loss_kl = ALL_KL_COST * all_kl_loss + ACTION_TYPE_KL_COST * action_type_kl_loss

    # Entropy Loss:
    # There is an entropy loss with weight 1e-4 on all action arguments, 
    # masked by which arguments are possible for a given action type.
    # Thus ENT_WEIGHT = 1e-4
    ENT_WEIGHT = 1e-4

    # note: we want to maximize the entropy
    # so we gradient descent the -entropy
    # Original AlphaStar pseudocode is wrong
    # AlphaStar: loss_ent = entropy_loss(trajectories.behavior_logits, trajectories.masks)
    loss_ent = ENT_WEIGHT * (- entropy_loss_for_all_arguments(target_logits, trajectories, target_select_units_num, target_entity_num))

    #print("stop", len(stop))
    loss_all = loss_actor_critic + loss_upgo  # + loss_kl + loss_ent

    print("loss_actor_critic:", loss_actor_critic) if 1 else None
    print("loss_upgo:", loss_upgo) if 1 else None
    print("loss_kl:", loss_kl) if 1 else None
    print("loss_ent:", loss_ent) if 1 else None
    print("loss_all:", loss_all) if 1 else None

    return loss_all


def test():

    pass
