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
    'action_type',  # Action taken. need 'repeat'?
    'delay',
    'arguments',
    'queued',
    'selected_units',
    'target_unit',
    'target_location',
]

Mask = collections.namedtuple('Mask', ACTION_FIELDS)

# below now only consider action_type now
# baseline are also all zeros
# TODO, change to a more right implementation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def filter_by(action_fields, target):
    if action_fields == 'action_type':
        return target.action_type
    elif action_fields == 'delay':
        return target.delay
    elif action_fields == 'queue':
        return target.queue
    elif action_fields == 'units':
        return target.units
    elif action_fields == 'target_unit':
        return target.target_unit
    elif action_fields == 'target_location':
        return target.target_location


def filter_by_for_lists(action_fields, target_list):

    return torch.cat([getattr(b, action_fields) for a in target_list for b in a], dim=0)


def filter_by_for_masks(action_fields, target_mask):
    index_list = ['action_type', 'delay', 'queue', 'units', 'target_unit', 'target_location']
    index = index_list.index(action_fields)

    mask = torch.tensor(target_mask, device=device)
    mask = mask[:, :, index]

    return mask


def compute_over_actions(f, *args):
    """Runs f over all elements in the lists composing *args.

    Autoregressive actions are composed of many logits. We run losses functions
    over all sets of logits.
    """

    '''
    # show the middle results
    for a in zip(*args):
        print("a:", a)
        r = f(*a)
        print("r:", r)
    '''

    return sum(f(*a) for a in zip(*args))


def mergeArgsActionLogits(list_args_action_logits):
    l = [i.toList() for i in list_args_action_logits]
    a = [torch.cat(z, dim=0) for z in zip(*l)]
    b = [t.reshape(t.shape[0], -1) for t in a]

    return ArgsActionLogits(*b)


def entropy_loss_for_all_arguments(policy_logits, target_select_units_num, masks):
    """Computes the entropy loss for a set of logits.

    Args:
      policy_logits: namedtuple of the logits for each policy argument.
        Each shape is [..., N_i].
      masks: The masks. Each shape is policy_logits.shape[:-1].
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """

    index_list = ['action_type', 'delay', 'queue', 'units', 'target_unit', 'target_location']
    masks = torch.tensor(masks, device=device)

    entropy_list = []
    for x in index_list:    

        logits = getattr(policy_logits, x) 
        print("x name:", x) if debug else None
        print("logits.shape:", logits.shape) if debug else None

        if x == "target_unit":
            # remove the axis 2
            logits = logits.squeeze(dim=1)

        logits = logits.reshape(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        print("logits.shape:", logits.shape) if debug else None

        logits = logits.transpose(0, 1)
        print("logits.shape:", logits.shape) if debug else None

        # shape to be [seq_batch_size, channel_size]
        logits = logits.reshape(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))
        print("logits.shape:", logits.shape) if debug else None

        if x == "units":
            logits = logits.reshape(AHP.sequence_length * AHP.batch_size * AHP.max_selected, AHP.max_entities)
        if x == "target_location":
            logits = logits.reshape(AHP.sequence_length * AHP.batch_size, SCHP.world_size * SCHP.world_size)
        print("logits.shape:", logits.shape) if debug else None

        i = index_list.index(x)

        # shape to be [seq_size, batch_size, 1]
        mask = masks[:, :, i]

        # shape to be [seq_batch_size, 1]
        mask = mask.reshape(-1, 1)
        if x == "units":
            mask = [mask] * AHP.max_selected
            mask = torch.cat(mask, dim=0)
        print("mask.shape:", mask.shape) if debug else None

        entropy_item = RA.entropy(logits, mask)
        print("entropy_item:", entropy_item) if debug else None

        entropy_list.append(entropy_item)

    return torch.mean(torch.cat(entropy_list, axis=0))


def human_policy_kl_loss(student_logits, teacher_logits, target_select_units_num, action_type_kl_cost):
    """Computes the KL loss to the human policy.

    Args:
      trajectories: The trajectories.
      kl_cost: A float; the weighting to apply to the KL cost to the human policy.
      action_type_kl_cost: Additional cost applied to action_types for
        conditioned policies.
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """
    # student_logits: list of ArgsActionLogits
    action_type_loss = RA.kl(student_logits, teacher_logits, 1)

    kl_loss = action_type_kl_cost * torch.mean(action_type_loss)
    print("kl_loss:", kl_loss) if debug else None

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


def policy_gradient_loss(logits, actions, advantages, mask):
    """Helper function for computing policy gradient loss for UPGO and v-trace."""

    # logits: shape [BATCH_SIZE, CLASS_SIZE]
    # actions: shape [BATCH_SIZE]
    # advantages: shape [BATCH_SIZE]
    # mask: shape [BATCH_SIZE]
    action_log_prob = RA.log_prob(actions, logits, reduction="none")
    print("action_log_prob:", action_log_prob) if debug else None
    print("action_log_prob.shape:", action_log_prob.shape) if debug else None

    # advantages = stop_gradient(advantages)
    advantages = advantages.clone().detach()
    print("advantages:", advantages) if debug else None
    print("advantages.shape:", advantages.shape) if debug else None

    print("mask:", mask) if debug else None
    print("mask.shape:", mask.shape) if debug else None

    results = mask * advantages * action_log_prob
    print("results:", results) if debug else None
    print("results.shape:", results.shape) if debug else None

    outlier_remove = True
    if outlier_remove:
        outlier_mask = (torch.abs(results) >= 1e5)
        results = results * ~outlier_mask
    else:
        outlier_mask = (torch.abs(results) >= 1e5)

        if outlier_mask.any() > 0:

            print("outlier_mask:", outlier_mask.nonzero(as_tuple=True)) if 1 else None
            index = outlier_mask.nonzero(as_tuple=True)[0]

            print("actions[index]:", actions[index]) if 1 else None
            print("logits[index]:", logits[index]) if 1 else None
            print("advantages[index]:", advantages[index]) if 1 else None
            print("mask[index]:", mask[index]) if 1 else None
            print("results[index]:", results[index]) if 1 else None

            results = results.reshape(AHP.sequence_length - 1, AHP.batch_size, AHP.max_selected)

            print("results:", results) if 1 else None
            print("results.shape:", results.shape) if 1 else None

            print("action_log_prob:", action_log_prob) if 1 else None
            print("action_log_prob.shape:", action_log_prob.shape) if 1 else None

            stop()

    # note, we should do policy ascent on the results
    # which means if we use policy descent, we should add a "-" sign for results
    loss = -results

    return loss


def vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                   action_fields, target_select_units_num=None):
    # Remove last timestep from trajectories and baselines.
    print("action_fields", action_fields) if debug else None

    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories))

    sequence_length = rewards.shape[0]
    batch_size = rewards.shape[1]

    rewards = rewards[:-1]
    values = baselines[:-1]

    # Filter for only the relevant actions/logits/masks.
    target_logits = filter_by(action_fields, target_logits)
    print("target_logits", target_logits) if debug else None
    print("target_logits.shape", target_logits.shape) if debug else None

    # shape: [batch_seq_size x action_size]
    split_target_logits = target_logits
    action_size = tuple(list(target_logits.shape[1:]))  # from the 3rd dim, it is action dim, may be [S] or [C, S] or [H, W]

    # shape: [batch_size x seq_size x action_size]
    split_target_logits = split_target_logits.reshape(batch_size, sequence_length, *action_size)
    # shape: [seq_size x batch_size x action_size]
    split_target_logits = torch.transpose(split_target_logits, 0, 1)
    # shape: [new_seq_size x batch_size x action_size]
    split_target_logits = split_target_logits[:-1]
    # shape: [seq_batch_size x action_size]
    split_target_logits = split_target_logits.reshape(-1, *action_size)

    if target_select_units_num is not None:
        target_select_units_num = target_select_units_num.reshape(batch_size, sequence_length, -1)
        target_select_units_num = torch.transpose(target_select_units_num, 0, 1)
        target_select_units_num = target_select_units_num[:-1]
        target_select_units_num = target_select_units_num.reshape((sequence_length - 1) * batch_size, -1)
        print("target_select_units_num", target_select_units_num) if debug else None
        print("target_select_units_num.shape", target_select_units_num.shape) if debug else None

    target_logits = split_target_logits
    seqbatch_shape = target_logits.shape[0]
    print("target_logits", target_logits) if debug else None
    print("target_logits.shape", target_logits.shape) if debug else None

    behavior_logits = filter_by_for_lists(action_fields, trajectories.behavior_logits)
    print("behavior_logits", behavior_logits) if debug else None
    print("behavior_logits.shape", behavior_logits.shape) if debug else None

    actions = filter_by_for_lists(action_fields, trajectories.action)
    print("actions", actions) if debug else None
    print("actions.shape", actions.shape) if debug else None

    selected_mask = None
    if action_fields == 'units' or action_fields == 'target_unit':
        seqbatch_unit_shape = target_logits.shape[0:2]
        select_max_size = target_logits.shape[1]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        player_select_units_num = torch.tensor(trajectories.player_select_units_num, device=target_logits.device).reshape(-1)
        print("player_select_units_num", player_select_units_num) if debug else None
        print("player_select_units_num.shape", player_select_units_num.shape) if debug else None

        # when computing logits, we consider the EOF
        player_select_units_num = player_select_units_num + 1

        selected_mask = torch.arange(select_max_size, device=target_logits.device).float()
        selected_mask = selected_mask.repeat(seqbatch_shape, 1)
        selected_mask = selected_mask < player_select_units_num.unsqueeze(dim=1)

        assert selected_mask.dtype == torch.bool

        if target_select_units_num is not None:
            # shape [seqbatch_shape, 1]
            target_select_units_num = target_select_units_num + 1

            target_selected_mask = torch.arange(select_max_size, device=target_logits.device).float()
            target_selected_mask = target_selected_mask.repeat(seqbatch_shape, 1)
            target_selected_mask = target_selected_mask < target_select_units_num

            assert target_selected_mask.dtype == torch.bool

    elif action_fields == 'target_location':
        target_logits = target_logits.reshape(target_logits.shape[0], -1)
        behavior_logits = behavior_logits.reshape(behavior_logits.shape[0], -1)

        actions_2 = torch.zeros(behavior_logits.shape[0], 1, dtype=torch.int64, device=behavior_logits.device)
        print("actions_2.shape", actions_2.shape) if debug else None

        for i, pos in enumerate(actions):
            # note: for pos, the first index is x, the seconde index is y
            # however, for the matrix, the first index is y (row), and the second index is x (col)
            x = pos[0]
            assert x >= 0
            assert x < SCHP.world_size
            y = pos[1]
            assert y >= 0
            assert y < SCHP.world_size
            index = SCHP.world_size * y + x
            actions_2[i][0] = index

        actions = actions_2
        print("actions_2.shape", actions_2.shape) if debug else None

    # Compute and return the v-trace policy gradient loss for the relevant subset of logits.
    clipped_rhos = RA.compute_importance_weights(behavior_logits, target_logits, actions)

    if action_fields == 'units' or action_fields == 'target_unit':
        clipped_rhos = clipped_rhos.reshape(seqbatch_unit_shape, -1)
        clipped_rhos = torch.mean(clipped_rhos, dim=-1)

    # To make the clipped_rhos shape to be [T-1, B]
    clipped_rhos = clipped_rhos.reshape(rewards.shape)
    print("clipped_rhos", clipped_rhos) if debug else None
    print("clipped_rhos.shape", clipped_rhos.shape) if debug else None

    discounts = ~np.array(trajectories.is_final, dtype=np.bool)
    discounts = torch.tensor(discounts, dtype=torch.float32, device=device)

    # we implement the vtrace_advantages
    # vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):
    weighted_advantage = RA.vtrace_advantages(clipped_rhos, rewards,
                                              discounts, values,
                                              baselines[-1])

    masks = filter_by_for_masks(action_fields, trajectories.masks)
    print("filtered masks", masks) if debug else None

    # AlphaStar: weighted_advantage = [weighted_advantage] * len(target_logits)
    # mAS: the weighted_advantage is already been unfolded, so we don't need the line
    # we need the pg_advantages of the VTrace_returns, which is in ths index of 1
    weighted_advantage = weighted_advantage[1]
    print("weighted_advantage", weighted_advantage) if debug else None

    # here we should reshape the target_logits and actions back to [T-1, B, C] size for computing policy gradient
    if action_fields == 'units':
        target_logits = target_logits.reshape(seqbatch_shape * AHP.max_selected, -1)
        actions = actions.reshape(seqbatch_shape * AHP.max_selected, -1)

        weighted_advantage = torch.cat([weighted_advantage] * AHP.max_selected, dim=1)
        masks = torch.cat([masks] * AHP.max_selected, dim=1)
        masks = masks.reshape(-1) * selected_mask.reshape(-1)

        if target_selected_mask is not None:
            masks = masks * target_selected_mask.reshape(-1)

    else:
        target_logits = target_logits.reshape(seqbatch_shape, -1)
        actions = actions.reshape(seqbatch_shape, -1)

    print("masks", masks) if debug else None
    print("masks.shape", masks.shape) if debug else None

    # result = compute_over_actions(policy_gradient_loss, target_logits,
    #                               actions, weighted_advantage, masks)

    result = policy_gradient_loss(target_logits, actions, weighted_advantage.reshape(-1), masks.reshape(-1))

    if action_fields == 'units':
        result = result.reshape(-1, AHP.max_selected)
        result = torch.mean(result, dim=-1)

    print("result", result) if debug else None
    print("result.shape", result.shape) if debug else None

    # note: we change back to use only result 
    return result


def split_upgo_loss(target_logits, target_select_units_num, baselines, trajectories):
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
    loss = sum_upgo_loss(target_logits, values, trajectories, returns, target_select_units_num)

    return loss


def sum_upgo_loss(target_logits, values, trajectories, returns, target_select_units_num):
    loss = 0.

    action_type_loss = upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'action_type')
    loss += action_type_loss
    print('action_type_loss', action_type_loss.mean()) if debug else None

    delay_loss = upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'delay')
    loss += delay_loss * 0  # dont' use delay loss now
    print('delay_loss', delay_loss.mean()) if 0 else None

    queue_loss = upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'queue')
    loss += queue_loss
    print('queue_loss', queue_loss.mean()) if debug else None

    units_loss = upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'units', target_select_units_num)
    loss += units_loss
    print('units_loss', units_loss.mean()) if debug else None

    target_unit_loss = upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'target_unit')
    loss += target_unit_loss
    print('target_unit_loss', target_unit_loss.mean()) if debug else None

    target_location_loss = upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'target_location')
    loss += target_location_loss
    print('target_location_loss', target_location_loss.mean()) if debug else None

    loss_upgo = loss.mean()
    print('loss_upgo', loss_upgo) if debug else None
    return loss_upgo


def split_vtrace_pg_loss(target_logits, target_select_units_num, baselines, rewards, trajectories):
    """Computes the split v-trace policy gradient loss.

    We compute the policy loss (and therefore update, via autodiff) separately for
    the action_type, delay, and arguments. Each of these component losses are
    weighted equally.

    Paper description:
      When applying V-trace to the policy in large action spaces, 
      the off-policy corrections truncate the trace early; 
      to mitigate this problem, we assume independence between the action type, 
      delay, and all other arguments, and so update the components 
      of the policy separately.
    """

    # The action_type argument, delay, and all other arguments are separately updated 
    # using a separate ("split") VTrace Actor-Critic losses. 
    # The weighting of these updates will be considered 1.0.

    loss = 0.

    action_type_loss = vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'action_type')
    loss += action_type_loss
    print('action_type_loss', action_type_loss.mean()) if debug else None

    if True:
        delay_loss = vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'delay')
        loss += delay_loss * 0
        print('delay_loss', delay_loss.mean()) if debug else None

        # note: here we use queue, units, target_unit and target_location to replace the single arguments
        queue_loss = vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'queue')
        loss += queue_loss
        print('queue_loss', queue_loss.mean()) if debug else None

        units_loss = vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'units', target_select_units_num)
        loss += units_loss
        print('units_loss', units_loss.mean()) if debug else None

        target_unit_loss = vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'target_unit')
        loss += target_unit_loss
        print('target_unit_loss', target_unit_loss.mean()) if debug else None

        target_location_loss = vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'target_location')
        loss += target_location_loss
        print('target_location_loss', target_location_loss.mean()) if debug else None

    sum_vtrace_pg_loss = loss.mean()
    print('sum_vtrace_pg_loss', sum_vtrace_pg_loss) if debug else None

    return sum_vtrace_pg_loss


def upgo_loss_like_vtrace(target_logits, values, trajectories, returns, action_fields, target_select_units_num=None):
    print("action_fields", action_fields) if debug else None

    # Filter for only the relevant actions/logits/masks.
    target_logits = filter_by(action_fields, target_logits)

    # shape: [batch_seq_size x action_size]
    split_target_logits = target_logits
    action_size = tuple(list(target_logits.shape[1:]))  # from the 3rd dim, it is action dim, may be [S] or [C, S] or [H, W]

    batch_size = AHP.batch_size
    sequence_length = AHP.sequence_length

    # shape: [batch_size x seq_size x action_size]
    split_target_logits = split_target_logits.reshape(batch_size, sequence_length, *action_size)

    # shape: [seq_size x batch_size x action_size]
    split_target_logits = torch.transpose(split_target_logits, 0, 1)

    # shape: [new_seq_size x batch_size x action_size]
    split_target_logits = split_target_logits[:-1]

    # shape: [seq_batch_size x action_size]
    split_target_logits = split_target_logits.reshape(-1, *action_size)

    if target_select_units_num is not None:
        target_select_units_num = target_select_units_num.reshape(batch_size, sequence_length, -1)
        target_select_units_num = torch.transpose(target_select_units_num, 0, 1)
        target_select_units_num = target_select_units_num[:-1]
        target_select_units_num = target_select_units_num.reshape((sequence_length - 1) * batch_size, -1)
        print("target_select_units_num", target_select_units_num) if debug else None
        print("target_select_units_num.shape", target_select_units_num.shape) if debug else None

    target_logits = split_target_logits
    seqbatch_shape = target_logits.shape[0]

    behavior_logits = filter_by_for_lists(action_fields, trajectories.behavior_logits)
    actions = filter_by_for_lists(action_fields, trajectories.action)

    selected_mask = None
    if action_fields == 'units' or action_fields == 'target_unit':
        seqbatch_unit_shape = target_logits.shape[0:2]
        select_max_size = target_logits.shape[1]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        player_select_units_num = torch.tensor(trajectories.player_select_units_num, device=target_logits.device).reshape(-1)
        print("player_select_units_num", player_select_units_num) if debug else None
        print("player_select_units_num.shape", player_select_units_num.shape) if debug else None

        # when computing logits, we consider the EOF
        # shape [seqbatch_shape]
        player_select_units_num = player_select_units_num + 1

        selected_mask = torch.arange(select_max_size, device=target_logits.device).float()
        selected_mask = selected_mask.repeat(seqbatch_shape, 1)
        selected_mask = selected_mask < player_select_units_num.unsqueeze(dim=1)
        print("selected_mask:", selected_mask) if debug else None
        print("selected_mask.shape:", selected_mask.shape) if debug else None

        assert selected_mask.dtype == torch.bool

        if target_select_units_num is not None:
            # shape [seqbatch_shape, 1]
            target_select_units_num = target_select_units_num + 1

            target_selected_mask = torch.arange(select_max_size, device=target_logits.device).float()
            target_selected_mask = target_selected_mask.repeat(seqbatch_shape, 1)
            target_selected_mask = target_selected_mask < target_select_units_num

            assert target_selected_mask.dtype == torch.bool

    if action_fields == 'target_location':
        target_logits = target_logits.reshape(target_logits.shape[0], -1)
        behavior_logits = behavior_logits.reshape(behavior_logits.shape[0], -1)

        actions_2 = torch.zeros(behavior_logits.shape[0], 1, dtype=torch.int64, device=device)

        for i, pos in enumerate(actions):
            # note: for pos, the first index is x, the seconde index is y
            # however, for the matrix, the first index is y (row), and the second index is x (col)
            x = pos[0]
            assert x >= 0
            assert x < SCHP.world_size
            y = pos[1]
            assert y >= 0
            assert y < SCHP.world_size
            index = SCHP.world_size * y + x
            actions_2[i][0] = index

        actions = actions_2
        print("actions_2.shape", actions_2.shape) if debug else None

    clipped_rhos = RA.compute_importance_weights(behavior_logits,
                                                 target_logits,
                                                 actions)

    if action_fields == 'units' or action_fields == 'target_unit':
        clipped_rhos = clipped_rhos.reshape(seqbatch_unit_shape, -1)
        clipped_rhos = torch.mean(clipped_rhos, dim=-1)

    # To make the clipped_rhos shape to be [T-1, B]
    clipped_rhos = clipped_rhos.reshape(values.shape)
    weighted_advantage = (returns - values) * clipped_rhos
    print("weighted_advantage", weighted_advantage) if debug else None

    masks = filter_by_for_masks(action_fields, trajectories.masks)

    # here we should reshape the target_logits and actions back to [T-1, B, C] size for computing policy gradient
    if action_fields == 'units':
        target_logits = target_logits.reshape(seqbatch_shape * AHP.max_selected, -1)
        actions = actions.reshape(seqbatch_shape * AHP.max_selected, -1)

        weighted_advantage = torch.cat([weighted_advantage] * AHP.max_selected, dim=1)
        masks = torch.cat([masks] * AHP.max_selected, dim=1)
        masks = masks.reshape(-1) * selected_mask.reshape(-1)

        if target_selected_mask is not None:
            masks = masks * target_selected_mask.reshape(-1)
    else:
        target_logits = target_logits.reshape(seqbatch_shape, -1)
        actions = actions.reshape(seqbatch_shape, -1)

    # result = compute_over_actions(policy_gradient_loss, target_logits,
    #                               actions, weighted_advantage, masks)

    result = policy_gradient_loss(target_logits, actions, weighted_advantage.reshape(-1), masks.reshape(-1))

    if action_fields == 'units':
        result = result.reshape(-1, AHP.max_selected)
        result = torch.mean(result, dim=-1)

    print("result", result) if debug else None
    print("result.shape", result.shape) if debug else None

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


def loss_function(agent, trajectories):
    """Computes the loss of trajectories given weights."""
    # All ALL_CAPS variables are constants.

    # QUESTIOM: The trajectories already have behavior_logits, why is the need
    # to calculate the target_logits?
    # Answer: To calculate the importance ratio = target_logits / behavior_logits
    # trajectories shape: list of trajectory
    # target_logits: ArgsActionLogits
    target_logits, baselines, target_select_units_num = agent.unroll(trajectories)

    the_action_type = target_logits.action_type
    print("the_action_type.shape:", the_action_type.shape) if debug else None
    print("baselines.shape:", baselines.shape) if debug else None

    # note, we change the structure of the trajectories
    # note, the size is all list
    # shape: [batch_size x dict_name x seq_size]
    trajectories = RU.stack_namedtuple(trajectories) 

    # shape: [dict_name x batch_size x seq_size]
    print("trajectories.reward", trajectories.reward) if debug else None   

    # shape: [dict_name x batch_size x seq_size]
    trajectories = RU.namedtuple_zip(trajectories) 

    # shape: [dict_name x seq_size x batch_size]
    print("trajectories.reward", trajectories.reward) if debug else None   

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

            # we add the split_vtrace_pg_loss
            pg_loss = split_vtrace_pg_loss(target_logits, target_select_units_num, baseline, rewards, trajectories)
            print("pg_loss:", pg_loss) if debug else None

            loss_actor_critic += (pg_cost * pg_loss)

        reward_index += 1

    # Note: upgo_loss has only one baseline which is just for winloss 
    # AlphaStar: loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines.winloss_baseline, trajectories)
    UPGO_WEIGHT = 1.0
    loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, target_select_units_num, baselines[0], trajectories)

    # Distillation Loss:
    # There is an distillation loss with weight 2e-3 on all action arguments, 
    # to match the output logits of the fine-tuned supervised policy 
    # which has been given the same observation.
    # If the trajectory was conditioned on `cumulative_statistics`, there is an additional 
    # distillation loss of weight 1e-1 on the action type logits 
    # for the first four minutes of the game.
    # Thus KL_COST = 2e-3
    # and ACTION_TYPE_KL_COST = 1e-1
    KL_COST = 2e-3
    ACTION_TYPE_KL_COST = 1e-1

    # We change to use the teacher logits
    print("trajectories.teacher_logits", trajectories.teacher_logits) if debug else None

    teacher_logits_action_type = filter_by_for_lists("action_type", trajectories.teacher_logits)
    print("target_logits.action_type.shape", target_logits.action_type.shape) if debug else None
    print("teacher_logits_action_type.shape", teacher_logits_action_type.shape) if debug else None

    # TODO: for all arguments
    # loss_kl = human_policy_kl_loss(target_logits.action_type, teacher_logits_action_type, 
    #                                target_select_units_num, ACTION_TYPE_KL_COST)

    # Entropy Loss:
    # There is an entropy loss with weight 1e-4 on all action arguments, 
    # masked by which arguments are possible for a given action type.
    # Thus ENT_WEIGHT = 1e-4
    ENT_WEIGHT = 1e-4

    # important! the behavior_logits should not be used due to the pytorch backward rule
    # because the values to compute it (behavior_logits) are lost, so we should forward it,
    # if we don't do so, it may cause runtime error! (like the inplace version problem)
    # loss_ent = ENT_WEIGHT * entropy_loss(trajectories.behavior_logits, trajectories.masks)
    print("target_logits", target_logits) if debug else None   
    print("trajectories.behavior_logits", trajectories.behavior_logits) if debug else None
    print("trajectories.masks", trajectories.masks) if debug else None

    # note: we want to maximize the entropy
    # so we gradient descent the -entropy
    # Original AlphaStar pseudocode is wrong
    # AlphaStar: loss_ent = entropy_loss(trajectories.behavior_logits, trajectories.masks)
    # loss_ent = ENT_WEIGHT * (- entropy_loss_for_all_arguments(target_logits, target_select_units_num, trajectories.masks))

    #print("stop", len(stop))
    loss_all = loss_actor_critic  # + loss_upgo  # + loss_kl + loss_ent

    print("loss_actor_critic:", loss_actor_critic) if 1 else None
    print("loss_upgo:", loss_upgo) if 0 else None
    print("loss_kl:", loss_kl) if 0 else None
    print("loss_ent:", loss_ent) if 0 else None
    print("loss_all:", loss_all) if 0 else None

    return loss_all


def test():

    pass
