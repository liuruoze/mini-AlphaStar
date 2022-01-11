#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for RL losses."

# modified from AlphaStar pseudo-code
import traceback
import collections
import itertools
import gc

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
FIELDS_WEIGHT = [1, 0, 1, 1, 1, 1]

SELECTED_UNITS_PLUS_ONE = False


def filter_by_for_lists(action_fields, target_list, device):
    return torch.cat([getattr(b, action_fields).to(device) for a in target_list for b in a], dim=0)


def filter_by_for_masks(action_fields, target_mask, device):
    return torch.tensor(target_mask, device=device)[:, :, ACTION_FIELDS.index(action_fields)]


def entropy_loss(target_logits, trajectories, selected_mask, entity_mask):
    """Computes the entropy loss for a set of logits."""

    device = target_logits.action_type.device
    mask = torch.tensor(trajectories.masks, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)
    [selected_mask, entity_mask, unit_type_entity_mask] = [x.view(-1, x.shape[-1]) for x in [selected_mask, entity_mask, unit_type_entity_mask]]

    loss = 0 
    for i, field in enumerate(ACTION_FIELDS):  
        x = get_kl_or_entropy(target_logits, field, RA.entropy, mask, selected_mask, entity_mask, unit_type_entity_mask)
        loss = loss + x * FIELDS_WEIGHT[i]
        del x

    del mask, unit_type_entity_mask, selected_mask, entity_mask

    return loss


def human_policy_kl_loss(target_logits, trajectories, selected_mask, entity_mask):
    """Computes the KL loss to the human policy."""

    device = target_logits.action_type.device
    mask = torch.tensor(trajectories.masks, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)
    [selected_mask, entity_mask, unit_type_entity_mask] = [x.view(-1, x.shape[-1]) for x in [selected_mask, entity_mask, unit_type_entity_mask]]

    loss = 0  
    for i, field in enumerate(ACTION_FIELDS):
        t_logits = filter_by_for_lists(field, trajectories.teacher_logits, device) 
        x = get_kl_or_entropy(target_logits, field, RA.kl, mask, selected_mask, entity_mask, unit_type_entity_mask, t_logits)
        loss = loss + x * FIELDS_WEIGHT[i]
        del x, t_logits

    del mask, unit_type_entity_mask, selected_mask, entity_mask

    return loss


def get_kl_or_entropy(target_logits, field, func, mask, selected_mask, entity_mask, unit_type_entity_mask, t_logits=None):
    logits = getattr(target_logits, field)
    logits = logits.view(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))
    all_logits = [logits]
    if t_logits is not None:
        all_logits.append(t_logits)

    if field == "units":
        all_logits = [logit.view(AHP.sequence_length * AHP.batch_size, AHP.max_selected, AHP.max_entities) for logit in all_logits]
    elif field == "target_unit":
        all_logits = [logit.view(AHP.sequence_length * AHP.batch_size * 1, AHP.max_entities) for logit in all_logits]     
    elif field == "target_location":
        all_logits = [logit.view(AHP.sequence_length * AHP.batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]

    if field == "units":
        x = func(all_logits, selected_mask, entity_mask.unsqueeze(-2), unit_type_entity_mask.unsqueeze(-2))
        x = x.sum(dim=-2)
    else:
        if field == "target_unit":
            x = func(all_logits, entity_mask=entity_mask, unit_type_entity_mask=unit_type_entity_mask)
        else:
            x = func(all_logits)
    x = x.sum(dim=-1)
    if t_logits is None:
        x = x / torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32, device=logits.device))     # Normalize by actions available.

    mask = mask[:, :, ACTION_FIELDS.index(field)].view(-1, 1)
    x = torch.mean(x * mask)

    del logits, t_logits, all_logits, mask

    return x


def human_policy_kl_loss_action(target_logits, trajectories):

    device = target_logits.action_type.device
    seconds = torch.tensor(trajectories.game_loop, device=device).view(AHP.sequence_length * AHP.batch_size) / 22.4
    flag = seconds < (4 * 60)  # the first 4 minutes
    i = 0  # index for action_type
    field = ACTION_FIELDS[i]  # action_type
    mask = torch.tensor(trajectories.masks, device=device)[:, :, i].view(-1, 1)

    logits = getattr(target_logits, field) 
    logits = logits.view(AHP.sequence_length * AHP.batch_size, *tuple(logits.shape[2:]))
    t_logits = filter_by_for_lists(field, trajectories.teacher_logits, device) 

    kl = RA.kl([logits, t_logits]).sum(dim=-1) * flag
    kl = torch.mean(kl * mask)

    del seconds, flag, mask, logits, t_logits

    return kl


def td_lambda_loss(baselines, rewards, trajectories, device): 

    discounts = ~np.array(trajectories.is_final[:-1], dtype=np.bool)  # note, use '~' must ensure the type is bool
    discounts = torch.tensor(discounts, device=device)

    baselines = baselines
    rewards = rewards[:-1]  # rewards should be T_0 -> T_{n-1}

    with torch.no_grad():
        returns = RA.lambda_returns(baselines[1:], rewards, discounts, lambdas=0.8)

    print('returns', returns) if 1 else None
    print('baselines', baselines) if 1 else None

    result = returns - baselines[:-1]
    print('result', result) if 1 else None

    del discounts, baselines, rewards, returns

    td_lambda_loss = 0.5 * torch.mean(torch.square(result))
    print('td_lambda_loss', td_lambda_loss.item()) if 1 else None

    return td_lambda_loss


def get_baseline_hyperparameters():
    # Accroding to detailed_architecture.txt, baselines contains the following 5 items:
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
    for i, field in enumerate(ACTION_FIELDS):  
        logits = getattr(target_logits, field) 
        #logits = logits.view(AHP.batch_size, AHP.sequence_length, *tuple(logits.shape[1:]))
        logits = logits.transpose(0, 1).contiguous()
        logits = logits.view(AHP.sequence_length, AHP.batch_size, *tuple(logits.shape[2:]))
        setattr(target_logits, field, logits)

    del logits
    return target_logits


def transpose_baselines(baseline_list):
    baselines = [baseline.view(AHP.batch_size, AHP.sequence_length).transpose(0, 1).contiguous() for baseline in baseline_list]
    return baselines


def transpose_sth(x):
    x = x.view(AHP.batch_size, AHP.sequence_length).transpose(0, 1).contiguous()
    return x


def get_useful_masks(select_units_num, entity_num, device):
    selected_mask = torch.arange(AHP.max_selected, device=device).float()
    selected_mask = selected_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    selected_mask = selected_mask < (select_units_num + 1).reshape(-1).unsqueeze(dim=-1)

    entity_mask = torch.arange(AHP.max_entities, device=device).float()
    entity_mask = entity_mask.repeat(AHP.sequence_length * AHP.batch_size, 1)
    entity_mask = entity_mask < entity_num.reshape(-1).unsqueeze(dim=-1)

    selected_mask = selected_mask.reshape(AHP.sequence_length, AHP.batch_size, AHP.max_selected)
    entity_mask = entity_mask.reshape(AHP.sequence_length, AHP.batch_size, AHP.max_entities)

    del select_units_num, entity_num

    return selected_mask, entity_mask


def sum_upgo_loss(target_logits_all, trajectories, baselines, selected_mask, entity_mask, device):
    """Computes the split upgo policy gradient loss."""
    print('sum_upgo_loss') if debug else None

    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories))
    discounts = torch.tensor(~np.array(trajectories.is_final, dtype=np.bool), dtype=torch.float32, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)
    reward = torch.tensor(np.array(trajectories.reward), dtype=torch.float32, device=device)

    values = baselines[:-1]
    returns = RA.upgo_returns(values, reward, discounts, baselines[-1])
    [selected_mask, entity_mask] = [a[:-1] for a in [selected_mask, entity_mask]]
    mask_provided = [selected_mask, entity_mask, unit_type_entity_mask]

    fields_weight = [1, 0, 1, 0, 1, 1]

    loss = 0.
    for i, field in enumerate(ACTION_FIELDS):
        target_log_prob, clipped_rhos, masks = get_logprob_and_rhos(target_logits_all, field, trajectories, mask_provided)
        with torch.no_grad():
            weighted_advantage = ((returns - values) * clipped_rhos).reshape(-1)
        loss_field = (-target_log_prob) * weighted_advantage * masks.reshape(-1)
        loss_field = loss_field.mean()
        print('field', field, 'loss_val', loss_field.item()) if debug else None

        loss = loss + loss_field * fields_weight[i]
        del loss_field, weighted_advantage, target_log_prob, clipped_rhos, masks

    del trajectories, discounts, unit_type_entity_mask, reward
    del values, returns, selected_mask, entity_mask, mask_provided

    return loss


def sum_vtrace_loss(target_logits_all, trajectories, baselines, rewards, selected_mask, entity_mask, device):
    """Computes the split v-trace policy gradient loss."""
    print('sum_vtrace_pg_loss') if debug else None

    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories))
    discounts = torch.tensor(~np.array(trajectories.is_final, dtype=np.bool), dtype=torch.float32, device=device)
    unit_type_entity_mask = torch.tensor(np.array(trajectories.unit_type_entity_mask), dtype=torch.float32, device=device)

    values = baselines[:-1]
    [rewards, selected_mask, entity_mask] = [a[:-1] for a in [rewards, selected_mask, entity_mask]]
    mask_provided = [selected_mask, entity_mask, unit_type_entity_mask]

    fields_weight = [1, 0, 1, 0, 1, 1]

    loss = 0.
    for i, field in enumerate(ACTION_FIELDS):
        target_log_prob, clipped_rhos, masks = get_logprob_and_rhos(target_logits_all, field, trajectories, mask_provided)
        with torch.no_grad():
            weighted_advantage = RA.vtrace_advantages(clipped_rhos, rewards, discounts, values, baselines[-1])[1].reshape(-1)
        loss_field = (-target_log_prob) * weighted_advantage * masks.reshape(-1)
        loss_field = loss_field.mean()
        print('field', field, 'loss_val', loss_field.item()) if debug else None

        loss = loss + loss_field * fields_weight[i]
        del loss_field, weighted_advantage, target_log_prob, clipped_rhos, masks

    del trajectories, discounts, unit_type_entity_mask, rewards
    del values, selected_mask, entity_mask, mask_provided

    return loss


def get_logprob_and_rhos(target_logits_all, field, trajectories, mask_provided):
    target_logits = getattr(target_logits_all, field)[:-1]
    device = target_logits.device
    [selected_mask, entity_mask, unit_type_entity_mask] = mask_provided
    mask_used = [None, None, None] 

    [sequence_length, batch_size] = target_logits.shape[0:2]
    seqbatch_shape = sequence_length * batch_size
    target_logits = target_logits.view(seqbatch_shape, *tuple(target_logits.shape[2:]))

    behavior_logits = filter_by_for_lists(field, trajectories.behavior_logits, device)
    actions = filter_by_for_lists(field, trajectories.action, device)
    masks = filter_by_for_masks(field, trajectories.masks, device)

    all_logits = [target_logits, behavior_logits]
    if field == "units":
        all_logits = [logit.reshape(sequence_length * batch_size, AHP.max_selected, AHP.max_entities) for logit in all_logits]
        mask_used = [selected_mask, entity_mask, unit_type_entity_mask]
    elif field == "target_unit":
        all_logits = [logit.reshape(sequence_length * batch_size, 1, AHP.max_entities) for logit in all_logits]
        mask_used = [None, entity_mask, None]
    elif field == "target_location":
        all_logits = [logit.reshape(sequence_length * batch_size, SCHP.world_size * SCHP.world_size) for logit in all_logits]
        actions_tmp = torch.zeros(seqbatch_shape, 1, dtype=torch.int64, device=device)
        for i, pos in enumerate(actions):
            [x, y] = pos
            index = SCHP.world_size * y + x
            actions_tmp[i][0] = index
        actions = actions_tmp

    [target_logits, behavior_logits] = all_logits

    target_log_prob = RA.log_prob(target_logits, actions, mask_used)
    behavior_log_prob = RA.log_prob(behavior_logits, actions, mask_used)
    with torch.no_grad():
        clipped_rhos = RA.compute_cliped_importance_weights(target_log_prob, behavior_log_prob)

    if field == 'units':
        clipped_rhos = clipped_rhos.reshape(seqbatch_shape, -1).sum(dim=-1)
    clipped_rhos = clipped_rhos.reshape(sequence_length, batch_size)

    del behavior_log_prob, target_logits, behavior_logits, all_logits, target_logits_all
    del actions, selected_mask, entity_mask, unit_type_entity_mask

    return target_log_prob, clipped_rhos, masks


def loss_function(agent, trajectories, use_opponent_state=True, no_replay_learn=False):
    """Computes the loss of trajectories given weights."""

    # target_logits: ArgsActionLogits
    target_logits, baselines, target_select_units_num, target_entity_num = agent.rl_unroll(trajectories, use_opponent_state)
    device = target_logits.action_type.device

    # transpose to [seq_size x batch_size x -1]
    target_logits = transpose_target_logits(target_logits)
    baselines = transpose_baselines(baselines)

    # transpose to [seq_size x batch_size x -1]
    target_select_units_num = transpose_sth(target_select_units_num)
    target_entity_num = transpose_sth(target_entity_num)

    # get used masks
    selected_mask, entity_mask = get_useful_masks(target_select_units_num, target_entity_num, device)

    # note, we change the structure of the trajectories
    # shape before: [dict_name x batch_size x seq_size]
    trajectories = RU.stack_namedtuple(trajectories) 

    # shape after: [dict_name x seq_size x batch_size]
    trajectories = RU.namedtuple_zip(trajectories) 

    loss_actor_critic = 0.

    # We use a number of actor-critic losses - one for the winloss baseline, which
    # outputs the probability of victory, and one for each pseudo-reward
    # associated with following the human strategy statistic z.
    BASELINE_COSTS_AND_REWARDS = get_baseline_hyperparameters()

    loss_all = 0.
    loss_dict = {}

    # Vtrace Loss:
    reward_index = 0
    for baseline, costs_and_rewards in zip(baselines, BASELINE_COSTS_AND_REWARDS):
        if no_replay_learn:
            if reward_index != 0:
                break

        vtrace_cost, baseline_cost, reward_name = costs_and_rewards
        print("reward_name:", reward_name) if debug else None

        rewards = PR.compute_pseudoreward(trajectories, reward_name, device)
        print("rewards:", rewards) if 1 else None

        # The action_type argument, delay, and all other arguments are separately updated 
        # using a separate ("split") VTrace Actor-Critic losses. 
        loss_baseline = td_lambda_loss(baseline, rewards, trajectories, device)
        loss_baseline = baseline_cost * loss_baseline
        print("loss_baseline:", loss_baseline) if debug else None

        loss_dict.update({reward_name + "-loss_baseline:": loss_baseline.item()})
        loss_actor_critic += 1 * loss_baseline

        # we add vtrace loss
        loss_vtrace = sum_vtrace_loss(target_logits, trajectories, baseline, rewards, selected_mask, entity_mask, device)
        loss_vtrace = vtrace_cost * loss_vtrace 
        print("loss_vtrace:", loss_vtrace) if debug else None

        loss_dict.update({reward_name + "-loss_vtrace:": loss_vtrace.item()})
        loss_actor_critic += 1 * (loss_vtrace)

        reward_index += 1

        del loss_baseline, loss_vtrace, rewards

    # Upgo Loss:
    # The weighting of these updates will be considered 1.0. action_type, delay, and other arguments are 
    # also similarly separately updated using UPGO, in the same way as the VTrace Actor-Critic loss, with relative weight 1.0. 
    # AlphaStar: loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines.winloss_baseline, trajectories)
    UPGO_WEIGHT = 1.0
    baseline_winloss = baselines[0]
    loss_upgo = UPGO_WEIGHT * sum_upgo_loss(target_logits, trajectories, baseline_winloss, selected_mask, entity_mask, device)
    loss_dict.update({"loss_upgo:": loss_upgo.item()})
    del baselines

    # Distillation Loss:
    # There is an distillation loss with weight 2e-3 on all action arguments, to match the output logits of the fine-tuned supervised policy 
    # which has been given the same observation. If the trajectory was conditioned on `cumulative_statistics`, there is an additional 
    # distillation loss of weight 1e-1 on the action type logits for the first four minutes of the game.
    # Thus ALL_KL_COST = 2e-3 and ACTION_TYPE_KL_COST = 1e-1
    ALL_KL_COST = 2e-3
    ACTION_TYPE_KL_COST = 1e-1

    # for all arguments
    all_kl_loss = human_policy_kl_loss(target_logits, trajectories, selected_mask, entity_mask)
    loss_dict.update({"all_kl_loss:": all_kl_loss.item()})

    action_type_kl_loss = human_policy_kl_loss_action(target_logits, trajectories)
    loss_dict.update({"action_type_kl_loss:": action_type_kl_loss.item()})

    loss_kl = ALL_KL_COST * all_kl_loss + ACTION_TYPE_KL_COST * action_type_kl_loss
    loss_dict.update({"loss_kl:": loss_kl.item()})
    del all_kl_loss, action_type_kl_loss

    # Entropy Loss:
    # There is an entropy loss with weight 1e-4 on all action arguments, masked by which arguments are possible for a given action type.
    # Thus ENT_WEIGHT = 1e-4
    ENT_WEIGHT = 1e-4

    # note: we want to maximize the entropy so we gradient descent the -entropy. Original AlphaStar pseudocode is wrong
    # AlphaStar: loss_ent = entropy_loss(trajectories.behavior_logits, trajectories.masks)
    loss_ent = ENT_WEIGHT * (- entropy_loss(target_logits, trajectories, selected_mask, entity_mask))
    loss_dict.update({"loss_ent:": loss_ent.item()})
    del trajectories, selected_mask, entity_mask, target_logits

    loss_all = loss_actor_critic + loss_ent + loss_kl + loss_upgo  # loss_actor_critic + loss_upgo + loss_kl + loss_ent

    # if False:
    #     print("loss_actor_critic:", loss_actor_critic.item()) if debug else None
    #     print("loss_upgo:", loss_upgo.item()) if debug else None
    #     print("loss_kl:", loss_kl.item()) if debug else None
    #     print("loss_ent:", loss_ent.item()) if debug else None
    #     print("loss_all:", loss_all.item()) if debug else None

    del loss_actor_critic, loss_upgo, loss_kl, loss_ent

    return loss_all, loss_dict


def test():

    pass
