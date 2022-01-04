#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for RL algorithms."

# modified from AlphaStar pseudo-code
import traceback
import collections
import itertools

import numpy as np

import torch
import torch.nn as nn

from alphastarmini.core.rl.rl_utils import Trajectory
from alphastarmini.core.rl.action import ArgsActionLogits

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


def reverse_seq(sequence, sequence_lengths=None):
    """Reverse sequence along dim 0.
    Args:
      sequence: Tensor of shape [T, B, ...].
      sequence_lengths: (optional) tensor of shape [B]. If `None`, only reverse
        along dim 0.
    Returns:
      Tensor of same shape as sequence with dim 0 reversed up to sequence_lengths.
    """
    if sequence_lengths is None:
        return torch.flip(sequence, [0])
    else:
        raise NotImplementedError


def lambda_returns(values_tp1, rewards, discounts, lambdas=0.8):
    """Computes lambda returns.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
    """

    # mAS: we only implment the lambda return version in AlphaStar when lambdas=0.8
    # assert lambdas != 1

    # assert v_tp1 = torch.concat([values[1:, :], torch.unsqueeze(bootstrap_value, 0)], axis=0)
    # `back_prop=False` prevents gradients flowing into values and
    # bootstrap_value, which is what you want when using the bootstrapped
    # lambda-returns in an update as targets for values.
    return multistep_forward_view(
        rewards,
        discounts,
        values_tp1,
        lambdas,
        back_prop=False,
        name="0.8_lambda_returns")


def multistep_forward_view(rewards, pcontinues, state_values, lambda_,
                           back_prop=True, 
                           name="multistep_forward_view_op"):
    """Evaluates complex backups (forward view of eligibility traces).
      ```python
      result[t] = rewards[t] +
          pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t])
      result[last] = rewards[last] + pcontinues[last]*state_values[last]
      ```
      This operation evaluates multistep returns where lambda_ parameter controls
      mixing between full returns and boostrapping.
      ```
    Args:
      rewards: Tensor of shape `[T, B]` containing rewards.
      pcontinues: Tensor of shape `[T, B]` containing discounts.
      state_values: Tensor of shape `[T, B]` containing state values.
      lambda_: Mixing parameter lambda.
          The parameter can either be a scalar or a Tensor of shape `[T, B]`
          if mixing is a function of state.
      back_prop: Whether to backpropagate.
      name: Sets the name_scope for this op.
    Returns:
        Tensor of shape `[T, B]` containing multistep returns.
    """

    # Regroup:
    #   result[t] = (rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]) +
    #               pcontinues[t]*lambda_*result[t + 1]
    # Define:
    #   sequence[t] = rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]
    #   discount[t] = pcontinues[t]*lambda_
    # Substitute:
    #   result[t] = sequence[t] + discount[t]*result[t + 1]
    # Boundary condition:
    #   result[last] = rewards[last] + pcontinues[last]*state_values[last]
    # Add and subtract the same quantity at BC:
    #   state_values[last] =
    #       lambda_*state_values[last] + (1-lambda_)*state_values[last]
    # This makes:
    #   result[last] =
    #       (rewards[last] + pcontinues[last]*(1-lambda_)*state_values[last]) +
    #       pcontinues[last]*lambda_*state_values[last]
    # Substitute in definitions for sequence and discount:
    #   result[last] = sequence[last] + discount[last]*state_values[last]
    # Define:
    #   initial_value=state_values[last]
    # We get the following recurrent relationship:
    #   result[last] = sequence[last] + decay[last]*initial_value
    #   result[k] = sequence[k] + decay[k] * result[k + 1]
    # This matches the form of scan_discounted_sum:
    #   result = scan_sum_with_discount(sequence, discount,
    #                                   initial_value = state_values[last])  

    sequence = rewards + pcontinues * state_values * (1 - lambda_)
    print("sequence", sequence) if debug else None
    print("sequence.shape", sequence.shape) if debug else None

    discount = pcontinues * lambda_
    print("discount", discount) if debug else None
    print("discount.shape", discount.shape) if debug else None

    initial_value = state_values[-1]
    print("initial_value", initial_value) if debug else None

    return scan_discounted_sum(sequence, discount, initial_value,
                               reverse=True, 
                               back_prop=back_prop)


def scan_discounted_sum(sequence, decay, initial_value, reverse=False,
                        back_prop=True,
                        name="scan_discounted_sum"):
    """Evaluates a cumulative discounted sum along dimension 0.
      ```python
      if reverse = False:
        result[1] = sequence[1] + decay[1] * initial_value
        result[k] = sequence[k] + decay[k] * result[k - 1]
      if reverse = True:
        result[last] = sequence[last] + decay[last] * initial_value
        result[k] = sequence[k] + decay[k] * result[k + 1]
      ```
    Respective dimensions T, B and ... have to be the same for all input tensors.
    T: temporal dimension of the sequence; B: batch dimension of the sequence.
    Args:
      sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
      decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
      initial_value: Tensor of shape `[B, ...]` containing initial value.
      reverse: Whether to process the sum in a reverse order.
      back_prop: Whether to backpropagate.
      name: Sets the name_scope for this op.
    Returns:
      Cumulative sum with discount. Same shape and type as `sequence`.
    """

    elems = [sequence, decay]
    if reverse:
        elems = [reverse_seq(s) for s in elems]
        [sequence, decay] = elems

    # another implementions
    # result = torch.empty_like(sequence)
    # result[0] = sequence[0] + decay[0] * initial_value
    # for i in range(len(result) - 1):
    #     result[i + 1] = sequence[i + 1] + decay[i + 1] * result[i]

    elems = [s.unsqueeze(0) for s in elems]
    elems = torch.cat(elems, dim=0) 
    elems = torch.transpose(elems, 0, 1)
    print("elems", elems) if debug else None
    print("elems.shape", elems.shape) if debug else None

    def scan(foo, x, initial_value, debug=False):
        res = []
        a_ = initial_value.clone().detach()
        print("a_", a_) if debug else None
        print("a_.shape", a_.shape) if debug else None

        res.append(foo(a_, x[0]).unsqueeze(0))
        print("res", res) if debug else None
        print("len(x)", len(x)) if debug else None

        a_ = foo(a_, x[0])

        for i in range(1, len(x)):
            print("i", i) if debug else None
            res.append(foo(a_, x[i]).unsqueeze(0))
            print("res", res) if debug else None

            a_ = foo(a_, x[i])
            print("a_", a_) if debug else None
            print("a_.shape", a_.shape) if debug else None

        return torch.cat(res)

    result = scan(lambda a, x: x[0] + x[1] * a, elems, initial_value=initial_value)

    if reverse:
        result = reverse_seq(result)

    print("result", result) if debug else None
    print("result.shape", result.shape) if debug else None 

    return result


def vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):
    """Computes v-trace return advantages.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    see below function "vtrace_from_importance_weights"
    """
    return vtrace_from_importance_weights(rhos=clipped_rhos, discounts=discounts,
                                          rewards=rewards, values=values,
                                          bootstrap_value=bootstrap_value)


VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def vtrace_from_importance_weights(
        rhos, discounts, rewards, values, bootstrap_value,
        clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0,
        name='vtrace_from_importance_weights'):
    r"""
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    V-trace from log importance weights.
    Calculates V-trace actor critic targets as described in
    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.
    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions. This code also supports the
    case where all tensors have the same number of additional dimensions, e.g.,
    `rewards` is `[T, B, C]`, `values` is `[T, B, C]`, `bootstrap_value`
    is `[B, C]`.
    Args:
      log_rhos: A float32 tensor of shape `[T, B, NUM_ACTIONS]` representing the
        log importance sampling weights, i.e.
        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
        on rhos in log-space for numerical stability.
        # note: in mAS we change it to rhos instead of log_rhos
      discounts: A float32 tensor of shape `[T, B]` with discounts encountered
        when following the behaviour policy.
      rewards: A float32 tensor of shape `[T, B]` containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape `[T, B]` with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape `[B]` with the value function estimate
        at time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper. If None, no clipping is applied.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
        None, no clipping is applied.
      name: The name scope that all V-trace operations will be created in.
    Returns:
      A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape `[T, B]`. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape `[T, B]`. Can be used as the
          advantage in the calculation of policy gradients.
    """

    if clip_rho_threshold is not None:
        clip_rho_threshold = torch.tensor(clip_rho_threshold,
                                          dtype=torch.float32, device=values.device)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold,
                                             dtype=torch.float32, device=values.device)

    # Make sure tensor ranks are consistent.
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.)

    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], bootstrap_value.unsqueeze(0)], axis=0)

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
    print("deltas:", deltas) if debug else None
    print("deltas.shape:", deltas.shape) if debug else None

    # Note that all sequences are reversed, computation starts from the back.
    '''
    Note this code is wrong, we should use zip to concat
    sequences = (
        torch.flip(discounts, dims=[0]),
        torch.flip(cs, dims=[0]),
        torch.flip(deltas, dims=[0]),
    )
    '''
    flip_discounts = torch.flip(discounts, dims=[0])
    flip_cs = torch.flip(cs, dims=[0])
    flip_deltas = torch.flip(deltas, dims=[0])

    sequences = [item for item in zip(flip_discounts, flip_cs, flip_deltas)]

    # V-trace vs are calculated through a 
    # scan from the back to the beginning
    # of the given trajectory.
    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc
    initial_values = torch.zeros_like(bootstrap_value, device=bootstrap_value.device)

    # our implemented scan function for pytorch
    def scan(foo, x, initial_value):
        res = []
        a_ = initial_value.clone().detach()
        res.append(foo(a_, x[0]).unsqueeze(0))

        # should not miss this line
        a_ = foo(a_, x[0])

        for i in range(1, len(x)):
            res.append(foo(a_, x[i]).unsqueeze(0))
            a_ = foo(a_, x[i])
        return torch.cat(res)

    vs_minus_v_xs = scan(foo=scanfunc, x=sequences, initial_value=initial_values)

    # Reverse the results back to original order.
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[0])

    # Add V(x_s) to get v_s.
    vs = torch.add(vs_minus_v_xs, values)

    # Advantage for policy gradient.
    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], axis=0)

    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos

    pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

    # Make sure no gradients backpropagated through the returned values.
    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())


def upgo_returns(values, rewards, discounts, bootstrap):
    """Computes the UPGO return targets.

    Args:
      values: Estimated state values. Shape [T, B].
      rewards: Rewards received moving to the next state. Shape [T, B].
      discounts: If the step is NOT final. Shape [T, B].
      bootstrap: Bootstrap values. Shape [B].
    Returns:
      UPGO return targets. Shape [T, B].
    """
    print("rewards", rewards) if debug else None
    print("discounts", discounts) if debug else None

    # we change it to pytorch version
    # next_values = np.concatenate((values[1:], np.expand_dims(bootstrap, axis=0)), axis=0)
    # next_values. Shape [T, B].
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(dim=0)], dim=0)
    print("next_values", next_values) if debug else None
    print("next_values.shape", next_values.shape) if debug else None

    # Upgo can be viewed as a lambda return! The trace continues (i.e. lambda =
    # 1.0) if r_t + V_tp1 > V_t. original G_t = r_t +  
    lambdas = (rewards + discounts * next_values) >= values
    print("lambdas", lambdas) if debug else None
    print("lambdas.shape", lambdas.shape) if debug else None

    # change the bool tensor to float tensor
    lambdas = lambdas.float()

    # Shift lambdas left one slot, such that V_t matches indices with lambda_tp1.
    # lambdas = np.concatenate((lambdas[1:], np.ones_like(lambdas[-1:])), axis=0)
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:], device=lambdas.device)], dim=0)

    return lambda_returns(next_values, rewards, discounts, lambdas)


def generalized_lambda_returns(rewards,
                               pcontinues,
                               values,
                               bootstrap_value,
                               lambda_=1,
                               name="generalized_lambda_returns"):
    """
    # not used actually in mini-AlphaStar
    code at https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
    Computes lambda-returns along a batch of (chunks of) trajectories.
    For lambda=1 these will be multistep returns looking ahead from each
    state to the end of the chunk, where bootstrap_value is used. If you pass an
    entire trajectory and zeros for bootstrap_value, this is just the Monte-Carlo
    return / TD(1) target.
    For lambda=0 these are one-step TD(0) targets.
    The sequences in the tensors should be aligned such that an agent in a state
    with value `V` transitions into another state with value `V'`, receiving
    reward `r` and pcontinue `p`. Then `V`, `r` and `p` are all at the same index
    `i` in the corresponding tensors. `V'` is at index `i+1`, or in the
    `bootstrap_value` tensor if `i == T`.
    Subtracting `values` from these lambda-returns will yield estimates of the
    advantage function which can be used for both the policy gradient loss and
    the baseline value function loss in A3C / GAE.
    Args:
      rewards: 2-D Tensor with shape `[T, B]`.
      pcontinues: 2-D Tensor with shape `[T, B]`.
      values: 2-D Tensor containing estimates of the state values for timesteps
        0 to `T-1`. Shape `[T, B]`.
      bootstrap_value: 1-D Tensor containing an estimate of the value of the
        final state at time `T`, used for bootstrapping the target n-step
        returns. Shape `[B]`.
      lambda_: an optional scalar or 2-D Tensor with shape `[T, B]`.
      name: Customises the name_scope for this op.
    Returns:
      2-D Tensor with shape `[T, B]`
    """

    if lambda_ == 1:
                # This is actually equivalent to the branch below, just an optimisation
                # to avoid unnecessary work in this case:
        return scan_discounted_sum(
            rewards,
            pcontinues,
            initial_value=bootstrap_value,
            reverse=True,
            back_prop=False,
            name="multistep_returns")
    else:
        v_tp1 = torch.concat([values[1:, :], torch.unsqueeze(bootstrap_value, 0)], axis=0)
        # `back_prop=False` prevents gradients flowing into values and
        # bootstrap_value, which is what you want when using the bootstrapped
        # lambda-returns in an update as targets for values.
        return multistep_forward_view(
            rewards,
            pcontinues,
            v_tp1,
            lambda_,
            back_prop=False,
            name="generalized_lambda_returns")


def entropy(policy_logits, selected_mask=None, entity_mask=None, unit_type_entity_mask=None,):
    # policy_logits shape: [seq_batch_size, channel_size]
    # masks shape: [seq_batch_size, 1]

    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    #policy = softmax(policy_logits)
    log_policy = logsoftmax(policy_logits)
    policy = torch.exp(log_policy)

    ent = -policy * log_policy

    if selected_mask is not None:
        print("ent.shape:", ent.shape) if debug else None
        print("selected_mask.shape:", selected_mask.shape) if debug else None
        ent = ent * selected_mask.unsqueeze(-1)    

    if entity_mask is not None:
        print("ent.shape:", ent.shape) if debug else None
        print("entity_mask.shape:", entity_mask.shape) if debug else None
        ent = ent * entity_mask

    return ent


def kl(student_logits, teacher_logits, selected_mask=None, entity_mask=None, 
       unit_type_entity_mask=None, outlier_remove=True, target_entity_num=None,
       target_select_units_num=None, player_select_units_num=None):
    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    s_logprobs = logsoftmax(student_logits)
    print("s_logprobs:", s_logprobs) if debug else None
    print("s_logprobs.shape:", s_logprobs.shape) if debug else None    

    t_logprobs = logsoftmax(teacher_logits)
    print("t_logprobs:", t_logprobs) if debug else None
    print("t_logprobs.shape:", t_logprobs.shape) if debug else None  

    #teacher_probs = softmax(teacher_logits)
    teacher_probs = torch.exp(t_logprobs)
    print("teacher_probs:", teacher_probs) if debug else None
    print("teacher_probs.shape:", teacher_probs.shape) if debug else None

    kl = teacher_probs * (t_logprobs - s_logprobs)

    if selected_mask is not None:
        print("kl.shape:", kl.shape) if 1 else None
        print("selected_mask.shape:", selected_mask.shape) if 1 else None
        kl = kl * selected_mask.unsqueeze(-1) 

    if entity_mask is not None:
        print("kl.shape:", kl.shape) if 1 else None
        print("entity_mask.shape:", entity_mask.shape) if 1 else None
        kl = kl * entity_mask

    if unit_type_entity_mask is not None:
        print("kl.shape:", kl.shape) if 1 else None
        print("unit_type_entity_mask.shape:", unit_type_entity_mask.shape) if 1 else None
        kl = kl * unit_type_entity_mask

    print("kl:", kl) if debug else None

    if outlier_remove:
        outlier_mask = (torch.abs(kl) >= 1e6)
        kl = kl * ~outlier_mask
    else:
        outlier_mask = (torch.abs(kl) >= 1e6)
        if outlier_mask.any() > 0:
            print("outlier_mask:", outlier_mask.nonzero(as_tuple=True)) if 1 else None
            index = outlier_mask.nonzero(as_tuple=True)

            print("kl[index]:", kl[index]) if 1 else None
            print("student_logits[index]:", student_logits[index]) if 1 else None
            print("teacher_logits[index]:", teacher_logits[index]) if 1 else None

            id0 = index[0]

            print("target_entity_num[id0]:", target_entity_num[id0]) if 1 else None
            print("target_select_units_num[id0]:", target_select_units_num[id0]) if 1 else None
            print("player_select_units_num[id0]:", player_select_units_num[id0]) if 1 else None            

            # stop()

    print("kl.shape:", kl.shape) if debug else None

    return kl


def compute_unclipped_logrho(behavior_logits, target_logits, actions, selected_mask=None,
                             entity_mask=None, debug=False):
    """Helper function."""

    target_log_prob = log_prob_masked(actions, target_logits, reduction="none", selected_mask=selected_mask,
                                      entity_mask=entity_mask, debug=debug)
    print("target_log_prob:", target_log_prob) if debug else None

    behavior_log_prob = log_prob_masked(actions, behavior_logits, reduction="none", selected_mask=selected_mask,
                                        entity_mask=entity_mask, debug=debug)
    print("behavior_log_prob:", behavior_log_prob) if debug else None

    subtract = target_log_prob - behavior_log_prob
    print("subtract:", subtract) if debug else None

    return subtract


def compute_importance_weights(behavior_logits, target_logits, actions, selected_mask=None,
                               entity_mask=None, debug=False):
    """Computes clipped importance weights."""

    logrho = compute_unclipped_logrho(behavior_logits, target_logits, actions, selected_mask=selected_mask,
                                      entity_mask=entity_mask, debug=debug)
    print("logrho:", logrho) if debug else None
    print("logrho.shape:", logrho.shape) if debug else None

    rho = torch.exp(logrho)
    print("rho:", rho) if debug else None

    # change to pytorch version
    clip_rho = torch.clamp(rho, max=1.)
    print("clip_rho:", clip_rho) if debug else None

    return clip_rho   


def compute_cliped_importance_weights(target_log_prob, behavior_log_prob):
    """Computes clipped importance weights."""

    logrho = target_log_prob - behavior_log_prob
    rho = torch.exp(logrho)
    clip_rho = torch.clamp(rho, max=1.)

    return clip_rho 


def log_prob_masked(logits, actions, mask_used):
    """Returns the log probability of taking an action given the logits."""
    [selected_mask, entity_mask, unit_type_entity_mask] = mask_used

    if entity_mask is not None:
        # for selected units head and target unit head
        entity_mask = entity_mask.reshape(-1, entity_mask.shape[-1]).unsqueeze(dim=1).bool()

        if unit_type_entity_mask is not None:
            unit_type_entity_mask = unit_type_entity_mask.reshape(-1, unit_type_entity_mask.shape[-1]).unsqueeze(dim=1).bool()
            print("unit_type_entity_mask.shape:", unit_type_entity_mask.shape) if 1 else None

        print("logits.shape:", logits.shape) if 1 else None
        print("actions.shape:", actions.shape) if 1 else None
        print("entity_mask.shape:", entity_mask.shape) if 1 else None

        # logits: shape [BATCH_SIZE, MAX_SELECT, MAX_ENTITY]
        # entity_mask: shape [BATCH_SIZE, 1, MAX_ENTITY]
        # unit_type_entity_mask: shape [BATCH_SIZE, 1, MAX_ENTITY]
        logits = logits  
        logits = logits.masked_fill(~entity_mask, -1e9)

        if unit_type_entity_mask is not None:
            logits = logits.masked_fill(~unit_type_entity_mask, -1e9)

        if len(logits.shape) == 3:
            select_size = logits.shape[1]
            logits = logits.reshape(-1, logits.shape[-1])
        if len(actions.shape) == 3:
            actions = actions.reshape(-1, actions.shape[-1])

    print("logits.shape:", logits.shape) if 1 else None
    print("actions.shape:", actions.shape) if 1 else None

    loss = torch.nn.CrossEntropyLoss(reduction="none")

    # actions: shape [BATCH_SIZE, 1]
    actions = torch.squeeze(actions, dim=-1)

    # logits: shape [BATCH_SIZE, CLASS_SIZE]
    # actions: shape [BATCH_SIZE]
    loss_result = loss(logits, actions)

    if selected_mask is not None:
        # for selected units head
        selected_mask = selected_mask.reshape(-1, selected_mask.shape[-1])
        if len(loss_result.shape) == 1:
            loss_result = loss_result.reshape(-1, AHP.max_selected)

        print("loss_result.shape:", loss_result.shape) if 1 else None
        print("selected_mask.shape:", selected_mask.shape) if 1 else None

        loss_result = loss_result * selected_mask
        loss_result = torch.sum(loss_result, dim=-1)

        # action_log_prob: shape [BATCH_SIZE]
        print("loss_result.shape:", loss_result.shape) if debug else None

    # change to right log_prob
    # the_log_prob: shape [BATCH_SIZE]
    the_log_prob = -loss_result

    return the_log_prob


def log_prob(actions, logits, reduction="none"):
    """Returns the log probability of taking an action given the logits."""
    # Equivalent to tf.sparse_softmax_cross_entropy_with_logits.

    # note CrossEntropyLoss is $ - log(e^(x_i) / \sum_j{e^{x_j}}) $
    # such for log_prob is actually -CrossEntropyLoss
    loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    print("logits:", logits) if debug else None
    print("logits.shape:", logits.shape) if debug else None
    print("actions:", actions) if debug else None
    print("actions.shape:", actions.shape) if debug else None

    # actions: shape [BATCH_SIZE, 1]
    actions = torch.squeeze(actions, dim=-1)

    # logits: shape [BATCH_SIZE, CLASS_SIZE]
    # actions: shape [BATCH_SIZE]
    loss_result = loss(logits, actions)

    # Original AlphaStar pseudocode is wrong
    # AlphaStar: return loss_result

    # change to right log_prob
    # the_log_prob: shape [BATCH_SIZE]
    the_log_prob = -loss_result

    return the_log_prob


def test():

    test_td_lamda_loss = False

    if test_td_lamda_loss:
        batch_size = 2
        seq_len = 4

        device = 'cpu'

        is_final = [[0, 0], [0, 0], [0, 0], [0, 0]]
        baselines = [[2, 1], [3, 4], [0, 5], [2, 7]]
        rewards = [[0, 0], [0, 0], [0, 1], [-1, 0]]

        baselines = torch.tensor(baselines, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        discounts = ~np.array(is_final[:-1], dtype=np.bool)  # don't forget dtype=np.bool!, or '~' ouput -1 instead of 1
        discounts = torch.tensor(discounts, dtype=torch.float, device=device)

        print("discounts:", discounts) if debug else None

        boostrapvales = baselines[1:]
        rewards_short = rewards[:-1]

        if 1:
            # The baseline is then updated using TDLambda, with relative weighting 10.0 and lambda 0.8.
            returns = lambda_returns(boostrapvales, rewards_short, discounts, lambdas=0.8)

            # returns = stop_gradient(returns)
            returns = returns.detach()
            print("returns:", returns) if debug else None

            result = returns - baselines[:-1]
            print("result:", result) if debug else None

            # change to pytorch version
            td_lambda_loss = 0.5 * torch.mean(torch.square(result))
            print('td_lambda_loss', td_lambda_loss)

        if 2:
            lambdas = 0.8

            returns = torch.empty_like(rewards_short)

            returns[-1, :] = rewards_short[-1, :] + boostrapvales[-1, :]
            for t in reversed(range(rewards_short.shape[0] - 1)):
                returns[t, :] = rewards_short[t, :] + lambdas * returns[t + 1, :] \
                    + (1 - lambdas) * boostrapvales[t, :]
            print("returns:", returns) if debug else None

            result = returns.detach() - baselines[:-1]
            print("result:", result) if debug else None

            td_lambda_loss_2 = 0.5 * torch.pow(result, 2).mean()
            print('td_lambda_loss_2', td_lambda_loss_2)
