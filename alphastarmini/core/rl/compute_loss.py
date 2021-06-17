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

from alphastarmini.core.rl.utils import Trajectory
from alphastarmini.core.rl.action import ArgsActionLogits

from alphastarmini.core.rl import utils as U
from alphastarmini.core.rl import pseudo_reward as PR

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

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

MASK_FIELDS = [
    'action_type',  # Action taken.
    'delay',
    'queued',
    'selected_units',
    'target_unit',
    'target_location',
]

BEHAVIOR_LOGITS_FIELDS = [
    'action_type',  # Action taken.
    'delay',
    'queued',
    'selected_units',
    'target_unit',
    'target_location',
]

TEACHER_LOGITS_FIELDS = [
    'action_type',  # Action taken.
    'delay',
    'queued',
    'selected_units',
    'target_unit',
    'target_location',
]


Mask = collections.namedtuple('Mask', ACTION_FIELDS)

# below now only consider action_type now
# baseline are also all zeros


def log_prob(actions, logits, reduction="none"):
    """Returns the log probability of taking an action given the logits."""
    # Equivalent to tf.sparse_softmax_cross_entropy_with_logits.

    loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    # logits: shape [BATCH_SIZE, CLASS_SIZE]
    # actions: shape [BATCH_SIZE]
    return loss(logits, torch.squeeze(actions, dim=-1))


def is_sampled(z):
    """Takes a tensor of zs. Returns a mask indicating which z's are sampled."""
    return True


def filter_by(action_fields, target):
    """Returns the subset of `target` corresponding to `action_fields`.

    Autoregressive actions are composed of many logits.  We often want to select a
    subset of these logits.

    Args:
      action_fields: One of 'action_type', 'delay', or 'arguments'.
      target: A ArgsActionLogits tensor.
    Returns:
      A tensor
    """
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
    """Returns the subset of `target` corresponding to `action_fields`.

    Autoregressive actions are composed of many logits.  We often want to select a
    subset of these logits.

    Args:
      action_fields: One of 'action_type', 'delay', or 'arguments'.
      target: A list of tensors corresponding to the SC2 action spec. [T x B x S]
    Returns:
      A tensor corresponding to a subset of `target`, with only the tensors relevant
      to `action_fields`.
    """
    return torch.cat([getattr(b, action_fields) for a in target_list for b in a], dim=0)


def filter_by_for_masks(action_fields, target_mask):
    """Returns the subset of `mask` corresponding to `action_fields`.

    Autoregressive actions are composed of many logits.  We often want to select a
    subset of these logits.

    Args:
      action_fields: One of 'action_type', 'delay', or 'arguments'.
      target_mask: A list of tensors corresponding to the masks. [T x B x 1]
    Returns:
      A tensor corresponding to a subset of `target`, with only the tensors relevant
      to `action_fields`.
    """
    index_list = ['action_type', 'delay', 'queue', 'units', 'target_unit', 'target_location']
    index = index_list.index(action_fields)

    mask = torch.tensor(target_mask)
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


def entropy(policy_logits, masks):
    # policy_logits shape: [seq_batch_size, channel_size]
    # masks shape: [seq_batch_size, 1]

    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    policy = softmax(policy_logits)
    log_policy = logsoftmax(policy_logits)

    ent = torch.sum(-policy * log_policy * masks, axis=-1)  # Aggregate over actions.
    # Normalize by actions available.
    normalized_entropy = ent / torch.log(torch.tensor(policy_logits.shape[-1], dtype=torch.float32))

    return normalized_entropy


def entropy_loss(policy_logits):
    """Computes the entropy loss for a set of logits.

    Args:
      policy_logits: namedtuple of the logits for each policy argument.
        Each shape is [..., N_i].
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """

    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    policy = softmax(policy_logits.action_type)
    log_policy = logsoftmax(policy_logits.action_type)

    return torch.mean(torch.sum(-policy * log_policy, axis=-1))


def entropy_loss_for_all_arguments(policy_logits, masks):
    """Computes the entropy loss for a set of logits.

    Args:
      policy_logits: namedtuple of the logits for each policy argument.
        Each shape is [..., N_i].
      masks: The masks. Each shape is policy_logits.shape[:-1].
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """

    index_list = ['action_type', 'delay', 'queue', 'units', 'target_unit', 'target_location']
    masks = torch.tensor(masks)

    entropy_list = []
    for x in index_list:     
        logits = getattr(policy_logits, x) 

        print("x name:", x) if debug else None
        print("logits.shape:", logits.shape) if debug else None

        if x == "target_unit":
            # remove the axis 2
            logits = logits.squeeze(dim=1)

        print("logits.shape:", logits.shape) if debug else None

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

        entropy_item = entropy(logits, mask)

        print("entropy_item:", entropy_item) if debug else None

        entropy_list.append(entropy_item)

    return torch.mean(torch.cat(entropy_list, axis=0))


def kl(student_logits, teacher_logits, mask):
    softmax = nn.Softmax(dim=-1)
    logsoftmax = nn.LogSoftmax(dim=-1)

    s_logprobs = logsoftmax(student_logits)
    t_logprobs = logsoftmax(teacher_logits)
    teacher_probs = softmax(teacher_logits)
    return teacher_probs * (t_logprobs - s_logprobs) * mask


def human_policy_kl_loss(student_logits, teacher_logits, action_type_kl_cost):
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
    action_type_loss = kl(student_logits, teacher_logits, 1)
    kl_loss = action_type_kl_cost * torch.mean(action_type_loss)

    return kl_loss


def _reverse_seq(sequence, sequence_lengths=None):
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


def scan_discounted_sum(sequence, decay, initial_value, reverse=False,
                        sequence_lengths=None, back_prop=True,
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
      if sequence_lengths is set then x1 and x2 below are equivalent:
      ```python
      x1 = zero_pad_to_length(
        scan_discounted_sum(
            sequence[:length], decays[:length], **kwargs), length=T)
      x2 = scan_discounted_sum(sequence, decays,
                               sequence_lengths=[length], **kwargs)
      ```
    Args:
      sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
      decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
      initial_value: Tensor of shape `[B, ...]` containing initial value.
      reverse: Whether to process the sum in a reverse order.
      sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
        (reversed and then) summed.
      back_prop: Whether to backpropagate.
      name: Sets the name_scope for this op.
    Returns:
      Cumulative sum with discount. Same shape and type as `sequence`.
    """
    # Note this can be implemented in terms of cumprod and cumsum,
    # approximately as (ignoring boundary issues and initial_value):
    #
    # cumsum(decay_prods * sequence) / decay_prods
    # where decay_prods = reverse_cumprod(decay)
    #
    # One reason this hasn't been done is that multiplying then dividing again by
    # products of decays isn't ideal numerically, in particular if any of the
    # decays are zero it results in NaNs.
    if sequence_lengths is not None:
        raise NotImplementedError

    elems = [sequence, decay]

    print("initial_value", initial_value) if debug else None
    print("initial_value.shape", initial_value) if debug else None

    if reverse:
        elems = [_reverse_seq(s, sequence_lengths) for s in elems]

    print("sequence", elems[0]) if debug else None
    print("sequence.shape", elems[0].shape) if debug else None

    print("decay", elems[1]) if debug else None
    print("decay.shape", elems[1].shape) if debug else None

    elems = [s.unsqueeze(0) for s in elems]

    elems = torch.cat(elems, dim=0) 
    print("elems", elems) if debug else None
    print("elems.shape", elems.shape) if debug else None

    elems = torch.transpose(elems, 0, 1)
    print("elems", elems) if debug else None
    print("elems.shape", elems.shape) if debug else None

    # we change it to a pytorch version
    def scan(foo, x, initial_value):
        res = []
        a_ = initial_value.clone().detach()
        print("a_", a_) if debug else None
        print("a_.shape", a_.shape) if debug else None

        res.append(foo(a_, x[0]).unsqueeze(0))
        print("res", res) if debug else None
        print("len(x)", len(x)) if debug else None

        for i in range(1, len(x)):
            print("i", i) if debug else None
            res.append(foo(a_, x[i]).unsqueeze(0))
            print("res", res) if debug else None

            a_ = foo(a_, x[i])
            print("a_", a_) if debug else None
            print("a_.shape", a_.shape) if debug else None

        return torch.cat(res)

    # summed = tf.scan(lambda a, x: x[0] + x[1] * a,
    #                 elems,
    #                 initializer=tf.convert_to_tensor(initial_value),
    #                 parallel_iterations=1,
    #                 back_prop=back_prop)    
    summed = scan(lambda a, x: x[0] + x[1] * a, elems, initial_value=initial_value)

    print("summed", summed) if debug else None
    print("summed.shape", summed.shape) if debug else None   

    if reverse:
        summed = _reverse_seq(summed, sequence_lengths)

    print("summed", summed) if debug else None

    return summed


def multistep_forward_view(rewards, pcontinues, state_values, lambda_,
                           back_prop=True, sequence_lengths=None,
                           name="multistep_forward_view_op"):
    """Evaluates complex backups (forward view of eligibility traces).
      ```python
      result[t] = rewards[t] +
          pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t])
      result[last] = rewards[last] + pcontinues[last]*state_values[last]
      ```
      This operation evaluates multistep returns where lambda_ parameter controls
      mixing between full returns and boostrapping. It is users responsibility
      to provide state_values. Depending on how state_values are evaluated this
      function can evaluate targets for Q(lambda), Sarsa(lambda) or some other
      multistep boostrapping algorithm.
      More information about a forward view is given here:
        http://incompleteideas.net/sutton/book/ebook/node74.html
      Please note that instead of evaluating traces and then explicitly summing
      them we instead evaluate mixed returns in the reverse temporal order
      by using the recurrent relationship given above.
      The parameter lambda_ can either be a constant value (e.g for Peng's
      Q(lambda) and Sarsa(_lambda)) or alternatively it can be a tensor containing
      arbitrary values (Watkins' Q(lambda), Munos' Retrace, etc).
      The result of evaluating this recurrence relation is a weighted sum of
      n-step returns, as depicted in the diagram below. One strategy to prove this
      equivalence notes that many of the terms in adjacent n-step returns
      "telescope", or cancel out, when the returns are summed.
      Below L3 is lambda at time step 3 (important: this diagram is 1-indexed, not
      0-indexed like Python). If lambda is scalar then L1=L2=...=Ln.
      g1,...,gn are discounts.
      ```
      Weights:  (1-L1)        (1-L2)*l1      (1-L3)*l1*l2  ...  L1*L2*...*L{n-1}
      Returns:    |r1*(g1)+     |r1*(g1)+      |r1*(g1)+          |r1*(g1)+
                v1*(g1)         |r2*(g1*g2)+   |r2*(g1*g2)+       |r2*(g1*g2)+
                              v2*(g1*g2)       |r3*(g1*g2*g3)+    |r3*(g1*g2*g3)+
                                             v3*(g1*g2*g3)               ...
                                                                  |rn*(g1*...*gn)+
                                                                vn*(g1*...*gn)
      ```
    Args:
      rewards: Tensor of shape `[T, B]` containing rewards.
      pcontinues: Tensor of shape `[T, B]` containing discounts.
      state_values: Tensor of shape `[T, B]` containing state values.
      lambda_: Mixing parameter lambda.
          The parameter can either be a scalar or a Tensor of shape `[T, B]`
          if mixing is a function of state.
      back_prop: Whether to backpropagate.
      sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
        (reversed and then) summed, same as in `scan_discounted_sum`.
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

    return scan_discounted_sum(sequence, discount, state_values[-1],
                               reverse=True, sequence_lengths=sequence_lengths,
                               back_prop=back_prop)


def lambda_returns(values_tp1, rewards, discounts, lambdas=0.8):
    """Computes lambda returns.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
    """

    # we only implment the lambda return version in AlphaStar when lambdas=0.8
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


def generalized_lambda_returns(rewards,
                               pcontinues,
                               values,
                               bootstrap_value,
                               lambda_=1,
                               name="generalized_lambda_returns"):
    """
    code at https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74

    Computes lambda-returns along a batch of (chunks of) trajectories.
    For lambda=1 these will be multistep returns looking ahead from each
    state to the end of the chunk, where bootstrap_value is used. If you pass an
    entire trajectory and zeros for bootstrap_value, this is just the Monte-Carlo
    return / TD(1) target.
    For lambda=0 these are one-step TD(0) targets.
    For inbetween values of lambda these are lambda-returns / TD(lambda) targets,
    except that traces are always cut off at the end of the chunk, since we can't
    see returns beyond then. If you pass an entire trajectory with zeros for
    bootstrap_value though, then they're plain TD(lambda) targets.
    lambda can also be a tensor of values in [0, 1], determining the mix of
    bootstrapping vs further accumulation of multistep returns at each timestep.
    This can be used to implement Retrace and other algorithms. See
    `sequence_ops.multistep_forward_view` for more info on this. Another way to
    think about the end-of-chunk cutoff is that lambda is always effectively zero
    on the timestep after the end of the chunk, since at the end of the chunk we
    rely entirely on bootstrapping and can't accumulate returns looking further
    into the future.
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

    # values.get_shape().assert_has_rank(2)
    # rewards.get_shape().assert_has_rank(2)
    # pcontinues.get_shape().assert_has_rank(2)
    # bootstrap_value.get_shape().assert_has_rank(1)

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


VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):
    """Computes v-trace return advantages.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    see below function "vtrace_from_importance_weights"
    """
    return vtrace_from_importance_weights(rhos=clipped_rhos, discounts=discounts,
                                          rewards=rewards, values=values,
                                          bootstrap_value=bootstrap_value)


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

    #rhos = torch.tensor(rhos, dtype=torch.float32)
    #discounts = torch.tensor(discounts, dtype=torch.float32)
    #rewards = torch.tensor(rewards, dtype=torch.float32)
    #values = torch.tensor(values, dtype=torch.float32)
    #bootstrap_value = torch.tensor(bootstrap_value, dtype=torch.float32)

    if clip_rho_threshold is not None:
        clip_rho_threshold = torch.tensor(clip_rho_threshold,
                                          dtype=torch.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold,
                                             dtype=torch.float32)

    # Make sure tensor ranks are consistent.
    # 
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.)
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], bootstrap_value.unsqueeze(0)], axis=0)

    print("rhos:", rhos) if debug else None
    print("rhos.shape:", rhos.shape) if debug else None

    print("rewards:", rewards) if debug else None
    print("rewards.shape:", rewards.shape) if debug else None

    print("bootstrap_value:", bootstrap_value) if debug else None
    print("bootstrap_value.shape:", bootstrap_value.shape) if debug else None

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
        print("sequence_item", sequence_item) if debug else None
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = torch.zeros_like(bootstrap_value)

    # our implemented scan function for pytorch
    def scan(foo, x, initial_value):
        res = []
        a_ = initial_value.clone().detach()
        print("a_", a_) if debug else None
        print("a_.shape", a_.shape) if debug else None

        res.append(foo(a_, x[0]).unsqueeze(0))
        print("res", res) if debug else None
        print("len(x)", len(x)) if debug else None

        for i in range(1, len(x)):
            print("i", i) if debug else None
            res.append(foo(a_, x[i]).unsqueeze(0))
            print("res", res) if debug else None

            a_ = foo(a_, x[i])
            print("a_", a_) if debug else None
            print("a_.shape", a_.shape) if debug else None

        return torch.cat(res)

    vs_minus_v_xs = scan(foo=scanfunc, x=sequences, initial_value=initial_values)

    '''
    # the original tensorflow code
    vs_minus_v_xs = tf.scan(
        fn=scanfunc,
        elems=sequences,
        initializer=initial_values,
        parallel_iterations=1,
        back_prop=False)
        '''

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


def td_lambda_loss(baselines, rewards, trajectories): 
    discounts = ~np.array(trajectories.is_final[:-1])
    discounts = torch.tensor(discounts)

    baselines = baselines
    rewards = rewards[1:]

    # The baseline is then updated using TDLambda, with relative weighting 10.0 and lambda 0.8.
    returns = lambda_returns(baselines[1:], rewards, discounts, lambdas=0.8)
    # returns = stop_gradient(returns)
    print("returns:", returns) if debug else None
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
    action_log_prob = log_prob(actions, logits, reduction="none")
    print("action_log_prob:", action_log_prob) if debug else None
    print("action_log_prob.shape:", action_log_prob.shape) if debug else None

    # advantages = stop_gradient(advantages)
    advantages = advantages.clone().detach()
    print("advantages:", advantages) if debug else None
    print("advantages.shape:", advantages.shape) if debug else None

    results = mask * advantages * action_log_prob
    print("results:", results) if debug else None
    print("results.shape:", results.shape) if debug else None
    return results


def compute_unclipped_logrho(behavior_logits, target_logits, actions):
    """Helper function for compute_importance_weights."""
    target_log_prob = log_prob(actions, target_logits, reduction="none")
    behavior_log_prob = log_prob(actions, behavior_logits, reduction="none")

    return target_log_prob - behavior_log_prob


def compute_importance_weights(behavior_logits, target_logits, actions):
    """Computes clipped importance weights."""
    logrho = compute_unclipped_logrho(behavior_logits, target_logits, actions)
    print("logrho:", logrho) if debug else None
    print("logrho.shape:", logrho.shape) if debug else None

    # change to pytorch version
    return torch.clamp(torch.exp(logrho), max=1.)


def vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                   action_fields):
    """Computes v-trace policy gradient loss. Helper for split_vtrace_pg_loss."""
    # Remove last timestep from trajectories and baselines.
    print("action_fields", action_fields) if debug else None

    trajectories = Trajectory(*tuple(item[:-1] for item in trajectories))
    print("trajectories.reward", trajectories.reward) if debug else None

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
    split_target_logits = split_target_logits.reshape(AHP.batch_size, AHP.sequence_length, *action_size)
    # shape: [seq_size x batch_size x action_size]
    split_target_logits = torch.transpose(split_target_logits, 0, 1)
    # shape: [new_seq_size x batch_size x action_size]
    split_target_logits = split_target_logits[:-1]
    # shape: [seq_batch_size x action_size]
    split_target_logits = split_target_logits.reshape(-1, *action_size)
    print("split_target_logits", split_target_logits) if debug else None   
    print("split_target_logits.shape", split_target_logits.shape) if debug else None 

    target_logits = split_target_logits
    print("target_logits", target_logits) if debug else None
    print("target_logits.shape", target_logits.shape) if debug else None

    behavior_logits = filter_by_for_lists(action_fields, trajectories.behavior_logits)
    print("behavior_logits", behavior_logits) if debug else None
    print("behavior_logits.shape", behavior_logits.shape) if debug else None

    actions = filter_by_for_lists(action_fields, trajectories.action)
    print("actions", actions) if debug else None
    print("actions.shape", actions.shape) if debug else None

    if action_fields == 'units' or action_fields == 'target_unit':
        seqbatch_unit_shape = target_logits.shape[0:2]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

    if action_fields == 'target_location':
        target_logits = target_logits.reshape(target_logits.shape[0], -1)
        behavior_logits = behavior_logits.reshape(behavior_logits.shape[0], -1)

        actions_2 = torch.zeros(behavior_logits.shape[0], 1, dtype=torch.int64)
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
    clipped_rhos = compute_importance_weights(behavior_logits, target_logits, actions)

    if action_fields == 'units' or action_fields == 'target_unit':
        clipped_rhos = clipped_rhos.reshape(seqbatch_unit_shape, -1)
        clipped_rhos = torch.mean(clipped_rhos, dim=-1)

    # To make the clipped_rhos shape to be [T-1, B]
    clipped_rhos = clipped_rhos.reshape(rewards.shape)
    print("clipped_rhos", clipped_rhos) if debug else None
    print("clipped_rhos.shape", clipped_rhos.shape) if debug else None

    discounts = ~np.array(trajectories.is_final)
    discounts = torch.tensor(discounts, dtype=torch.float32)

    # we implement the vtrace_advantages
    # vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):

    weighted_advantage = vtrace_advantages(clipped_rhos, rewards,
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
        target_logits = target_logits.reshape(AHP.sequence_length - 1, AHP.batch_size * AHP.max_selected, -1)
        actions = actions.reshape(AHP.sequence_length - 1, AHP.batch_size * AHP.max_selected, -1)

        weighted_advantage = torch.cat([weighted_advantage] * AHP.max_selected, dim=1)
        masks = torch.cat([masks] * AHP.max_selected, dim=1)
    else:
        target_logits = target_logits.reshape(AHP.sequence_length - 1, AHP.batch_size, -1)
        actions = actions.reshape(AHP.sequence_length - 1, AHP.batch_size, -1)

    result = compute_over_actions(policy_gradient_loss, target_logits,
                                  actions, weighted_advantage, masks)

    if action_fields == 'units':
        result = result.reshape(-1, AHP.max_selected)
        result = torch.mean(result, dim=-1)

    print("result", result) if debug else None
    print("result.shape", result.shape) if debug else None

    # note: in mAS, we should make the result not beyond 0
    # return result
    return 0.5 * torch.mean(torch.square(result))


def split_vtrace_pg_loss(target_logits, baselines, rewards, trajectories):
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
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'action_type')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'delay')

    # note: here we use queue, units, target_unit and target_location to replace the single arguments
    # loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'arguments')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'queue')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'units')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'target_unit')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories, 'target_location')

    return loss.sum()


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
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    print("next_values", next_values) if debug else None
    print("next_values.shape", next_values.shape) if debug else None

    # Upgo can be viewed as a lambda return! The trace continues (i.e. lambda =
    # 1.0) if r_t + V_tp1 > V_t.
    lambdas = (rewards + discounts * next_values) >= values
    print("lambdas", lambdas) if debug else None
    print("lambdas.shape", lambdas.shape) if debug else None

    # change the bool tensor to float tensor
    lambdas = lambdas.float()

    # Shift lambdas left one slot, such that V_t matches indices with lambda_tp1.
    # lambdas = np.concatenate((lambdas[1:], np.ones_like(lambdas[-1:])), axis=0)
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:])], dim=0)

    print("lambdas", lambdas) if debug else None
    print("lambdas.shape", lambdas.shape) if debug else None

    return lambda_returns(next_values, rewards, discounts, lambdas)


def split_upgo_loss(target_logits, baselines, trajectories):
    """Computes split UPGO policy gradient loss.

    See split_vtrace_pg_loss docstring for details on split updates.
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
    reward_tensor = torch.tensor(np.array(trajectories.reward))
    discounts = torch.tensor(~np.array(trajectories.is_final), dtype=torch.float32)

    returns = upgo_returns(values, reward_tensor, discounts, baselines[-1])

    # shape: list of [seq_size x batch_size]
    print("returns", returns) if debug else None
    print("returns.shape", returns.shape) if debug else None

    # Compute the UPGO loss for each action subset.
    # action_type, delay, and other arguments are also similarly separately 
    # updated using UPGO, in the same way as the VTrace Actor-Critic loss, 
    # with relative weight 1.0.
    # We make upgo also contains all the arguments
    loss = sum_upgo_loss(target_logits, values, trajectories, returns)

    return loss


def sum_upgo_loss(target_logits, values, trajectories, returns):
    loss = 0.
    loss += upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'action_type')
    loss += upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'delay')

    loss += upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'queue')
    loss += upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'units')

    loss += upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'target_unit')
    loss += upgo_loss_like_vtrace(target_logits, values, trajectories, returns, 'target_location')

    return loss.sum()


def upgo_loss_like_vtrace(target_logits, values, trajectories, returns, action_fields):
    print("action_fields", action_fields) if debug else None

    # Filter for only the relevant actions/logits/masks.
    target_logits = filter_by(action_fields, target_logits)

    # shape: [batch_seq_size x action_size]
    split_target_logits = target_logits
    action_size = tuple(list(target_logits.shape[1:]))  # from the 3rd dim, it is action dim, may be [S] or [C, S] or [H, W]

    # shape: [batch_size x seq_size x action_size]
    split_target_logits = split_target_logits.reshape(AHP.batch_size, AHP.sequence_length, *action_size)
    # shape: [seq_size x batch_size x action_size]
    split_target_logits = torch.transpose(split_target_logits, 0, 1)
    # shape: [new_seq_size x batch_size x action_size]
    split_target_logits = split_target_logits[:-1]
    # shape: [seq_batch_size x action_size]
    split_target_logits = split_target_logits.reshape(-1, *action_size)

    target_logits = split_target_logits

    behavior_logits = filter_by_for_lists(action_fields, trajectories.behavior_logits)

    actions = filter_by_for_lists(action_fields, trajectories.action)

    if action_fields == 'units' or action_fields == 'target_unit':
        seqbatch_unit_shape = target_logits.shape[0:2]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        behavior_logits = behavior_logits.reshape(-1, behavior_logits.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

    if action_fields == 'target_location':
        target_logits = target_logits.reshape(target_logits.shape[0], -1)
        behavior_logits = behavior_logits.reshape(behavior_logits.shape[0], -1)

        actions_2 = torch.zeros(behavior_logits.shape[0], 1, dtype=torch.int64)

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

    clipped_rhos = compute_importance_weights(behavior_logits,
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
        target_logits = target_logits.reshape(AHP.sequence_length - 1, AHP.batch_size * AHP.max_selected, -1)
        actions = actions.reshape(AHP.sequence_length - 1, AHP.batch_size * AHP.max_selected, -1)

        weighted_advantage = torch.cat([weighted_advantage] * AHP.max_selected, dim=1)
        masks = torch.cat([masks] * AHP.max_selected, dim=1)
    else:
        target_logits = target_logits.reshape(AHP.sequence_length - 1, AHP.batch_size, -1)
        actions = actions.reshape(AHP.sequence_length - 1, AHP.batch_size, -1)

    result = compute_over_actions(policy_gradient_loss, target_logits,
                                  actions, weighted_advantage, masks)

    if action_fields == 'units':
        result = result.reshape(-1, AHP.max_selected)
        result = torch.mean(result, dim=-1)

    print("result", result) if debug else None
    print("result.shape", result.shape) if debug else None

    return result


def compute_pseudoreward(trajectories, reward_name):
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

    '''
    # if the trajectories is a list
    leven_rewards = []
    for i, traj in enumerate(trajectories):
        home_obs_seq = traj.observation
        bo_seq = traj.build_order
        z_bo_seq = traj.z_build_order

        r_traj = []
        for j, home_obs in enumerate(home_obs_seq):
            bo = bo_seq[j]
            z_bo = z_bo_seq[j]
            r = PR.reward_by_build_order(bo, z_bo)
            r_traj.append(r)

        leven_rewards.append(r_traj)

    hamming_rewards = []
    for i, traj in enumerate(trajectories):
        home_obs_seq = traj.observation
        ucb_seq = traj.unit_counts
        z_ucb_seq = traj.z_unit_counts

        r_traj = []
        for j, home_obs in enumerate(home_obs_seq):
            ucb = ucb_seq[j]
            z_ucb = z_ucb_seq[j]
            r = PR.reward_by_unit_counts(ucb, z_ucb)
            r_traj.append(r)

        hamming_rewards.append(r_traj)
    '''

    print("trajectories.build_order", trajectories.build_order) if debug else None
    print("trajectories.z_build_order", trajectories.z_build_order) if debug else None
    print("trajectories.unit_counts", trajectories.unit_counts) if debug else None
    print("trajectories.z_unit_counts", trajectories.z_unit_counts) if debug else None

    weight_leven = 1.0
    weight_hamming = 1.0

    rewards_traj = []
    for t1, t2, t3, t4 in zip(trajectories.build_order, trajectories.z_build_order,
                              trajectories.unit_counts, trajectories.z_unit_counts):
        # Calculate the Levenshtein distance,
        leven_batch = []
        for b1, b2 in zip(t1, t2):
            r = PR.reward_by_build_order(b1, b2)
            leven_batch.append(r)
        print('leven_batch:', leven_batch) if debug else None

        # Calculate the Hamming distance,
        hamming_batch = []
        for b1, b2 in zip(t3, t4):
            r = PR.reward_by_unit_counts(b1, b2)
            hamming_batch.append(r)
        print('hamming_batch:', hamming_batch) if debug else None

        reward_batch = []
        for r1, r2 in zip(leven_batch, hamming_batch):
            reward_batch.append(weight_leven * r1 + weight_hamming * r2)
        print('reward_batch:', reward_batch) if debug else None

        rewards_traj.append(reward_batch)

    print('rewards_traj:', rewards_traj) if debug else None 
    rewards_numpy = np.array(rewards_traj)
    rewards_tensor = torch.tensor(rewards_numpy, dtype=torch.float32)

    return rewards_tensor


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
    # trajectories shape: list of trajectory
    # target_logits: ArgsActionLogits
    target_logits, baselines = agent.unroll(trajectories)

    the_action_type = target_logits.action_type
    print("the_action_type:", the_action_type) if debug else None
    print("the_action_type.shape:", the_action_type.shape) if debug else None

    print("baselines.shape:", baselines.shape) if debug else None

    # note, we change the structure of the trajectories
    # note, the size is all list
    # shape: [batch_size x dict_name x seq_size]
    trajectories = U.stack_namedtuple(trajectories) 
    # shape: [dict_name x batch_size x seq_size]
    print("trajectories.reward", trajectories.reward) if debug else None   

    # shape: [dict_name x batch_size x seq_size]
    trajectories = U.namedtuple_zip(trajectories) 
    # shape: [dict_name x seq_size x batch_size]
    print("trajectories.reward", trajectories.reward) if debug else None   

    loss_actor_critic = 0.
    # We use a number of actor-critic losses - one for the winloss baseline, which
    # outputs the probability of victory, and one for each pseudo-reward
    # associated with following the human strategy statistic z.
    # See the paper methods and detailed_architecture.txt for more details.
    BASELINE_COSTS_AND_REWARDS = get_baseline_hyperparameters()
    if True:  # wait for furture fullfilled
        for baseline, costs_and_rewards in zip(baselines, BASELINE_COSTS_AND_REWARDS):
            # baseline is for caluculation in td_lambda and vtrace_pg
            # costs_and_rewards are only weight for loss

            # reward_name is for pseudoreward
            # pg_cost is for vtrace_pg_loss
            # baseline_cost is for lambda_loss
            pg_cost, baseline_cost, reward_name = costs_and_rewards

            print("reward_name:", reward_name) if debug else None
            rewards = compute_pseudoreward(trajectories, reward_name)

            # The action_type argument, delay, and all other arguments are separately updated 
            # using a separate ("split") VTrace Actor-Critic losses. The weighting of these 
            # updates will be considered 1.0. action_type, delay, and other arguments are 
            # also similarly separately updated using UPGO, in the same way as the VTrace 
            # Actor-Critic loss, with relative weight 1.0. 

            lambda_loss = td_lambda_loss(baseline, rewards, trajectories)
            print("lambda_loss:", lambda_loss) if debug else None
            loss_actor_critic += (baseline_cost * lambda_loss)

            # we add the split_vtrace_pg_loss
            pg_loss = split_vtrace_pg_loss(target_logits, baseline, rewards, trajectories)
            print("pg_loss:", pg_loss) if debug else None
            loss_actor_critic += (pg_cost * pg_loss)

    # Note: upgo_loss has only one baseline which is just for winloss 
    # AlphaStar: loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines.winloss_baseline, trajectories)
    UPGO_WEIGHT = 1.0
    loss_upgo = UPGO_WEIGHT * split_upgo_loss(target_logits, baselines[0], trajectories)

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

    loss_he = human_policy_kl_loss(target_logits.action_type, teacher_logits_action_type, ACTION_TYPE_KL_COST)

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

    loss_ent = ENT_WEIGHT * entropy_loss_for_all_arguments(target_logits, trajectories.masks)

    #print("stop", len(stop))

    loss_all = loss_actor_critic + loss_upgo + loss_he + loss_ent

    print("loss_actor_critic:", loss_actor_critic) if debug else None
    print("loss_upgo:", loss_upgo) if debug else None
    print("loss_he:", loss_he) if debug else None
    print("loss_ent:", loss_ent) if debug else None
    print("loss_all:", loss_all) if debug else None

    return loss_all
