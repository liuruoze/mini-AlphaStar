"""Library for RL losses."""
import collections

import numpy as np


OBSERVATION_FIELDS = [
    'game_seconds',  # Game timer in seconds.
]

ACTION_FIELDS = [
    'action_type',  # Action taken.
    # Other fields, e.g. arguments, repeat, delay, queued.
]

TRAJECTORY_FIELDS = [
    'observation',  # Player observation.
    'opponent_observation',  # Opponent observation, used for value network.
    'state',  # State of the agent (used for initial LSTM state).
    'z',  # Conditioning information for the policy.
    'is_final',  # If this is the last step.
    # namedtuple of masks for each action component. 0/False if final_step of
    # trajectory, or the argument is not used; else 1/True.
    'masks',
    'action',  # Action taken by the agent.
    'behavior_logits',  # namedtuple of logits of the behavior policy.
    'teacher_logits',  # namedtuple of logits of the supervised policy.
    'reward',  # Reward for the agent after taking the step.
]

Trajectory = collections.namedtuple('Trajectory', TRAJECTORY_FIELDS)


def log_prob(actions, logits):
    """Returns the log probability of taking an action given the logits."""
    # Equivalent to tf.sparse_softmax_cross_entropy_with_logits.


def is_sampled(z):
    """Takes a tensor of zs. Returns a mask indicating which z's are sampled."""


def filter_by(action_fields, target):
    """Returns the subset of `target` corresponding to `action_fields`.

    Autoregressive actions are composed of many logits.  We often want to select a
    subset of these logits.

    Args:
      action_fields: One of 'action_type', 'delay', or 'arguments'.
      target: A list of tensors corresponding to the SC2 action spec.
    Returns:
      A list corresponding to a subset of `target`, with only the tensors relevant
      to `action_fields`.
    """


def compute_over_actions(f, *args):
    """Runs f over all elements in the lists composing *args.

    Autoregressive actions are composed of many logits. We run losses functions
    over all sets of logits.
    """
    return sum(f(*a) for a in zip(*args))


def entropy(policy_logits):
    policy = softmax(policy_logits)
    log_policy = logsoftmax(policy_logits)
    ent = np.sum(-policy * log_policy, axis=-1)  # Aggregate over actions.
    # Normalize by actions available.
    normalized_entropy = ent / np.log(policy_logits.shape[-1])
    return normalized_entropy


def entropy_loss(policy_logits, masks):
    """Computes the entropy loss for a set of logits.

    Args:
      policy_logits: namedtuple of the logits for each policy argument.
        Each shape is [..., N_i].
      masks: The masks. Each shape is policy_logits.shape[:-1].
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """
    return np.mean(compute_over_actions(entropy, policy_logits, masks))


def kl(student_logits, teacher_logits, mask):
    s_logprobs = logsoftmax(student_logits)
    t_logprobs = logsoftmax(teacher_logits)
    teacher_probs = softmax(teacher_logits)
    return teacher_probs * (t_logprobs - s_logprobs) * mask


def human_policy_kl_loss(trajectories, kl_cost, action_type_kl_cost):
    """Computes the KL loss to the human policy.

    Args:
      trajectories: The trajectories.
      kl_cost: A float; the weighting to apply to the KL cost to the human policy.
      action_type_kl_cost: Additional cost applied to action_types for
        conditioned policies.
    Returns:
      Per-example entropy loss, as an array of shape policy_logits.shape[:-1].
    """
    student_logits = trajectories.behavior_logits
    teacher_logits = trajectories.teacher_logits
    masks = trajectories.masks
    kl_loss = compute_over_actions(kl, student_logits, teacher_logits, masks)

    # We add an additional KL-loss on only the action_type for the first 4 minutes
    # of each game if z is sampled.
    game_seconds = trajectories.observation.game_seconds
    action_type_mask = masks.action_type & (game_seconds > 4 * 60)
    action_type_mask = action_type_mask & is_sampled(trajectories.z)
    action_type_loss = kl(student_logits.action_type, teacher_logits.action_type,
                          action_type_mask)
    return (kl_cost * np.mean(kl_loss)
            + action_type_kl_cost * np.mean(action_type_loss))


def lambda_returns(values_tp1, rewards, discounts, lambdas):
    """Computes lambda returns.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py#L74
    """


def generalized_lambda_returns(rewards,
                               pcontinues,
                               values,
                               bootstrap_value,
                               lambda_=1,
                               name="generalized_lambda_returns"):
    """Computes lambda-returns along a batch of (chunks of) trajectories.
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
    values.get_shape().assert_has_rank(2)
    rewards.get_shape().assert_has_rank(2)
    pcontinues.get_shape().assert_has_rank(2)
    bootstrap_value.get_shape().assert_has_rank(1)
    scoped_values = [rewards, pcontinues, values, bootstrap_value, lambda_]
    with tf.name_scope(name, values=scoped_values):
        if lambda_ == 1:
            # This is actually equivalent to the branch below, just an optimisation
            # to avoid unnecessary work in this case:
            return sequence_ops.scan_discounted_sum(
                rewards,
                pcontinues,
                initial_value=bootstrap_value,
                reverse=True,
                back_prop=False,
                name="multistep_returns")
        else:
            v_tp1 = tf.concat(
                axis=0, values=[values[1:, :],
                                tf.expand_dims(bootstrap_value, 0)])
            # `back_prop=False` prevents gradients flowing into values and
            # bootstrap_value, which is what you want when using the bootstrapped
            # lambda-returns in an update as targets for values.
            return sequence_ops.multistep_forward_view(
                rewards,
                pcontinues,
                v_tp1,
                lambda_,
                back_prop=False,
                name="generalized_lambda_returns")


def vtrace_advantages(clipped_rhos, rewards, discounts, values, bootstrap_value):
    """Computes v-trace return advantages.

    Refer to the following for a similar function:
    https://github.com/deepmind/trfl/blob/40884d4bb39f99e4a642acdbe26113914ad0acec/trfl/vtrace_ops.py#L154
    """


def vtrace_from_importance_weights(
        log_rhos, discounts, rewards, values, bootstrap_value,
        clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0,
        name='vtrace_from_importance_weights'):
    r"""V-trace from log importance weights.
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
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
    if clip_rho_threshold is not None:
        clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold,
                                                  dtype=tf.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(clip_pg_rho_threshold,
                                                     dtype=tf.float32)

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims  # Usually 2.
    values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)
    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)

    with tf.name_scope(name, values=[
            log_rhos, discounts, rewards, values, bootstrap_value]):
        rhos = tf.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
        else:
            clipped_rhos = rhos

        cs = tf.minimum(1.0, rhos, name='cs')
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = tf.concat(
            [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        # Note that all sequences are reversed, computation starts from the back.
        sequences = (
            tf.reverse(discounts, axis=[0]),
            tf.reverse(cs, axis=[0]),
            tf.reverse(deltas, axis=[0]),
        )
        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.

        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        vs_minus_v_xs = tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            name='scan')
        # Reverse the results back to original order.
        vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0], name='vs_minus_v_xs')

        # Add V(x_s) to get v_s.
        vs = tf.add(vs_minus_v_xs, values, name='vs')

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat([
            vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos,
                                         name='clipped_pg_rhos')
        else:
            clipped_pg_rhos = rhos
        pg_advantages = (
            clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=tf.stop_gradient(vs),
                             pg_advantages=tf.stop_gradient(pg_advantages))


def td_lambda_loss(baselines, rewards, trajectories):
    discounts = ~trajectories.is_final[:-1]
    returns = lambda_returns(baselines[1:], rewards, discounts, lambdas=0.8)
    returns = stop_gradient(returns)
    return 0.5 * np.mean(np.square(returns - baselines[:-1]))


def policy_gradient_loss(logits, actions, advantages, mask):
    """Helper function for computing policy gradient loss for UPGO and v-trace."""
    action_log_prob = log_prob(actions, logits)
    advantages = stop_gradient(advantages)
    return mask * advantages * action_log_prob


def compute_unclipped_logrho(behavior_logits, target_logits, actions):
    """Helper function for compute_importance_weights."""
    return log_prob(actions, target_logits) - log_prob(actions, behavior_logits)


def compute_importance_weights(behavior_logits, target_logits, actions):
    """Computes clipped importance weights."""
    logrho = compute_over_actions(compute_unclipped_logrho, behavior_logits,
                                  target_logits, actions)
    return np.minimum(1., np.exp(logrho))


def vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                   action_fields):
    """Computes v-trace policy gradient loss. Helper for split_vtrace_pg_loss."""
    # Remove last timestep from trajectories and baselines.
    trajectories = Trajectory(*tuple(t[:-1] for t in trajectories))
    rewards = rewards[:-1]
    values = baselines[:-1]

    # Filter for only the relevant actions/logits/masks.
    target_logits = filter_by(action_fields, target_logits)
    behavior_logits = filter_by(action_fields, trajectories.behavior_logits)
    actions = filter_by(action_fields, trajectories.actions)
    masks = filter_by(action_fields, trajectories.masks)

    # Compute and return the v-trace policy gradient loss for the relevant subset
    # of logits.
    clipped_rhos = compute_importance_weights(behavior_logits, target_logits,
                                              actions)
    weighted_advantage = vtrace_advantages(clipped_rhos, rewards,
                                           trajectories.discounts, values,
                                           baselines[-1])
    weighted_advantage = [weighted_advantage] * len(target_logits)
    return compute_over_actions(policy_gradient_loss, target_logits,
                                actions, weighted_advantage, masks)


def split_vtrace_pg_loss(target_logits, baselines, rewards, trajectories):
    """Computes the split v-trace policy gradient loss.

    We compute the policy loss (and therefore update, via autodiff) separately for
    the action_type, delay, and arguments. Each of these component losses are
    weighted equally.
    """
    loss = 0.
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                           'action_type')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                           'delay')
    loss += vtrace_pg_loss(target_logits, baselines, rewards, trajectories,
                           'arguments')
    return loss


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
    next_values = np.concatenate(
        values[1:], np.expand_dims(bootstrap, axis=0), axis=0)
    # Upgo can be viewed as a lambda return! The trace continues (i.e. lambda =
    # 1.0) if r_t + V_tp1 > V_t.
    lambdas = (rewards + discounts * next_values) >= values
    # Shift lambdas left one slot, such that V_t matches indices with lambda_tp1.
    lambdas = np.concatenate(lambdas[1:], np.ones_like(lambdas[-1:]), axis=0)
    return lambda_returns(next_values, rewards, discounts, lambdas)


def split_upgo_loss(target_logits, baselines, trajectories):
    """Computes split UPGO policy gradient loss.

    See split_vtrace_pg_loss docstring for details on split updates.
    See Methods for details on UPGO.
    """
    # Remove last timestep from trajectories and baselines.
    trajectories = Trajectory(*tuple(t[:-1] for t in trajectories))
    values = baselines[:-1]
    returns = upgo_returns(values, trajectories.rewards, trajectories.discounts,
                           baselines[-1])

    # Compute the UPGO loss for each action subset.
    loss = 0.
    for action_fields in ['action_type', 'delay', 'arguments']:
        split_target_logits = filter_by(action_fields, target_logits)
        behavior_logits = filter_by(action_fields, trajectories.behavior_logits)
        actions = filter_by(action_fields, trajectories.actions)
        masks = filter_by(action_fields, trajectories.masks)

        importance_weights = compute_importance_weights(behavior_logits,
                                                        split_target_logits,
                                                        actions)
        weighted_advantage = (returns - values) * importance_weights
        weighted_advantage = [weighted_advantage] * len(split_target_logits)
        loss += compute_over_actions(policy_gradient_loss, split_target_logits,
                                     actions, weighted_advantage, masks)
    return loss


def compute_pseudoreward(trajectories, reward_name):
    """Computes the relevant pseudoreward from trajectories.

    See Methods and detailed_architecture.txt for details.
    """


def loss_function(agent, trajectories):
    """Computes the loss of trajectories given weights."""
    # All ALL_CAPS variables are constants.
    target_logits, baselines = agent.unroll(trajectories)

    loss_actor_critic = 0.
    # We use a number of actor-critic losses - one for the winloss baseline, which
    # outputs the probability of victory, and one for each pseudo-reward
    # associated with following the human strategy statistic z.
    # See the paper methods and detailed_architecture.txt for more details.
    for baseline, costs_and_rewards in zip(baselines,
                                           BASELINE_COSTS_AND_REWARDS):
        pg_cost, baseline_cost, reward_name = costs_and_rewards
        rewards = compute_pseudoreward(trajectories, reward_name)
        loss_actor_critic += (
            baseline_cost * td_lambda_loss(baseline, rewards, trajectories))
        loss_actor_critic += (
            pg_cost
            * split_vtrace_pg_loss(target_logits, baseline, rewards, trajectories))

    loss_upgo = UPGO_WEIGHT * split_upgo_loss(
        target_logits, baselines.winloss_baseline, trajectories)
    loss_he = human_policy_kl_loss(trajectories, KL_COST, ACTION_TYPE_KL_COST)

    loss_ent = entropy_loss(trajectories.behavior_logits, trajectories.masks)
    loss_ent = loss_ent * ENT_WEIGHT

    return loss_actor_critic + loss_upgo + loss_he + loss_ent
