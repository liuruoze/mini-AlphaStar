#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the sc2 agents, may be modified"

# modified from pysc2 code

import gc

from time import time

import numpy as np
import torch

from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.env import environment as E

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.rl.state import MsState
from alphastarmini.core.rl.action import ArgsAction, ArgsActionLogits

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


class BaseAgent(object):
    """A base agent to write custom scripted agents.

    It can also act as a passive agent that does nothing but no-ops.
    """

    def __init__(self, name):
        # agent name
        self.name = name
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward

        func_call = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        return func_call


class RandomAgent(BaseAgent):
    """A random agent for starcraft."""

    def __init__(self, name, raw=True):
        super(RandomAgent, self).__init__(name=name)

        # whether use raw actions
        self.use_raw = raw

    def step(self, obs):
        noop_func_call = super(RandomAgent, self).step(obs)

        print('name:', self.name) if debug else None

        observation = obs.observation
        if not self.use_raw:
            # print('obs.observation.available_actions:', obs.observation.available_actions)
            function_id = np.random.choice(observation.available_actions)
        else:
            ids = [f.id for f in actions.RAW_FUNCTIONS if f.avail_fn]
            function_id = np.random.choice(ids)

        print('function_id:', function_id) if debug else None
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        print('args:', args) if debug else None

        func_call = actions.FunctionCall.init_with_validation(function_id, args, raw=self.use_raw)
        print('func_call:', func_call) if debug else None

        if func_call is not None:
            return func_call
        else:
            # only return base action (no-op)
            return noop_func_call


class AlphaStarAgent(RandomAgent):
    """A alphastar agent for starcraft.

    Demonstrates agent interface.

    In practice, this needs to be instantiated with the right neural network
    architecture.
    """

    def __init__(self, name, race=sc2_env.Race.protoss, initial_weights=None):
        # AlphaStarAgent use raw actions
        super(AlphaStarAgent, self).__init__(name=name, raw=True)

        self.race = race
        self.weights = initial_weights

        # initial the neural network agent with the initial weights
        if self.weights is not None:
            self.agent_nn = Agent(self.weights)
        else:
            self.agent_nn = Agent()
            self.weights = self.agent_nn.get_weights()

        # init lstm hidden state
        self.memory_state = self.initial_state()

    def set_rl_training(self, staus):
        self.agent_nn.set_rl_training(staus)

    def initial_state(self):
        """Returns the hidden state of the agent for the start of an episode."""
        # Network details elided.
        initial_state = self.agent_nn.init_hidden_state()

        return initial_state

    def reset(self):
        # init lstm hidden state
        self.episodes += 1
        self.memory_state = self.initial_state()

    def set_weights(self, weights):
        self.weights = weights
        self.agent_nn.set_weights(weights)

    def get_weights(self):
        #assert self.weights == self.agent_nn.get_weights()
        if self.agent_nn is not None:
            return self.agent_nn.get_weights()
        else:
            return None

    def get_parameters(self):
        return self.agent_nn.model.parameters()

    def get_steps(self):
        """How many agent steps the agent has been trained for."""

        return self.steps

    def preprocess_state_all(self, obs):
        if isinstance(obs, E.TimeStep):
            obs = obs.observation

        state = Agent.preprocess_state_all(obs=obs)

        return state

    def step_nn(self, observation, last_state):
        """Performs inference on the observation, given hidden state last_state."""

        state = Agent.preprocess_state_all(obs=observation)
        device = self.agent_nn.device()
        state.to(device)

        action_logits, action, hidden_state, select_units_num = self.agent_nn.action_logits_by_state(state, single_inference=True,
                                                                                                     hidden_state=last_state)
        del state

        return action, action_logits, hidden_state, select_units_num

    def step(self, obs):
        # note here obs is actually timestep 
        rand_func_call = super(AlphaStarAgent, self).step(obs)
        print('name:', self.name) if debug else None

        # note someimes obs is timestep 
        if isinstance(obs, E.TimeStep):
            obs = obs.observation

        action, _, self.memory_state, select_units_num = self.step_nn(obs, self.memory_state)

        if action is not None:
            func_call = self.agent_nn.action_to_func_call(action, select_units_num, self.action_spec)
            del action

            return func_call
        else:
            # only return random action
            return rand_func_call

    def step_logits(self, obs, last_state):       
        print('name:', self.name) if debug else None

        # note someimes obs is timestep 
        if isinstance(obs, E.TimeStep):
            obs = obs.observation

        action, action_logits, new_state, select_units_num = self.step_nn(observation=obs, last_state=last_state)

        func_call = self.agent_nn.action_to_func_call(action, select_units_num, self.action_spec)

        del select_units_num

        return func_call, action, action_logits, new_state

    def step_from_state(self, state, hidden_state, obs=None):       
        device = self.agent_nn.device()
        state.to(device)

        action_logits, action, hidden_state, select_units_num, entity_num = self.agent_nn.action_logits_by_state(state, 
                                                                                                                 single_inference=True,
                                                                                                                 hidden_state=hidden_state,
                                                                                                                 obs=obs)

        func_call = self.agent_nn.action_to_func_call(action, select_units_num, self.action_spec)

        del state

        return func_call, action, action_logits, hidden_state, select_units_num, entity_num

    def step_based_on_actions(self, state, hidden_state, action_gt, gt_select_units_num):  
        device = self.agent_nn.device()

        state.to(device)
        action_gt.to(device)
        hidden_state = tuple(h.to(device) for h in hidden_state)

        gt_select_units_num = gt_select_units_num.to(device)

        action_logits, select_units_num, hidden_state = self.agent_nn.action_logits_based_on_actions(state, 
                                                                                                     action_gt=action_gt, 
                                                                                                     gt_select_units_num=gt_select_units_num, 
                                                                                                     single_inference=True,
                                                                                                     hidden_state=hidden_state)
        del state, action_gt, gt_select_units_num, hidden_state

        return action_logits

    def rl_unroll(self, trajectories, use_opponent_state=True, show=False):
        """Unrolls the network over the trajectory.

        The actions taken by the agent and the initial state of the unroll are
        dictated by trajectory.
        """
        ACTION_FIELDS = [
            'action_type',  
            'delay',
            'queue',
            'units',
            'target_unit',
            'target_location',
        ]

        device = self.agent_nn.device()
        print("unroll device:", device) if debug else None

        # trajectories shape: list of trajectory
        policy_logits = None
        baselines = None

        hidden_traj = []  # traj.memory
        cell_traj = []

        state_traj = []  # traj.state
        baseline_state_traj = []  # traj.baseline_state
        baseline_state_op_traj = []  # traj.baseline_state_op
        actions_traj = []
        select_units_num_traj = []
        entity_nums_traj = []

        batch_size = len(trajectories)
        for i, traj in enumerate(trajectories):
            seq_length = len(traj.state)

            hidden = list(zip(*traj.memory))[0]
            print('hidden', hidden) if debug else None

            hidden_traj.extend(hidden)

            cell = list(zip(*traj.memory))[1]
            print('cell', cell) if debug else None

            cell_traj.extend(cell)
            del hidden, cell

            state_traj.extend(traj.state)
            baseline_state_traj.extend(traj.baseline_state)
            if use_opponent_state:
                baseline_state_op_traj.extend(traj.baseline_state_op)

            actions_traj.extend(traj.action)
            select_units_num_traj.extend(traj.player_select_units_num)
            entity_nums_traj.extend(traj.entity_num)
            del traj

        print('batch_size', batch_size) if debug else None
        print('seq_length', seq_length) if debug else None

        # add the state
        entity_state_list = []
        statistical_state_list = []
        map_state_list = []
        for s in state_traj:
            entity_state_list.append(s.entity_state)
            statistical_state_list.append(s.statistical_state)
            map_state_list.append(s.map_state)

        action_type_list = []
        delay_list = []
        queue_list = []
        units_list = []
        target_unit_list = []
        target_location_list = []
        for a in actions_traj:
            action_type_list.append(a.action_type)
            delay_list.append(a.delay)
            queue_list.append(a.queue)
            units_list.append(a.units)
            target_unit_list.append(a.target_unit)
            target_location_list.append(a.target_location)

        entity_state_all = torch.cat([l.to(device) for l in entity_state_list], dim=0)
        entity_state_all = entity_state_all.view(batch_size, seq_length, *tuple(entity_state_all.shape[1:]))
        del entity_state_list, state_traj

        statistical_state_all = [torch.cat([l.to(device) for l in statis], dim=0) for statis in zip(*statistical_state_list)]
        statistical_state_all = [l.view(batch_size, seq_length, *tuple(l.shape[1:])) for l in statistical_state_all]
        del statistical_state_list

        map_state_all = torch.cat([l.to(device) for l in map_state_list], dim=0).to(device)
        map_state_all = map_state_all.view(batch_size, seq_length, *tuple(map_state_all.shape[1:]))
        del map_state_list

        select_units_num_all = torch.cat([l.to(device) for l in select_units_num_traj], dim=0)
        select_units_num_all = select_units_num_all.view(batch_size, seq_length, *tuple(select_units_num_all.shape[1:]))
        del select_units_num_traj

        entity_nums_all = torch.cat([l.to(device) for l in entity_nums_traj], dim=0)
        entity_nums_all = entity_nums_all.view(batch_size, seq_length, *tuple(entity_nums_all.shape[1:]))
        del entity_nums_traj

        # # note, hidden has the size of [num_of_lstm_layers, batch_size, hidden_size]
        hidden_all = torch.cat([l.to(device) for l in hidden_traj], dim=1)
        hidden_all = hidden_all.transpose(0, 1)
        hidden_all = hidden_all.view(batch_size, seq_length, *tuple(hidden_all.shape[1:]))
        del hidden_traj

        cell_all = torch.cat([l.to(device) for l in cell_traj], dim=1)
        cell_all = cell_all.transpose(0, 1)
        cell_all = cell_all.view(batch_size, seq_length, *tuple(cell_all.shape[1:]))
        del cell_traj

        action_type_all = torch.cat([l.to(device) for l in action_type_list], dim=0)
        action_type_all = action_type_all.view(batch_size, seq_length, *tuple(action_type_all.shape[1:]))
        del action_type_list

        delay_all = torch.cat([l.to(device) for l in delay_list], dim=0)
        delay_all = delay_all.view(batch_size, seq_length, *tuple(delay_all.shape[1:]))
        del delay_list

        queue_all = torch.cat([l.to(device) for l in queue_list], dim=0)
        queue_all = queue_all.view(batch_size, seq_length, *tuple(queue_all.shape[1:]))
        del queue_list

        units_all = torch.cat([l.to(device) for l in units_list], dim=0)
        units_all = units_all.view(batch_size, seq_length, *tuple(units_all.shape[1:]))
        del units_list

        target_unit_all = torch.cat([l.to(device) for l in target_unit_list], dim=0)
        target_unit_all = target_unit_all.view(batch_size, seq_length, *tuple(target_unit_all.shape[1:]))
        del target_unit_list

        target_location_all = torch.cat([l.to(device) for l in target_location_list], dim=0)
        target_location_all = target_location_all.view(batch_size, seq_length, *tuple(target_location_all.shape[1:]))
        del target_location_list

        baseline_state_all = [torch.cat([l.to(device) for l in statis], dim=0) for statis in zip(*baseline_state_traj)]
        baseline_state_all = [l.view(batch_size, seq_length, *tuple(l.shape[1:])) for l in baseline_state_all]
        del baseline_state_traj

        if use_opponent_state:
            baseline_state_op_all = [torch.cat([l.to(device) for l in statis], dim=0) for statis in zip(*baseline_state_op_traj)]
            baseline_state_op_all = [l.view(batch_size, seq_length, *tuple(l.shape[1:])) for l in baseline_state_op_all]
            del baseline_state_op_traj

        logits_list = []
        baseline_list = []

        for i in range(seq_length):
            entity_state = entity_state_all[:, i].float()
            statistical_state = [s[:, i].float() for s in statistical_state_all]
            map_state = map_state_all[:, i].float()

            state = MsState(entity_state=entity_state, statistical_state=statistical_state, map_state=map_state)

            del entity_state, statistical_state, map_state

            select_units_num = select_units_num_all[:, i]
            hidden = hidden_all[:, i].transpose(0, 1).contiguous()  # .detach()
            cell = cell_all[:, i].transpose(0, 1).contiguous()  # .detach()
            memory = tuple([hidden, cell])
            del hidden, cell

            baseline_state = [s[:, i].float() for s in baseline_state_all]
            if use_opponent_state:
                baseline_opponent_state = [s[:, i].float() for s in baseline_state_op_all]
            else:
                baseline_opponent_state = None

            action_type = action_type_all[:, i]
            delay = delay_all[:, i]
            queue = queue_all[:, i]
            units = units_all[:, i]
            target_unit = target_unit_all[:, i]
            target_location = target_location_all[:, i]

            action = ArgsAction(action_type=action_type, delay=delay, queue=queue,
                                units=units, target_unit=target_unit, target_location=target_location)

            del action_type, delay, queue, units, target_unit, target_location

            baselines, logits, _, _ = self.agent_nn.action_logits_on_actions_for_unroll(state, action, select_units_num, 
                                                                                        hidden_state=memory, 
                                                                                        batch_size=batch_size, 
                                                                                        sequence_length=1,
                                                                                        baseline_state=baseline_state, 
                                                                                        baseline_opponent_state=baseline_opponent_state,
                                                                                        show=show)
            del state, action, select_units_num, memory, baseline_state, baseline_opponent_state

            logits_list.append(logits)
            baseline_list.append(baselines)

            del baselines, logits

        del action_type_all, delay_all, queue_all, units_all, target_unit_all, target_location_all
        del baseline_state_all, hidden_all, cell_all
        if use_opponent_state:
            del baseline_state_op_all

        # concate them in the sequence dim
        action_type_list = []
        delay_list = []
        queue_list = []
        units_list = []
        target_unit_list = []
        target_location_list = []
        for a in logits_list:
            action_type_list.append(a.action_type.unsqueeze(1))
            delay_list.append(a.delay.unsqueeze(1))
            queue_list.append(a.queue.unsqueeze(1))
            units_list.append(a.units.unsqueeze(1))
            target_unit_list.append(a.target_unit.unsqueeze(1))
            target_location_list.append(a.target_location.unsqueeze(1))

        action_type_logits = torch.cat(action_type_list, dim=1)
        delay_logits = torch.cat(delay_list, dim=1)
        queue_logits = torch.cat(queue_list, dim=1)
        units_logits = torch.cat(units_list, dim=1)
        target_unit_logits = torch.cat(target_unit_list, dim=1)
        target_location_logits = torch.cat(target_location_list, dim=1)

        policy_logits = ArgsActionLogits(action_type=action_type_logits, delay=delay_logits, queue=queue_logits,
                                         units=units_logits, target_unit=target_unit_logits, 
                                         target_location=target_location_logits)

        baseline_list = [torch.cat(l, dim=1) for l in zip(*baseline_list)]

        del action_type_logits, delay_logits, queue_logits, units_logits, target_unit_logits, target_location_logits
        del action_type_list, delay_list, queue_list, units_list, target_unit_list, target_location_list
        del logits_list

        return policy_logits, baseline_list, select_units_num_all, entity_nums_all
