#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the sc2 agents, may be modified"

# modified from pysc2 code

from time import time

import numpy as np
import torch

from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.env import environment as E

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.rl.state import MsState

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

        t = time()

        state = Agent.preprocess_state_all(obs=observation)

        print('step_nn, t1', time() - t) if speed else None
        t = time()

        device = self.agent_nn.device()
        state.to(device)

        print('step_nn, t2', time() - t) if speed else None
        t = time()

        action_logits, action, hidden_state, select_units_num = self.agent_nn.action_logits_by_state(state, single_inference=True,
                                                                                                     hidden_state=last_state)
        print('step_nn, t3', time() - t) if speed else None
        t = time()

        return action, action_logits, hidden_state, select_units_num

    def step(self, obs):
        # note here obs is actually timestep 
        rand_func_call = super(AlphaStarAgent, self).step(obs)
        print('name:', self.name) if debug else None

        # note someimes obs is timestep 
        if isinstance(obs, E.TimeStep):
            obs = obs.observation

        action, _, self.memory_state, select_units_num = self.step_nn(obs, self.memory_state)

        if actions is not None:
            # we use single_inference here
            #assert len(actions) == 1
            #action = actions[0]
            func_call = self.agent_nn.action_to_func_call(action, select_units_num, self.action_spec)
            return func_call
        else:
            # only return random action
            return rand_func_call

    def step_logits(self, obs, last_state):       
        print('name:', self.name) if debug else None

        # note someimes obs is timestep 
        if isinstance(obs, E.TimeStep):
            obs = obs.observation

        t = time()    

        action, action_logits, new_state, select_units_num = self.step_nn(observation=obs, last_state=last_state)

        print('step_logits, t1', time() - t) if speed else None
        t = time()

        func_call = self.agent_nn.action_to_func_call(action, select_units_num, self.action_spec)

        print('step_logits, t2', time() - t) if speed else None
        t = time()

        return func_call, action, action_logits, new_state

    def step_from_state(self, state, hidden_state):       
        device = self.agent_nn.device()
        state.to(device)

        action_logits, action, hidden_state, select_units_num = self.agent_nn.action_logits_by_state(state, 
                                                                                                     single_inference=True,
                                                                                                     hidden_state=hidden_state)

        func_call = self.agent_nn.action_to_func_call(action, select_units_num, self.action_spec)

        return func_call, action, action_logits, hidden_state, select_units_num

    def step_based_on_actions(self, state, hidden_state, action_gt, gt_select_units_num):  
        action_logits, select_units_num, hidden_state = self.agent_nn.action_logits_based_on_actions(state, 
                                                                                                     action_gt=action_gt, 
                                                                                                     gt_select_units_num=gt_select_units_num, 
                                                                                                     single_inference=True,
                                                                                                     hidden_state=hidden_state)

        return action_logits, select_units_num, hidden_state

    def unroll(self, trajectories, use_opponent_state=False):
        """Unrolls the network over the trajectory.

        The actions taken by the agent and the initial state of the unroll are
        dictated by trajectory.
        """

        # trajectories shape: list of trajectory
        policy_logits = None
        baselines = None

        initial_memory_list = []

        state_traj = []  # traj.state
        baseline_state_traj = []  # traj.baseline_state
        baseline_state_op_traj = []  # traj.baseline_state_op

        # TODO: improve it
        for i, traj in enumerate(trajectories):
            # add the initial memory state          
            memory_seq = traj.memory
            initial_memory = memory_seq[0]
            initial_memory_list.append(initial_memory)

            state_traj.extend(traj.state)
            baseline_state_traj.extend(traj.baseline_state)
            if use_opponent_state:
                baseline_state_op_traj.extend(traj.baseline_state_op)

            # add the state
        entity_state_list = []
        statistical_state_list = []
        map_state_list = []
        for s in state_traj:
            entity_state_list.append(s.entity_state)
            statistical_state_list.append(s.statistical_state)
            map_state_list.append(s.map_state)

        entity_state_all = torch.cat(entity_state_list, dim=0)
        statistical_state_all = [torch.cat(statis, dim=0) for statis in zip(*statistical_state_list)]
        map_state_all = torch.cat(map_state_list, dim=0)

        state_all = MsState(entity_state=entity_state_all, statistical_state=statistical_state_all, map_state=map_state_all)

        device = self.agent_nn.device()
        print("unroll device:", device) if debug else None

        print("initial_memory.shape:", initial_memory_list[0][0].shape) if debug else None
        # note the bacth size is in the second dim of hidden state
        initial_memory_state = [torch.cat(l, dim=1) for l in zip(*initial_memory_list)]

        baseline_state_all = [torch.cat(statis, dim=0) for statis in zip(*baseline_state_traj)]
        print("baseline_state_all.shape:", baseline_state_all[0].shape) if debug else None

        if use_opponent_state:
            baseline_state_op_all = [torch.cat(statis, dim=0) for statis in zip(*baseline_state_op_traj)]
        # change to device
        state_all.to(device)  # note: MsStata.to(device) in place operation
        initial_memory_state = [l.float().to(device) for l in initial_memory_state]
        baseline_state_all = [l.float().to(device) for l in baseline_state_all]

        if use_opponent_state:
            baseline_state_op_all = [l.float().to(device) for l in baseline_state_op_all]
        else:
            baseline_state_op_all = None

        # shape [batch_seq_size, embedding_size]
        baseline_list, policy_logits, select_units_num = self.agent_nn.unroll_traj(state_all=state_all, 
                                                                                   initial_state=initial_memory_state, 
                                                                                   baseline_state=baseline_state_all, 
                                                                                   baseline_opponent_state=baseline_state_op_all)

        baselines = [baseline.reshape(AHP.batch_size, AHP.sequence_length).transpose(0, 1) for baseline in baseline_list]

        return policy_logits, baselines, select_units_num
