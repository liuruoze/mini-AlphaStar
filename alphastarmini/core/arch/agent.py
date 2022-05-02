#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Agent."

import gc

from time import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib import actions as A

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.arch.entity_encoder import EntityEncoder, Entity

from alphastarmini.core.rl.action import ArgsAction, ArgsActionLogits
from alphastarmini.core.rl.state import MsState

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import MiniStar_Arch_Hyper_Parameters as MAHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

from pysc2.lib.units import get_unit_type

__author__ = "Ruo-Ze Liu"

debug = False
speed = False


class Agent(object):
    def __init__(self, weights=None):
        self.model = ArchModel()
        self.hidden_state = None

        if weights is not None:
            self.set_weights(weights)

    # def __init__(self, model, idx):
    #     self.model = model
    #     self.hidden_state = None
    #     self.idx = idx

    def init_hidden_state(self):
        if self.model is not None:
            return self.model.init_hidden_state()
        else:
            return None

    def set_rl_training(self, staus):
        self.model.set_rl_training(staus)

    def device(self):
        device = next(self.model.parameters()).device
        return device

    def to(self, DEVICE):
        self.model.to(DEVICE)

    def unroll(self, one_traj):
        action_output = []
        for traj_step in one_traj:
            (feature, label, is_final) = traj_step
            state = Feature.feature2state(feature)
            action_logits_prdict, self.hidden_state, select_units_num = self.action_logits_by_state(state, self.hidden_state)
            action_output.append(action_logits_prdict)
            if is_final:
                self.hidden_state = self.init_hidden_state()

        return action_output

    @staticmethod
    def get_state_and_action_from_pickle_numpy(obs, last_list=None, build_order=None):
        batch_entities_tensor, entity_pos = Agent.preprocess_state_entity_numpy(obs, return_entity_pos=True)
        map_data = Agent.preprocess_state_spatial_numpy(obs, entity_pos_list=entity_pos)
        scalar_list = Agent.preprocess_state_scalar_numpy(obs, build_order=build_order, last_list=last_list)

        batch_entities_tensor = torch.tensor(batch_entities_tensor)
        scalar_list = [torch.tensor(s) for s in scalar_list]
        map_data = torch.tensor(map_data)

        state = MsState(entity_state=batch_entities_tensor, 
                        statistical_state=scalar_list, map_state=map_data)

        del batch_entities_tensor, scalar_list, map_data

        return state

    def preprocess_baseline_state(self, home_obs, away_obs, build_order=None):
        batch_size = 1

        agent_scalar_list = self.get_scalar_list(home_obs, build_order)    
        opponenet_scalar_out = self.get_scalar_list(away_obs)  

        return agent_scalar_list, opponenet_scalar_out

    @staticmethod
    def preprocess_state_entity_numpy(obs, return_entity_pos=False):
        t = time()

        raw_units = obs["raw_units"]
        entities_array, entity_pos = ArchModel.preprocess_entity_numpy(raw_units, return_entity_pos=return_entity_pos)
        batch_entities_array = np.expand_dims(entities_array, axis=0) 

        print('preprocess_state_entity_numpy, t1', time() - t) if speed else None
        t = time()

        del raw_units, entities_array

        if return_entity_pos:
            return batch_entities_array, entity_pos

        return batch_entities_array

    @staticmethod
    def preprocess_state_scalar_numpy(obs, build_order=None, last_list=None):
        return ArchModel.preprocess_scalar_numpy(obs, build_order=build_order, last_list=last_list)

    @staticmethod
    def preprocess_state_spatial_numpy(obs, entity_pos_list=None):
        return ArchModel.preprocess_spatial_numpy(obs, entity_pos_list=entity_pos_list)

    @staticmethod
    def get_baseline_state_from_multi_source_state(obs, msstate):
        scalar_list = msstate.statistical_state

        # statistical_state feature
        # scalar_list.append(agent_statistics) 0
        # scalar_list.append(home_race) 1
        # scalar_list.append(away_race) 2
        # scalar_list.append(upgrades) 3
        # scalar_list.append(enemy_upgrades) 4
        # scalar_list.append(time) 5
        # scalar_list.append(available_actions) 6
        # scalar_list.append(unit_counts_bow) 7
        # scalar_list.append(mmr) 8
        # scalar_list.append(units_buildings) 9
        # scalar_list.append(effects) 10
        # scalar_list.append(upgrade) 11
        # scalar_list.append(beginning_build_order) 12 
        # scalar_list.append(last_delay) 13
        # scalar_list.append(last_action_type) 14 
        # scalar_list.append(last_repeat_queued) 15

        # baseline feature
        # scalar_list.append(agent_statistics) 0
        # scalar_list.append(upgrades) 3
        # scalar_list.append(unit_counts_bow) 7
        # scalar_list.append(units_buildings) 9
        # scalar_list.append(effects) 10
        # scalar_list.append(upgrade) 11 
        # scalar_list.append(beginning_build_order) 12
        baseline_index = [0, 3, 7, 9, 10, 11, 12]

        # can not use index list in python list
        # return scalar_list[baseline_index]
        baseline_state = [e for i, e in enumerate(scalar_list) if i in baseline_index]

        # add cumulative_score in mAS 1.06
        # AlphaStar:cumulative_score - Various score metrics tracked by the game, and only used for baselines, 
        # not for inference during play. These are not visible to humans while playing, and include score, 
        # idle production and work time, total value of units and structure, total destroyed value of units and structures, 
        # total collected minerals and vespene, rate of minerals and vespene collection, and total spent minerals and vespene
        cumulative_score = torch.tensor(np.array(obs['score_cumulative']), dtype=torch.float32).reshape(1, -1)
        print('cumulative_score', cumulative_score) if debug else None
        print('cumulative_score.shape', cumulative_score.shape) if debug else None

        baseline_state.append(cumulative_score)
        del scalar_list, cumulative_score

        return baseline_state

    @staticmethod
    def preprocess_state_all(obs, build_order=None, last_list=None):

        t = time()

        batch_entities_tensor, entity_pos = Agent.preprocess_state_entity_numpy(obs, return_entity_pos=True)

        print('preprocess_state_all, t1', time() - t) if speed else None
        t = time()

        map_data = Agent.preprocess_state_spatial_numpy(obs, entity_pos_list=entity_pos)

        print('preprocess_state_all, t2', time() - t) if speed else None
        t = time()

        scalar_list = Agent.preprocess_state_scalar_numpy(obs, build_order=build_order, last_list=last_list)

        print('preprocess_state_all, t3', time() - t) if speed else None
        t = time()

        batch_entities_tensor = torch.tensor(batch_entities_tensor)
        scalar_list = [torch.tensor(s) for s in scalar_list]
        map_data = torch.tensor(map_data)

        print('preprocess_state_all, t4', time() - t) if speed else None
        t = time()

        state = MsState(entity_state=batch_entities_tensor, 
                        statistical_state=scalar_list, map_state=map_data)

        print('preprocess_state_all, t5', time() - t) if speed else None
        t = time()

        del batch_entities_tensor, scalar_list, map_data

        return state

    def action_logits_by_state(self, state, hidden_state = None, single_inference = False, obs = None):
        batch_size = 1 if single_inference else None
        sequence_length = 1 if single_inference else None

        if single_inference:
            obs_list = [obs]
        else:
            obs_list = obs

        action_logits, actions, new_state, select_units_num, entity_nums = self.model.forward(state, batch_size = batch_size,
                                                                                              sequence_length = sequence_length,
                                                                                              hidden_state = hidden_state, 
                                                                                              return_logits = True,
                                                                                              obs_list = obs_list)
        return action_logits, actions, new_state, select_units_num, entity_nums

    def action_logits_based_on_actions(self, state, action_gt, gt_select_units_num, hidden_state=None, 
                                       single_inference=False, batch_size=None, sequence_length=None):
        batch_size = 1 if single_inference else batch_size
        sequence_length = 1 if single_inference else sequence_length

        baselinelist, action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
            delay_logits, queue_logits, \
            units_logits, target_unit_logits, \
            target_location_logits, select_units_num, new_state, unit_types_one = self.model.mimic_forward(state, 
                                                                                                           action_gt, 
                                                                                                           gt_select_units_num,
                                                                                                           gt_is_one_hot=False,
                                                                                                           batch_size=batch_size, 
                                                                                                           sequence_length=sequence_length, 
                                                                                                           hidden_state=hidden_state,
                                                                                                           multi_gpu_supvised_learning=False)

        # the mimic_forward will added one entity into the max selected size, so we shoulde substract one
        # units_logits = units_logits[:, :-1, :]

        action_logits = ArgsActionLogits(action_type=action_type_logits, delay=delay_logits, queue=queue_logits,
                                         units=units_logits, target_unit=target_unit_logits, 
                                         target_location=target_location_logits)

        del state, action_gt, gt_select_units_num, hidden_state
        del action_type_logits, delay_logits, queue_logits, units_logits, target_unit_logits, target_location_logits
        del baselinelist, action_pred, entity_nums, units, target_unit, target_location, unit_types_one

        return action_logits, select_units_num, new_state

    def action_logits_on_actions_for_unroll(self, state, action_gt, gt_select_units_num, hidden_state=None, 
                                            batch_size=None, sequence_length=None, baseline_state=None, baseline_opponent_state=None,
                                            show=False):

        baselinelist, action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
            delay_logits, queue_logits, \
            units_logits, target_unit_logits, \
            target_location_logits, select_units_num, new_state, unit_types_one = self.model.mimic_forward(state, 
                                                                                                           action_gt, 
                                                                                                           gt_select_units_num,
                                                                                                           gt_is_one_hot=False,
                                                                                                           batch_size=batch_size, 
                                                                                                           sequence_length=sequence_length, 
                                                                                                           hidden_state=hidden_state,
                                                                                                           multi_gpu_supvised_learning=False,
                                                                                                           baseline_state=baseline_state, 
                                                                                                           baseline_opponent_state=baseline_opponent_state,
                                                                                                           show=show)

        # the mimic_forward will added one entity into the max selected size, so we shoulde substract one
        # units_logits = units_logits[:, :-1, :]

        action_logits = ArgsActionLogits(action_type=action_type_logits, delay=delay_logits, queue=queue_logits,
                                         units=units_logits, target_unit=target_unit_logits, 
                                         target_location=target_location_logits)

        del action_type_logits, delay_logits, queue_logits, units_logits, target_unit_logits, target_location_logits
        del action_pred, entity_nums, units, target_unit, target_location, unit_types_one

        return baselinelist, action_logits, select_units_num, new_state

    def state_by_obs(self, obs, return_tag_list = False):
        state, tag_list = Agent.preprocess_state_all(obs, return_tag_list)

        if tag_list and return_tag_list:
            return state, tag_list

        return state, None

    @staticmethod
    def func_call_to_action(func_call, obs = None):
        # note: this is a pysc2 action, and the 
        # unit_tags and target_unit_tag are actually index in _raw_tags!
        # so they are different from the tags really used by a SC2-action!
        func = func_call.function
        args = func_call.arguments

        print('function:', func) if debug else None
        print('function value:', func.value) if debug else None
        print('arguments:', args) if debug else None

        args_action = ArgsAction()
        args_action.action_type = func.value

        # use a non-smart method to calculate the args of the action
        need_args = A.RAW_FUNCTIONS[func].args
        i = 0
        for arg in need_args:
            print("arg:", arg) if debug else None
            if arg.name == 'queued':
                args_action.queue = args[i][0].value
                i = i + 1
            elif arg.name == 'unit_tags':
                args_action.units = args[i]
                i = i + 1
            elif arg.name == 'target_unit_tag':
                args_action.target_unit = args[i][0]
                i = i + 1
            elif arg.name == 'world':
                scale_factor = 0.5 if SCHP.world_size == 128 else 1
                print('args[i]:', args[i]) if debug else None
                args_action.target_location = [int(x * scale_factor) for x in args[i]]
                print('args_action.target_location:', args_action.target_location) if debug else None
                i = i + 1

        if obs is not None:
            units = args_action.units
            print('units index:', units) if debug else None
            if units is not None:
                unit_type = get_unit_type(obs["raw_units"][units[0]].unit_type)
                print('selected unit is:', unit_type) if debug else None

        print('args_action:', args_action) if debug else None
        return args_action

    @staticmethod
    def action_to_func_call(action, select_units_num, action_spec, use_random_args = False):
        # assert the action is single
        print('action:', action) if debug else None
        print('action.get_shape():', action.get_shape()) if debug else None

        function_id = action.action_type.item()
        print('function_id:', function_id) if debug else None

        delay = action.delay.item()
        print('delay:', delay) if debug else None

        queue = action.queue.item()
        print('queue:', queue) if debug else None

        # we assume single inference
        units = action.units.cpu().detach().reshape(-1).numpy().tolist()
        print('units:', units) if debug else None

        print('select_units_num:', select_units_num) if debug else None 
        units_num = select_units_num.item()
        units = units[:units_num]
        print('units:', units) if debug else None

        target_unit = action.target_unit.item()  
        print('target_unit:', target_unit) if debug else None

        # we assume single inference
        target_location = action.target_location.cpu().detach().reshape(-1).numpy().tolist()
        print('target_location:', target_location) if debug else None

        del action, select_units_num

        need_args = action_spec.functions[function_id].args
        args = []

        def to_list(i):
            return [i]

        if not use_random_args:
            for arg in need_args:
                print("arg:", arg) if debug else None
                rand = [np.random.randint(0, size) for size in arg.sizes]

                if arg.name == 'queued':
                    size = arg.sizes[0]
                    if queue < 0 or queue > size - 1:
                        args.append(rand)
                        print("argument queue beyond the size!") if debug else None
                    else:
                        args.append(to_list(queue))
                    del size
                elif arg.name == 'unit_tags':
                    # the unit_tags size is actually the max selected number
                    size = arg.sizes[0]
                    units_args = []
                    for unit_index in units:
                        print('unit_index', unit_index) if debug else None
                        print('size', size) if debug else None
                        if unit_index < 0 or unit_index > size - 1:
                            units_args.append(np.random.randint(0, size))
                            print("argument unit_index beyond the size!") if debug else None
                        else:
                            units_args.append(unit_index)
                    args.append(units_args)
                    del units_args, size
                elif arg.name == 'target_unit_tag':
                    size = arg.sizes[0]
                    if target_unit < 0 or target_unit > size - 1:
                        args.append(rand)
                        print("argument target_unit beyond the size!") if debug else None
                    else:
                        args.append(to_list(target_unit))
                    del size
                elif arg.name == 'world':
                    world_args = []
                    for val, size in zip(target_location, arg.sizes):
                        if val < 0 or val > size - 1:
                            print('val', val) if debug else None
                            print('size', size) if debug else None
                            world_args.append(np.random.randint(0, size))
                            print("argument world beyond the size!") if debug else None
                        else:
                            world_args.append(val)                        
                    args.append(world_args)
                    del world_args

                del rand
        else:
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in action_spec.functions[function_id].args]

        print('args:', args) if debug else None

        # AlphaStar use the raw actions
        func_call = A.FunctionCall.init_with_validation(function=function_id, arguments=args, raw=True)

        del delay, queue, function_id, args, units_num, units, target_location, target_unit

        return func_call

    def unroll_traj(self, state_all, initial_state, baseline_state=None, baseline_opponent_state=None):
        baseline_value_list, action_logits, actions, _, select_units_num, entity_num = self.model.forward(state_all, batch_size=None, sequence_length=None, 
                                                                                                          hidden_state=initial_state, return_logits=True,
                                                                                                          baseline_state=baseline_state, 
                                                                                                          baseline_opponent_state=baseline_opponent_state,
                                                                                                          return_baseline=True)
        return baseline_value_list, action_logits, actions, select_units_num, entity_num

    def get_weights(self):
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
        return


def test():

    agent = Agent()

    batch_size = AHP.batch_size * AHP.sequence_length
    # dummy scalar list
    scalar_list = []

    agent_statistics = torch.ones(batch_size, SFS.agent_statistics)
    home_race = torch.randn(batch_size, SFS.home_race)
    away_race = torch.randn(batch_size, SFS.away_race)
    upgrades = torch.randn(batch_size, SFS.upgrades)
    enemy_upgrades = torch.randn(batch_size, SFS.upgrades)
    time = torch.randn(batch_size, SFS.time)

    available_actions = torch.randn(batch_size, SFS.available_actions)
    unit_counts_bow = torch.ones(batch_size, SFS.unit_counts_bow)
    mmr = torch.randn(batch_size, SFS.mmr)
    units_buildings = torch.ones(batch_size, SFS.units_buildings)
    effects = torch.randn(batch_size, SFS.effects)
    upgrade = torch.randn(batch_size, SFS.upgrade)

    beginning_build_order = torch.randn(batch_size, SCHP.count_beginning_build_order, 
                                        int(SFS.beginning_build_order / SCHP.count_beginning_build_order))
    last_delay = torch.randn(batch_size, SFS.last_delay)
    last_action_type = torch.randn(batch_size, SFS.last_action_type)
    last_repeat_queued = torch.randn(batch_size, SFS.last_repeat_queued)

    scalar_list.append(agent_statistics)
    scalar_list.append(home_race)
    scalar_list.append(away_race)
    scalar_list.append(upgrades)
    scalar_list.append(enemy_upgrades)
    scalar_list.append(time)

    scalar_list.append(available_actions)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(mmr)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)

    scalar_list.append(beginning_build_order)
    scalar_list.append(last_delay)
    scalar_list.append(last_action_type)
    scalar_list.append(last_repeat_queued)

    # dummy entity list
    e_list = []
    e1 = Entity(115, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 0, 100, 60, 50, 4, 8, 95, 0.2, 0.0, 0.0, 140, 60, 100,
                1, 123, 218, 3, True, False, True, True, False, 0, 0, 0, 0, 0, 0, 3.0, [2, 3], 2, 1, 0, True, False)
    e2 = Entity(1908, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 2, 1500, 0, 200, 0, 4, 15, 0.5, 0.8, 0.5, 1500, 0, 250,
                2, 69, 7, 3, True, False, False, True, False, 0, 0, 0, 0, 10, 16, 0.0, [1], 1, 1, 0, False, False)
    e_list.append(e1)
    e_list.append(e2)

    entities_tensor = torch.tensor(ArchModel.preprocess_entity_numpy(e_list))
    print('entities_tensor.shape:', entities_tensor.shape) if debug else None
    batch_entities_tensor = torch.unsqueeze(entities_tensor, dim=0)
    batch_entities_list = []
    for i in range(batch_size):
        batch_entities_tensor_copy = batch_entities_tensor.detach().clone()
        batch_entities_list.append(batch_entities_tensor_copy)

    batch_entities_tensor = torch.cat(batch_entities_list, dim=0)
    print('batch_entities_tensor.shape:', batch_entities_tensor.shape) if debug else None

    # dummy map list
    map_list = []
    map_data_1 = torch.zeros(batch_size, 6, AHP.minimap_size, AHP.minimap_size)

    map_list.append(map_data_1)
    map_data_2 = torch.zeros(batch_size, 18, AHP.minimap_size, AHP.minimap_size)
    map_list.append(map_data_2)
    map_data = torch.cat(map_list, dim=1)

    state = MsState(entity_state=batch_entities_tensor, statistical_state=scalar_list, map_state=map_data)

    print("Multi-source state:", state) if debug else None

    # action = agent.action_by_state(state)
    # print("action is:", action) if debug else None

    action_logits, actions, hidden_state, select_units_num, _ = agent.action_logits_by_state(state)
    print("action_logits is:", action_logits) if debug else None
    print("actions is:", actions) if debug else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()
