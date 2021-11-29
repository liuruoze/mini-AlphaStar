#!/usr/bin/env python
# -*- coding: utf-8 -*-

" ArchModel."

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from alphastarmini.core.arch.scalar_encoder import ScalarEncoder
from alphastarmini.core.arch.entity_encoder import EntityEncoder, Entity
from alphastarmini.core.arch.spatial_encoder import SpatialEncoder
from alphastarmini.core.arch.core import Core
from alphastarmini.core.arch.action_type_head import ActionTypeHead
from alphastarmini.core.arch.delay_head import DelayHead
from alphastarmini.core.arch.queue_head import QueueHead
from alphastarmini.core.arch.selected_units_head import SelectedUnitsHead
from alphastarmini.core.arch.target_unit_head import TargetUnitHead
from alphastarmini.core.arch.location_head import LocationHead
from alphastarmini.core.arch.baseline import Baseline

from alphastarmini.core.rl.action import ArgsAction, ArgsActionLogits
from alphastarmini.core.rl.state import MsState

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


class ArchModel(nn.Module):
    '''
    Inputs: state
    Outputs:
        action
    '''

    def __init__(self):
        super(ArchModel, self).__init__()
        self.scalar_encoder = ScalarEncoder()
        self.entity_encoder = EntityEncoder()
        self.spatial_encoder = SpatialEncoder()
        self.core = Core()
        self.action_type_head = ActionTypeHead()
        self.delay_head = DelayHead()
        self.queue_head = QueueHead()
        self.selected_units_head = SelectedUnitsHead()
        self.target_unit_head = TargetUnitHead()
        self.location_head = LocationHead()

        # build all baselines
        self.build_baselines()

        # init all parameters
        if AHP.init_net_params:
            self.init_paramters()

    def init_paramters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_baselines(self):
        self.winloss_baseline = Baseline(baseline_type='winloss')
        self.build_order_baseline = Baseline(baseline_type='build_order')
        self.built_units_baseline = Baseline(baseline_type='built_units')
        self.upgrades_baseline = Baseline(baseline_type='upgrades')
        self.effects_baseline = Baseline(baseline_type='effects')

    def set_rl_training(self, staus):
        self.action_type_head.set_rl_training(staus)

    def count_parameters(self):  
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def preprocess_entity_numpy(e_list, return_entity_pos=False):
        return EntityEncoder.preprocess_numpy(e_list, return_entity_pos=return_entity_pos)

    @staticmethod    
    def preprocess_scalar_numpy(obs, build_order=None):
        return ScalarEncoder.preprocess_numpy(obs, build_order=build_order)

    @staticmethod    
    def preprocess_spatial_numpy(obs, entity_pos_list=None):
        return SpatialEncoder.preprocess_numpy(obs, entity_pos_list=entity_pos_list)

    def init_hidden_state(self):
        return self.core.init_hidden_state()

    def forward(self, state, batch_size=None, sequence_length=None, hidden_state=None, return_logits=False, 
                baseline_state=None, baseline_opponent_state=None, 
                return_baseline=False, multi_gpu_supvised_learning=False,
                use_scatter_map=True):
        # shapes of embedded_entity, embedded_spatial, embedded_scalar are all [batch_size x embedded_size]
        entity_embeddings, embedded_entity, entity_num = self.entity_encoder(state.entity_state)   

        if AHP.scatter_channels:
            map_skip, embedded_spatial = self.spatial_encoder(state.map_state, entity_embeddings)
        else:
            map_skip, embedded_spatial = self.spatial_encoder(state.map_state)

        embedded_scalar, scalar_context = self.scalar_encoder(state.statistical_state)

        lstm_output, hidden_state = self.core(embedded_scalar, embedded_entity, embedded_spatial, 
                                              batch_size, sequence_length, hidden_state)
        print('lstm_output.shape:', lstm_output.shape) if debug else None
        print('lstm_output is nan:', torch.isnan(lstm_output).any()) if debug else None

        action_type_logits, action_type, autoregressive_embedding = self.action_type_head(lstm_output, scalar_context)
        delay_logits, delay, autoregressive_embedding = self.delay_head(autoregressive_embedding)
        queue_logits, queue, autoregressive_embedding = self.queue_head(autoregressive_embedding, action_type, embedded_entity)

        units_logits, units, autoregressive_embedding, select_units_num = self.selected_units_head(autoregressive_embedding, 
                                                                                                   action_type, 
                                                                                                   entity_embeddings, 
                                                                                                   entity_num)

        target_unit_logits, target_unit = self.target_unit_head(autoregressive_embedding, 
                                                                action_type, entity_embeddings)
        target_location_logits, target_location = self.location_head(autoregressive_embedding, action_type, map_skip)

        action_logits = ArgsActionLogits(action_type=action_type_logits, delay=delay_logits, queue=queue_logits,
                                         units=units_logits, target_unit=target_unit_logits, 
                                         target_location=target_location_logits)
        action = ArgsAction(action_type=action_type, delay=delay, queue=queue,
                            units=units, target_unit=target_unit, target_location=target_location)

        if multi_gpu_supvised_learning:
            return action_type, units, target_location, action_type_logits, delay_logits, queue_logits, \
                units_logits, target_unit_logits, target_location_logits, select_units_num

        if return_logits:

            if return_baseline:
                winloss_baseline_value = self.winloss_baseline.forward(lstm_output, baseline_state, baseline_opponent_state)
                build_order_baseline_value = self.build_order_baseline.forward(lstm_output, baseline_state, baseline_opponent_state)
                built_units_baseline_value = self.built_units_baseline.forward(lstm_output, baseline_state, baseline_opponent_state)
                upgrades_baseline_value = self.upgrades_baseline.forward(lstm_output, baseline_state, baseline_opponent_state)
                effects_baseline_value = self.effects_baseline.forward(lstm_output, baseline_state, baseline_opponent_state)

                baseline_value_list = [winloss_baseline_value, build_order_baseline_value, built_units_baseline_value,
                                       upgrades_baseline_value, effects_baseline_value]

                return baseline_value_list, action_logits, action, hidden_state, select_units_num

            return action_logits, action, hidden_state, select_units_num

        return action, hidden_state, select_units_num


def test():
    # init model
    arch_model = ArchModel()
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
    unit_counts_bow = torch.randn(batch_size, SFS.unit_counts_bow)
    mmr = torch.randn(batch_size, SFS.mmr)
    units_buildings = torch.randn(batch_size, SFS.units_buildings)
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

    # preprocess entity list
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
    map_data_1 = torch.zeros(batch_size, 18, AHP.minimap_size, AHP.minimap_size)

    map_list.append(map_data_1)
    map_data_2 = torch.zeros(batch_size, 6, AHP.minimap_size, AHP.minimap_size)
    map_list.append(map_data_2)
    map_data = torch.cat(map_list, dim=1)

    state = MsState(entity_state=batch_entities_tensor, statistical_state=scalar_list, map_state=map_data)
    print("Multi-source state:", state) if debug else None

    # dummy scalar list
    scalar_list = []

    agent_statistics = torch.ones(batch_size, SFS.agent_statistics)
    upgrades = torch.randn(batch_size, SFS.upgrades)
    unit_counts_bow = torch.randn(batch_size, SFS.unit_counts_bow)
    units_buildings = torch.randn(batch_size, SFS.units_buildings)
    effects = torch.randn(batch_size, SFS.effects)
    upgrade = torch.randn(batch_size, SFS.upgrade)
    beginning_build_order = torch.randn(batch_size, SCHP.count_beginning_build_order, 
                                        int(SFS.beginning_build_order / SCHP.count_beginning_build_order))

    scalar_list.append(agent_statistics)
    scalar_list.append(upgrades)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)
    scalar_list.append(beginning_build_order)

    opponenet_scalar_out = scalar_list

    action_logits, action, _, select_units_num = arch_model.forward(state, 
                                                                    batch_size=AHP.batch_size, 
                                                                    sequence_length=AHP.sequence_length, 
                                                                    return_logits=True)

    if action.action_type is not None:
        print("action:", action.action_type) if debug else None
    else:
        print("action is None!")

    # test loss and backward
    print("Test backward!")
    print("action_logits.action_type:", action_logits.action_type) if debug else None

    # if MiniStar_Arch_Hyper_Parameters is used, and Mini_Scale = 16,
    # then batch_size = 96 / 16 = 6, sequence_length = 64 / 16 = 4, batch_seq_size = 6 * 4 = 24.
    # Thus shape = [24, number_of_action_types=564]
    print("action_logits.action_type.shape:", action_logits.action_type.shape) if debug else None

    optimizer = Adam(arch_model.parameters(), lr=1e-4)
    with torch.autograd.set_detect_anomaly(True):
        hidden_state = None

        for _ in range(3):

            # important, if not set, below error will raise:
            # Trying to backward through the graph a second time, but the buffers have already 
            # been freed. Specify retain_graph=True when calling backward the first time
            if hidden_state is not None:
                (h, c) = hidden_state
                hidden_state = (h.detach(), c.detach())

            baseline_value, action_logits, \
                action, new_hidden_state, select_units_num = arch_model.forward(state, hidden_state=hidden_state, 
                                                                                return_logits=True, baseline_state=scalar_list, 
                                                                                baseline_opponent_state=opponenet_scalar_out, 
                                                                                return_baseline=True)
            optimizer.zero_grad()
            print("action_logits.action_type:", action_logits.action_type) if debug else None
            print("action_logits.action_type.shape:", action_logits.action_type.shape) if debug else None

            print("baseline_value:", baseline_value) if debug else None
            print("new_hidden_state.shape:", new_hidden_state[0].shape) if debug else None

            loss_base = sum([base.sum() for base in baseline_value])

            loss = action_logits.action_type.sum() + loss_base
            print("loss:", loss) if debug else None

            loss.backward()
            optimizer.step()

            hidden_state = new_hidden_state

    print("End test backward!")

    if action.target_unit is not None:
        print("target_unit:", action.target_unit) if debug else None
    else:
        print("target_unit is None!")

    if action.target_location is not None:
        print("target_location:", action.target_location) if debug else None
    else:
        print("target_location is None!")

    print("action is:", action) if debug else None
    print("action shape is:", action.get_shape()) if debug else None
    print("This is a test!") if debug else None

    print("ArchModel parameters:", arch_model.count_parameters()) if debug else None

    print("action.toLogits():", action.toLogits())


if __name__ == '__main__':
    test()
