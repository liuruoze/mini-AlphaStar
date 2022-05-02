#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Baseline Encoder."
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.core.arch.spatial_encoder import ResBlock1D

from alphastarmini.lib.alphastar_transformer import Transformer

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS

__author__ = "Ruo-Ze Liu"

debug = False


class Baseline(nn.Module):
    '''    
    Inputs: prev_state, scalar_features, opponent_observations, cumulative_score, action_type, lstm_output
    Outputs:
        winloss_baseline - A baseline value used for the `action_type` argument
    '''

    def __init__(self, baseline_type='winloss',
                 n_statistics=SFS.agent_statistics,
                 n_cumulatscore=SFS.cumulative_score,
                 baseline_input=AHP.winloss_baseline_input_size,
                 n_upgrades=SFS.upgrades, 
                 n_units_buildings=SFS.unit_counts_bow, 
                 n_effects=SFS.effects, n_upgrade=SFS.upgrade,
                 n_resblocks=AHP.n_resblocks, 
                 original_32=AHP.original_32,
                 original_64=AHP.original_64,
                 original_128=AHP.original_128,
                 original_256=AHP.original_256):
        super().__init__()
        self.baseline_type = baseline_type
        if baseline_type == 'build_order':
            baseline_input = AHP.build_order_baseline_input_size
        elif baseline_type == 'built_units':
            baseline_input = AHP.built_units_baseline_input_size
        elif baseline_type == 'upgrades':
            baseline_input = AHP.upgrades_baseline_input_size
        elif baseline_type == 'effects':
            baseline_input = AHP.effects_baseline_input_size

        self.statistics_fc = nn.Linear(n_statistics, original_64)  # with relu
        self.cumulatscore_fc = nn.Linear(n_cumulatscore, original_64)  # with relu
        self.upgrades_fc = nn.Linear(n_upgrades, original_128)  # with relu
        self.unit_counts_bow_fc = nn.Linear(n_units_buildings, original_64)  # A bag-of-words unit count from `entity_list`, with relu
        self.units_buildings_fc = nn.Linear(n_units_buildings, original_32)  # with relu, also goto scalar_context
        self.effects_fc = nn.Linear(n_effects, original_32)  # with relu, also goto scalar_context
        self.upgrade_fc = nn.Linear(n_upgrade, original_32)  # with relu, also goto scalar_context. What is the difference with upgrades_fc?

        self.build_order_model_size = 16
        self.before_beginning_build_order = nn.Linear(n_units_buildings + SCHP.count_beginning_build_order, 
                                                      self.build_order_model_size)  # without relu  
        self.beginning_build_order_transformer = Transformer(d_model=self.build_order_model_size, 
                                                             d_inner=self.build_order_model_size * 2,
                                                             n_layers=3, n_head=2, 
                                                             d_k=8, d_v=8, 
                                                             dropout=0.)
        self.relu = nn.ReLU()

        self.embed_fc = nn.Linear(baseline_input, original_256)  # with relu
        self.resblock_stack = nn.ModuleList([
            ResBlock1D(inplanes=original_256, planes=original_256, seq_len=1)
            for _ in range(n_resblocks)])

        self.out_fc = nn.Linear(original_256, 1) 

    def pre_forward(self, various_observations):
        [agent_statistics, upgrades, unit_counts_bow,
         units_buildings, effects, upgrade, beginning_build_order, cumulative_score] = various_observations

        # These features are all concatenated together to yield `action_type_input`, 
        # passed through a linear of size 256, then passed through 16 ResBlocks with 256 hidden units 
        # and layer normalization, passed through a ReLU, then passed through 
        # a linear with 1 hidden unit.

        device = next(self.parameters()).device
        embedded_scalar_list = []

        # agent_statistics: Embedded by taking log(agent_statistics + 1) and passing through a linear of size 64 and a ReLU
        the_log_statistics = torch.log(agent_statistics + 1)
        x = F.relu(self.statistics_fc(the_log_statistics))
        del agent_statistics, the_log_statistics
        embedded_scalar_list.append(x)

        score_log_statistics = torch.log(cumulative_score + 1)
        x = F.relu(self.cumulatscore_fc(score_log_statistics))
        del cumulative_score, score_log_statistics
        embedded_scalar_list.append(x)

        # upgrades: The boolean vector of whether an upgrade is present is embedded through a linear of size 128 and a ReLU
        x = F.relu(self.upgrades_fc(upgrades))
        del upgrades
        embedded_scalar_list.append(x)

        # unit_counts_bow: A bag-of-words unit count from `entity_list`. 
        # The unit count vector is embedded by square rooting, passing through a linear layer, and passing through a ReLU
        x = F.relu(self.unit_counts_bow_fc(unit_counts_bow))
        del unit_counts_bow
        embedded_scalar_list.append(x)

        # cumulative_statistics: The cumulative statistics (including units, buildings, effects, and upgrades) are preprocessed 
        # into a boolean vector of whether or not statistic is present in a human game. 
        # That vector is split into 3 sub-vectors of units/buildings, effects, and upgrades, 
        # and each subvector is passed through a linear of size 32 and a ReLU, and concatenated together.
        # The embedding is also added to `scalar_context`

        cumulative_statistics = []  # it is different in different baseline
        if self.baseline_type == "winloss" or self.baseline_type == "build_order" or self.baseline_type == "built_units":
            x = F.relu(self.units_buildings_fc(units_buildings))
            cumulative_statistics.append(x)
        if self.baseline_type == "effects" or self.baseline_type == "build_order":
            x = F.relu(self.effects_fc(effects))
            cumulative_statistics.append(x)
        if self.baseline_type == "upgrades" or self.baseline_type == "build_order":    
            x = F.relu(self.upgrade_fc(upgrade))
            cumulative_statistics.append(x)

        embedded_scalar_list.extend(cumulative_statistics)
        del units_buildings, effects, upgrade, cumulative_statistics

        # beginning_build_order: The first 20 constructed entities are converted to a 2D tensor of size 
        # [20, num_entity_types], concatenated with indices and the binary encodings 
        # (as in the Entity Encoder) of where entities were constructed (if applicable). 
        # The concatenation is passed through a transformer similar to the one in the entity encoder, 
        # but with keys, queries, and values of 8 and with a MLP hidden size of 32. 
        # The embedding is also added to `scalar_context`.
        print("beginning_build_order:", beginning_build_order) if debug else None
        print("beginning_build_order.shape:", beginning_build_order.shape) if debug else None

        batch_size = beginning_build_order.shape[0]

        seq = torch.arange(SCHP.count_beginning_build_order)
        seq = L.tensor_one_hot(seq, SCHP.count_beginning_build_order)
        seq = seq.unsqueeze(0).repeat(batch_size, 1, 1).to(beginning_build_order.device)

        bo_sum = beginning_build_order.sum(dim=-1, keepdim=False)
        bo_sum = bo_sum.sum(dim=-1, keepdim=False)
        bo_sum = bo_sum.unsqueeze(1)
        bo_sum = bo_sum.repeat(1, SCHP.count_beginning_build_order)

        mask = torch.arange(SCHP.count_beginning_build_order)
        mask = mask.unsqueeze(0).repeat(batch_size, 1).to(bo_sum.device)
        mask = mask < bo_sum
        mask = mask.unsqueeze(2).repeat(1, 1, SCHP.count_beginning_build_order)

        # add the seq info, referenced by the processing way of DI-star
        x = torch.cat([beginning_build_order, seq], dim=2)
        x = self.before_beginning_build_order(x)

        # like in entity encoder, we add a sequence mask
        x = self.beginning_build_order_transformer(x, mask=mask)
        x = x.reshape(x.shape[0], SCHP.count_beginning_build_order * self.build_order_model_size)

        embedded_scalar_list.append(x)
        embedded_scalar = torch.cat(embedded_scalar_list, dim=1)

        del x, mask, bo_sum, seq, embedded_scalar_list

        return embedded_scalar

    def forward(self, lstm_output, various_observations, opponent_observations=None):
        # check and improve it
        player_scalar_out = self.pre_forward(various_observations)
        del various_observations

        # AlphaStar: The baseline extracts those same observations from `opponent_observations`.
        if opponent_observations is not None:
            opponenet_scalar_out = self.pre_forward(opponent_observations)
            del opponent_observations
        else:
            opponenet_scalar_out = torch.zeros_like(player_scalar_out)

        # AlphaStar: These features are all concatenated together to yield `action_type_input`
        action_type_input = torch.cat([lstm_output, player_scalar_out, opponenet_scalar_out], dim=1)
        del lstm_output, player_scalar_out, opponenet_scalar_out
        print("action_type_input.shape:", action_type_input.shape) if debug else None

        # AlphaStar: passed through a linear of size 256
        x = self.embed_fc(action_type_input)
        del action_type_input
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: then passed through 16 ResBlocks with 256 hidden units and layer normalization,
        # Note: tje layer normalization is already inside each resblock
        x = x.unsqueeze(-1)
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = x.squeeze(-1)    
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: passed through a ReLU
        x = F.relu(x)

        # AlphaStar: then passed through a linear with 1 hidden unit.
        baseline = self.out_fc(x)
        print("baseline:", baseline) if debug else None

        # AlphaStar: This baseline value is transformed by ((2.0 / PI) * atan((PI / 2.0) * baseline)) and is used as the baseline value
        out = (2.0 / np.pi) * torch.atan((np.pi / 2.0) * baseline)
        print("out:", out) if debug else None

        del x, baseline

        return out


def test():
    base_line = Baseline()

    batch_size = 2
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

    cumulative_score = torch.ones(batch_size, SFS.cumulative_score)
    scalar_list.append(agent_statistics)
    scalar_list.append(upgrades)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)
    scalar_list.append(beginning_build_order)
    scalar_list.append(cumulative_score)

    opponenet_scalar_out = scalar_list

    lstm_output = torch.ones(batch_size, AHP.lstm_hidden_dim)

    out = base_line.forward(lstm_output, scalar_list, opponenet_scalar_out)

    print("out:", out) if debug else None
    print("out.shape:", out.shape) if debug else None

    if debug:
        print("This is a test!")
