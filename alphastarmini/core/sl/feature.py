#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Feature object, transfer the state to one-dimension feature"

import numpy as np

import torch

from alphastarmini.core.rl.state import MsState

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS
from alphastarmini.lib.hyper_parameters import ScalarFeature

__author__ = "Ruo-Ze Liu"

debug = False


class Feature(object):
    '''
    Inputs: state
    Outputs:
        Feature
    '''

    def __init__(self):
        super(Feature, self).__init__()
        pass

    @staticmethod
    def state2feature(state):
        ''' 
        input: MsState 
        outoput: [batch_size x feature_embedding_size]
        '''

        '''not used:
        map_data = state[2]
        batch_entities_tensor = state[1]
        scalar_list = state[0]
        '''

        map_data = state.map_state
        batch_entities_tensor = state.entity_state
        scalar_list = state.statistical_state

        batch_size = map_data.shape[0]  
        bbo_index = ScalarFeature.beginning_build_order
        scalar_list[bbo_index] = scalar_list[bbo_index].reshape(batch_size, SFS[bbo_index])
        for z in scalar_list:
            print("z.shape:", z.shape) if debug else None

        feature_1 = torch.cat(scalar_list, dim=1)
        print("feature_1.shape:", feature_1.shape) if debug else None

        print('batch_entities_tensor.shape', batch_entities_tensor.shape) if debug else None
        print('batch_size', batch_size) if debug else None
        print('AHP.max_entities', AHP.max_entities) if debug else None
        print('AHP.embedding_size', AHP.embedding_size) if debug else None

        feature_2 = batch_entities_tensor.reshape(batch_size, AHP.max_entities * AHP.embedding_size)
        print("feature_2.shape:", feature_2.shape) if debug else None

        print('map_data.shape', map_data.shape) if debug else None
        print('AHP.map_channels', AHP.map_channels) if debug else None
        print('AHP.minimap_size', AHP.minimap_size) if debug else None

        feature_3 = map_data.reshape(batch_size, AHP.map_channels * AHP.minimap_size * AHP.minimap_size)
        print("feature_3.shape:", feature_3.shape) if debug else None

        feature = torch.cat([feature_1, feature_2, feature_3], dim=1) 
        return feature

    @staticmethod
    def state2feature_numpy(state):
        ''' 
        input: MsState 
        outoput: [batch_size x feature_embedding_size]
        '''

        '''not used:
        map_data = state[2]
        batch_entities_tensor = state[1]
        scalar_list = state[0]
        '''

        map_data = state.map_state
        batch_entities_tensor = state.entity_state
        scalar_list = state.statistical_state

        batch_size = map_data.shape[0]  
        bbo_index = ScalarFeature.beginning_build_order
        scalar_list[bbo_index] = scalar_list[bbo_index].reshape(batch_size, SFS[bbo_index])
        for z in scalar_list:
            print("z.shape:", z.shape) if debug else None

        feature_1 = np.concatenate(scalar_list, axis=1)
        print("feature_1.shape:", feature_1.shape) if debug else None

        print('batch_entities_tensor.shape', batch_entities_tensor.shape) if debug else None
        print('batch_size', batch_size) if debug else None
        print('AHP.max_entities', AHP.max_entities) if debug else None
        print('AHP.embedding_size', AHP.embedding_size) if debug else None

        feature_2 = batch_entities_tensor.reshape(batch_size, AHP.max_entities * AHP.embedding_size)
        print("feature_2.shape:", feature_2.shape) if debug else None

        print('map_data.shape', map_data.shape) if debug else None
        print('AHP.map_channels', AHP.map_channels) if debug else None
        print('AHP.minimap_size', AHP.minimap_size) if debug else None

        feature_3 = map_data.reshape(batch_size, AHP.map_channels * AHP.minimap_size * AHP.minimap_size)
        print("feature_3.shape:", feature_3.shape) if debug else None

        feature = np.concatenate([feature_1, feature_2, feature_3], axis=1) 
        return feature

    @staticmethod    
    def getSize():

        # note: do not use AHP.scalar_feature_size
        #feature_1_size = AHP.scalar_feature_size
        size_all = 0
        for i in ScalarFeature:        
            size_all += SFS[i]
        feature_1_size = size_all

        feature_2_size = AHP.max_entities * AHP.embedding_size
        feature_3_size = AHP.map_channels * AHP.minimap_size * AHP.minimap_size 
        return feature_1_size + feature_2_size + feature_3_size

    @staticmethod    
    def feature2state(feature):
        ''' 
        input: [batch_size x feature_embedding_size]
        outoput: MsState
        '''
        batch_size = feature.shape[0]
        print('feature.shape', feature.shape) if debug else None

        # note: do not use AHP.scalar_feature_size
        #feature_1_size = AHP.scalar_feature_size

        size_all = 0
        for i in ScalarFeature:          
            size_all += SFS[i]
        feature_1_size = size_all

        feature_2_size = AHP.max_entities * AHP.embedding_size
        feature_3_size = AHP.map_channels * AHP.minimap_size * AHP.minimap_size

        print("feature_1_size + feature_2_size + feature_3_size:", 
              feature_1_size + feature_2_size + feature_3_size) if debug else None
        print('feature.shape[1]:', feature.shape[1]) if debug else None
        assert feature_1_size + feature_2_size + feature_3_size == feature.shape[1]

        feature_1 = feature[:, :feature_1_size]      
        scalar_list = []
        last_index = 0

        for i in ScalarFeature:          
            scalar_feature = feature_1[:, last_index:last_index + SFS[i]]
            print('added scalar_feature.shape:', scalar_feature.shape) if debug else None
            scalar_list.append(scalar_feature)
            last_index += SFS[i]

        bbo_index = ScalarFeature.beginning_build_order

        print('batch_size:', batch_size) if debug else None
        print('scalar_list[bbo_index].shape:', scalar_list[bbo_index].shape) if debug else None
        scalar_list[bbo_index] = scalar_list[bbo_index].reshape(batch_size, 
                                                                SCHP.count_beginning_build_order, 
                                                                int(SFS[bbo_index] / SCHP.count_beginning_build_order))

        feature_2 = feature[:, feature_1_size:feature_1_size + feature_2_size]
        batch_entities_tensor = feature_2.reshape(batch_size, AHP.max_entities, AHP.embedding_size)

        print("feature[:, -feature_3_size:].shape:", feature[:, -feature_3_size:].shape) if debug else None
        print("feature[:, feature_1_size + feature_2_size:].shape:", 
              feature[:, feature_1_size + feature_2_size:].shape) if debug else None
        # assert feature[:, -feature_3_size:] == feature[:, feature_1_size + feature_2_size:]
        #
        feature_3 = feature[:, -feature_3_size:]
        map_data = feature_3.reshape(batch_size, AHP.map_channels, AHP.minimap_size, AHP.minimap_size)

        state = MsState(entity_state=batch_entities_tensor, 
                        statistical_state=scalar_list, map_state=map_data)

        # not used:
        # return [scalar_list, batch_entities_tensor, map_data]
        return state
