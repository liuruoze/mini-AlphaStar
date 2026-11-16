#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Label object, transfer the action to one-dimension label"

import numpy as np

import torch

from alphastarmini.core.rl.action import ArgsAction, ArgsActionLogits

from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import LabelIndex

__author__ = "Ruo-Ze Liu"

debug = False


class Label(object):
    '''
    Inputs: action
    Outputs:
        Label
    '''

    def __init__(self):
        super(Label, self).__init__()
        pass

    @staticmethod
    def actionlist2label(actionlist):
        ''' 
        input: args action list 
        outoput: [batch_size x label_feature_size]
        '''
        # TOCHANGE
        tue_index = LabelIndex.target_location_encoding
        target_location_encoding = action[tue_index]
        batch_size = target_location_encoding.shape[0]

        print("target_location_encoding.shape before:", target_location_encoding.shape) if debug else None
        target_location_encoding = target_location_encoding.reshape(batch_size, LS[tue_index])
        print("target_location_encoding.shape after:", target_location_encoding.shape) if debug else None

        action[tue_index] = target_location_encoding

        label = torch.cat(action, dim=1)
        return label

    @staticmethod    
    def getSize():
        last_index = 0
        for i in LabelIndex:
            last_index += LS[i]
        return last_index

    @staticmethod
    def action2label(action):
        ''' 
        input: args action logits (tensor) 
        outoput: [batch_size x label_feature_size]
        '''
        target_location_index = LabelIndex.target_location_encoding
        target_location_encoding = action.target_location
        batch_size = target_location_encoding.shape[0]
        print("target_location_encoding.shape before:", target_location_encoding.shape) if debug else None
        target_location_encoding = target_location_encoding.reshape(batch_size, LS[target_location_index])
        print("target_location_encoding.shape after:", target_location_encoding.shape) if debug else None
        action.target_location = target_location_encoding

        units_index = LabelIndex.select_units_encoding
        units_encoding = action.units
        units_encoding = units_encoding.reshape(batch_size, LS[units_index])
        action.units = units_encoding

        target_unit_index = LabelIndex.target_unit_encoding
        target_unit_encoding = action.target_unit
        target_unit_encoding = target_unit_encoding.reshape(batch_size, LS[target_unit_index])
        action.target_unit = target_unit_encoding

        label = torch.cat(action.toList(), dim=1)
        return label

    @staticmethod
    def action2label_numpy(action):
        ''' 
        input: args action logits (tensor) 
        outoput: [batch_size x label_feature_size]
        '''
        target_location_index = LabelIndex.target_location_encoding
        target_location_encoding = action.target_location
        batch_size = target_location_encoding.shape[0]

        print("target_location_encoding.shape before:", target_location_encoding.shape) if debug else None
        target_location_encoding = target_location_encoding.reshape(batch_size, LS[target_location_index])
        print("target_location_encoding.shape after:", target_location_encoding.shape) if debug else None
        action.target_location = target_location_encoding

        units_index = LabelIndex.select_units_encoding
        units_encoding = action.units
        units_encoding = units_encoding.reshape(batch_size, LS[units_index])
        action.units = units_encoding

        target_unit_index = LabelIndex.target_unit_encoding
        target_unit_encoding = action.target_unit
        target_unit_encoding = target_unit_encoding.reshape(batch_size, LS[target_unit_index])
        action.target_unit = target_unit_encoding

        label = np.concatenate(action.toList(), axis=1)
        return label

    @staticmethod
    def label2action(label):
        ''' 
        input: [batch_size x label_feature_size]
        outoput: args action logits (tensor)
        '''

        batch_size = label.shape[0]
        action = None

        action_list = []
        last_index = 0
        for i in LabelIndex:
            label_i = label[:, last_index:last_index + LS[i]]
            print('added label_i.shape:', label_i.shape) if debug else None
            action_list.append(label_i)
            last_index += LS[i]

        tue_index = LabelIndex.target_location_encoding
        print("action_list[tue_index].shape before:", action_list[tue_index].shape) if debug else None  
        action_list[tue_index] = action_list[tue_index].reshape(batch_size, SCHP.world_size, 
                                                                int(LS[tue_index] / SCHP.world_size))
        print("action_list[tue_index].shape before:", action_list[tue_index].shape) if debug else None

        units_index = LabelIndex.select_units_encoding
        action_list[units_index] = action_list[units_index].reshape(batch_size, AHP.max_selected, 
                                                                    int(LS[units_index] / AHP.max_selected))

        target_unit_index = LabelIndex.target_unit_encoding
        action_list[target_unit_index] = action_list[target_unit_index].reshape(batch_size, 1, 
                                                                                int(LS[target_unit_index]))

        action = ArgsActionLogits(*action_list)
        return action

    @staticmethod
    def label2actionlist(label):
        ''' 
        input: [batch_size x label_feature_size]
        outoput: args action list
        '''

        # TOCHANGE
        batch_size = label.shape[0]
        action = None

        action_list = []
        last_index = 0
        for i in LabelIndex:
            label_i = label[:, last_index:last_index + LS[i]]
            print('added label_i.shape:', label_i.shape) if debug else None
            action_list.append(label_i)
            last_index += LS[i]

        tue_index = LabelIndex.target_location_encoding
        print("action_list[tue_index].shape before:", action_list[tue_index].shape) if debug else None  
        action_list[tue_index] = action_list[tue_index].reshape(batch_size, SCHP.world_size, 
                                                                int(LS[tue_index] / SCHP.world_size))
        print("action_list[tue_index].shape before:", action_list[tue_index].shape) if debug else None

        action = action_list
        return action
