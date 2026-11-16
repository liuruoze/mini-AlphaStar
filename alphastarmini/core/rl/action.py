#!/usr/bin/env python
# -*- coding: utf-8 -*-

" For the action with arguments, to be different with action which has only scalar value"

from collections import namedtuple

import numpy as np

import torch

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

__author__ = "Ruo-Ze Liu"

debug = False


class ArgsAction(object):
    '''
    For the action with arguments, to be different with action which has only scalar value
    '''
    __slots__ = ('action_type', 'delay', 'queue', 'units', 'target_unit', 'target_location')

    max_actions = LS.action_type_encoding
    max_delay = LS.delay_encoding
    max_queue = LS.queue_encoding

    max_units = AHP.max_entities
    max_selected = AHP.max_selected
    output_map_size = SCHP.world_size

    def __init__(self, action_type=-1, delay=SCHP.sc2_default_delay, queue=None, 
                 units=None, target_unit=None, 
                 target_location=None):
        super(ArgsAction, self).__init__()
        self.action_type = action_type
        self.delay = delay
        self.queue = queue
        self.units = units 
        self.target_unit = target_unit
        self.target_location = target_location

    def toTenser(self):
        # for numpy or int object to tensor
        # note, this is used in the one element case
        action_type_encoding = - torch.ones(1, 1, dtype=torch.long)
        if self.action_type is not None:
            action_type_encoding[0, 0] = self.action_type

        delay_encoding = - torch.ones(1, 1, dtype=torch.long)    
        if self.delay is not None:
            delay_encoding[0, 0] = self.delay

        queue_encoding = - torch.ones(1, 1, dtype=torch.long)    
        if self.queue is not None:
            queue_encoding[0, 0] = self.queue

        select_units_encoding = - torch.ones(1, self.max_selected, 1, dtype=torch.long)
        if self.units is not None:
            for i, u in enumerate(self.units):
                if i >= self.max_selected:
                    break
                # ensure the unit index not beyond the max entities we considered
                if u >= self.max_units:
                    u = self.max_units - 1
                select_units_encoding[0, i, 0] = u

        target_unit_encoding = - torch.ones(1, 1, 1, dtype=torch.long)
        if self.target_unit is not None:
            # ensure the target unit index not beyond the max entities we considered
            u = self.target_unit
            if u >= self.max_units:
                u = self.max_units - 1
            target_unit_encoding[0, 0, 0] = u

        target_location_encoding = - torch.ones(1, 2, dtype=torch.long)
        if self.target_location is not None:
            x = self.target_location[0]
            y = self.target_location[1]
            # ensure the position not beyond the map size
            if x >= self.output_map_size:
                x = self.output_map_size - 1
            if y >= self.output_map_size:
                y = self.output_map_size - 1
            target_location_encoding[0, 0] = x  
            target_location_encoding[0, 1] = y  

        return ArgsAction(action_type_encoding, delay_encoding, queue_encoding, select_units_encoding,
                          target_unit_encoding, target_location_encoding)

    def toArray(self):
        # for numpy or int object to tensor
        # note, this is used in the one element case
        # in 1.05 version, we first init them to -1
        action_type_encoding = - np.ones((1, 1), dtype=np.int32)
        if self.action_type is not None:
            action_type_encoding[0, 0] = self.action_type

        delay_encoding = - np.ones((1, 1), dtype=np.int32)    
        if self.delay is not None:
            delay_encoding[0, 0] = self.delay

        queue_encoding = - np.ones((1, 1), dtype=np.int32)    
        if self.queue is not None:
            queue_encoding[0, 0] = self.queue

        select_units_encoding = - np.ones((1, self.max_selected, 1), dtype=np.int32)
        if self.units is not None:
            for i, u in enumerate(self.units):
                if i >= self.max_selected:
                    break
                # ensure the unit index not beyond the max entities we considered
                if u >= self.max_units:
                    u = self.max_units - 1
                select_units_encoding[0, i, 0] = u

        target_unit_encoding = - np.ones((1, 1, 1), dtype=np.int32)
        if self.target_unit is not None:
            # ensure the target unit index not beyond the max entities we considered
            u = self.target_unit
            if u >= self.max_units:
                u = self.max_units - 1
            target_unit_encoding[0, 0, 0] = u

        target_location_encoding = - np.ones((1, 2), dtype=np.int32)
        if self.target_location is not None:
            x = self.target_location[0]
            y = self.target_location[1]
            # ensure the position not beyond the map size
            if x >= self.output_map_size:
                x = self.output_map_size - 1
            if y >= self.output_map_size:
                y = self.output_map_size - 1
            target_location_encoding[0, 0] = x  
            target_location_encoding[0, 1] = y 

        return ArgsAction(action_type_encoding, delay_encoding, queue_encoding, select_units_encoding,
                          target_unit_encoding, target_location_encoding)

    def toLogits(self, tag_list=None):
        # for tensor action to tensor action with one-hot embedding
        batch_size = self.action_type.shape[0]

        action_type_encoding = L.tensor_one_hot(self.action_type, self.max_actions).squeeze(-2)
        print('self.action_type:', self.action_type) if debug else None
        print('action_type_encoding:', action_type_encoding) if debug else None

        delay_encoding = L.tensor_one_hot(self.delay, self.max_delay).squeeze(-2)
        print('self.delay:', self.delay) if debug else None
        print('delay_encoding:', delay_encoding) if debug else None

        queue_encoding = L.tensor_one_hot(self.queue, self.max_queue).squeeze(-2)
        print('self.queue:', self.queue) if debug else None
        print('queue_encoding:', queue_encoding) if debug else None

        print('self.units_index:', self.units) if debug else None
        select_units_encoding = L.tensor_one_hot(self.units, self.max_units).squeeze(-2)
        print('select_units_encoding:', select_units_encoding) if debug else None

        target_unit_encoding = L.tensor_one_hot(self.target_unit, self.max_units).squeeze(-2)
        print('self.target_unit_index:', self.target_unit) if debug else None
        print('target_unit_encoding:', target_unit_encoding) if debug else None

        target_location_encoding = torch.zeros(batch_size, self.output_map_size, self.output_map_size)
        for i, z in enumerate(self.target_location):
            (x, y) = z
            target_row_col = (y, x)
            target_location_encoding[i, y, x] = 1

        print('self.target_location:', self.target_location) if debug else None
        print('target_location_encoding:', target_location_encoding) if debug else None
        return ArgsActionLogits(action_type_encoding, delay_encoding, queue_encoding, select_units_encoding,
                                target_unit_encoding, target_location_encoding)

    def toLogits_numpy(self, tag_list=None):
        # for tensor action to tensor action with one-hot embedding
        # note if the value is -1, np_one_hot will make the last index item has one in the one-hot embediing.
        batch_size = self.action_type.shape[0]

        action_type_encoding = L.np_one_hot(self.action_type, self.max_actions).squeeze(-2)
        print('self.action_type:', self.action_type) if debug else None
        print('action_type_encoding:', action_type_encoding) if debug else None

        delay_encoding = L.np_one_hot(self.delay, self.max_delay).squeeze(-2)
        print('self.delay:', self.delay) if debug else None
        print('delay_encoding:', delay_encoding) if debug else None

        queue_encoding = L.np_one_hot(self.queue, self.max_queue).squeeze(-2)
        print('self.queue:', self.queue) if debug else None
        print('queue_encoding:', queue_encoding) if debug else None

        select_units_encoding = L.np_one_hot(self.units, self.max_units).squeeze(-2)
        print('self.units_index:', self.units) if debug else None
        print('select_units_encoding:', select_units_encoding) if debug else None

        target_unit_encoding = L.np_one_hot(self.target_unit, self.max_units).squeeze(-2)
        print('self.target_unit_index:', self.target_unit) if debug else None
        print('target_unit_encoding:', target_unit_encoding) if debug else None

        target_location_encoding = np.zeros((batch_size, self.output_map_size, self.output_map_size))
        for i, z in enumerate(self.target_location):
            (x, y) = z
            # note the x, y axis are opposiate to row, col!
            target_row_col = (y, x)
            target_location_encoding[i, y, x] = 1
        print('self.target_location:', self.target_location) if debug else None
        print('target_location_encoding:', target_location_encoding) if debug else None
        return ArgsActionLogits(action_type_encoding, delay_encoding, queue_encoding, select_units_encoding,
                                target_unit_encoding, target_location_encoding)

    def toList(self):
        return [self.action_type, self.delay, self.queue, self.units, 
                self.target_unit, self.target_location]

    def get_shape(self): 
        result = """action_type:%s, delay:%s, queue:%s, units:%s, 
        target_unit:%s, target_location:%s""" % (self.action_type.shape, 
                                                 self.delay.shape, self.queue.shape, self.units.shape, 
                                                 self.target_unit.shape, 
                                                 self.target_location.shape)
        return result

    def __str__(self): 
        result = """action_type:%s, delay:%s, queue:%s, units:%s, 
        target_unit:%s, target_location:%s""" % (self.action_type, 
                                                 self.delay, self.queue, self.units, 
                                                 self.target_unit,  
                                                 self.target_location)
        return result

    @property    
    def device(self):
        return self.action_type.device

    def to(self, device):
        self.action_type = self.action_type.to(device)
        self.delay = self.delay.to(device)
        self.queue = self.queue.to(device)
        self.units = self.units.to(device)
        self.target_unit = self.target_unit.to(device)
        self.target_location = self.target_location.to(device)

    def detach(self):
        self.action_type = self.action_type.detach()
        self.delay = self.delay.detach()
        self.queue = self.queue.detach()
        self.units = self.units.detach()
        self.target_unit = self.target_unit.detach()
        self.target_location = self.target_location.detach()

        return self

    def clone(self):
        action_type = self.action_type.detach().clone()
        delay = self.delay.detach().clone()
        queue = self.queue.detach().clone()
        units = self.units.detach().clone()
        target_unit = self.target_unit.detach().clone()
        target_location = self.target_location.detach().clone()

        return ArgsAction(action_type, delay, queue, units, 
                          target_unit, target_location)


class ArgsActionLogits(object):
    '''
    For the action with arguments and using logits
    '''
    __slots__ = ('action_type', 'delay', 'queue', 'units', 'target_unit', 'target_location')

    def __init__(self, action_type, delay, queue, units, 
                 target_unit, target_location):
        super(ArgsActionLogits, self).__init__()
        self.action_type = action_type
        self.delay = delay
        self.queue = queue
        self.units = units
        self.target_unit = target_unit
        self.target_location = target_location

    def toList(self):
        return [self.action_type, self.delay, self.queue, self.units, 
                self.target_unit, self.target_location]

    def __str__(self):
        shape1 = str(self.action_type.shape)
        shape2 = str(self.delay.shape)
        shape3 = str(self.queue.shape)
        shape4 = str(self.units.shape) if self.units is not None else "None"
        shape5 = str(self.target_unit.shape) if self.target_unit is not None else "None"
        shape6 = str(self.target_location.shape) if self.target_location is not None else "None"

        return "%s %s %s %s %s %s" % (shape1, shape2, 
                                      shape3, 
                                      shape4, 
                                      shape5, shape6)

    def to(self, device):
        self.action_type = self.action_type.to(device).float()
        self.delay = self.delay.to(device).float()
        self.queue = self.queue.to(device).float()
        self.units = self.units.to(device).float()
        self.target_unit = self.target_unit.to(device).float()
        self.target_location = self.target_location.to(device).float()

    def detach(self):
        self.action_type = self.action_type.detach()
        self.delay = self.delay.detach()
        self.queue = self.queue.detach()
        self.units = self.units.detach()
        self.target_unit = self.target_unit.detach()
        self.target_location = self.target_location.detach()

        return self

    def clone(self):
        action_type = self.action_type.detach().clone()
        delay = self.delay.detach().clone()
        queue = self.queue.detach().clone()
        units = self.units.detach().clone()
        target_unit = self.target_unit.detach().clone()
        target_location = self.target_location.detach().clone()

        return ArgsActionLogits(action_type, delay, queue, units, 
                                target_unit, target_location)

    @property    
    def device(self):
        return self.action_type.device


def test():
    # refer to arch_model.py for test
    pass
