#!/usr/bin/env python
# -*- coding: utf-8 -*-

" For the multi-source state, including the entity state (Partial), \
the statistical state (Global) and the map state (Image)."

import time

__author__ = "Ruo-Ze Liu"

debug = False


class MsState(object):
    '''
    For the state with multi-source state, to be different with state which has only single vector or image
    '''

    def __init__(self, entity_state=None, statistical_state=None, map_state=None):
        super(MsState, self).__init__() 
        # called enetity state in alphastar
        # tensor, shape: [batch_size, entity_size, embedding_size]
        self.entity_state = entity_state  

        # called scalar state in alphastar
        # list of tensors, the reason for why this not be combined as one tensor is due to 
        # that we need do different pre-processing for different statistica
        self.statistical_state = statistical_state   

        # called spatital state in alphastar
        # tensor, shape: [batch_size, channel_size, width, height]
        self.map_state = map_state

        self._shape = None

    def _get_shape(self):
        shape1 = str(self.entity_state.shape)
        shape2 = ["%s" % str(s.shape) for s in self.statistical_state]
        shape3 = str(self.map_state.shape)

        self.shape1 = '\nentity_state: ' + shape1 + ';'
        self.shape2 = '\nstatistical_state: ' + ','.join(shape2) + ';'
        self.shape3 = '\nmap_state: ' + shape3 + '.'

        self._shape = self.shape1 + self.shape2 + self.shape3

    def toList(self):
        return [self.entity_state, self.statistical_state, self.map_state]

    def to(self, device):

        print('self.entity_state.device:', self.entity_state.device) if debug else None
        self.entity_state = self.entity_state.to(device).float()
        print('self.entity_state.device:', self.entity_state.device) if debug else None

        self.statistical_state = [s.to(device).float() for s in self.statistical_state]

        self.map_state = self.map_state.to(device).float()

        # note there is nothing retun!

    @property
    def shape(self):
        if self._shape is None:
            self._get_shape()

        return self._shape

    @property    
    def device(self):
        return self.entity_state.device

    def __str__(self):
        # "Multi-source State"
        if self._shape is None:
            self._get_shape()

        return self._shape
