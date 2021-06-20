#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for SL losses."

import numpy as np

import torch
import torch.nn as nn

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

__author__ = "Ruo-Ze Liu"

debug = False


def get_sl_loss(traj_batch, model):
    # criterion = nn.CrossEntropyLoss()
    # due to CrossEntropyLoss only accepts loss with lables.shape = [N]
    # we define a loss accept soft_target, which label.shape = [N, C] 
    def cross_entropy(pred, soft_targets):
        # class is always in the last dim
        logsoftmax = nn.LogSoftmax(dim=-1)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), -1)) 
    criterion = cross_entropy

    loss = 0
    feature_size = Feature.getSize()
    label_size = Label.getSize()

    print('traj_batch.shape:', traj_batch.shape) if 1 else None
    batch_size = traj_batch.shape[0]
    seq_len = traj_batch.shape[1]

    feature = traj_batch[:, :, :feature_size].reshape(batch_size * seq_len, feature_size)
    label = traj_batch[:, :, feature_size:feature_size + label_size].reshape(batch_size * seq_len, label_size)
    is_final = traj_batch[:, :, -1:]

    state = Feature.feature2state(feature)
    print('state:', state) if debug else None

    action_gt = Label.label2action(label)
    print('action_gt:', action_gt) if debug else None

    device = next(model.parameters()).device
    print("model.device:", device) if debug else None

    state.to(device)
    action_gt.to(device)

    def unroll(state, batch_size=None, sequence_length=None):
        action_pt, _, _ = model.forward(state, batch_size=batch_size, sequence_length=sequence_length, return_logits=True)
        return action_pt

    action_pt = unroll(state, batch_size=batch_size, sequence_length=seq_len)
    print('action_pt:', action_pt) if debug else None

    loss = get_classify_loss(action_pt, action_gt, criterion)
    print('loss:', loss) if debug else None

    return loss


def get_classify_loss(action_pt, action_gt, criterion):
    loss = 0

    action_type_loss = criterion(action_pt.action_type, action_gt.action_type)
    loss += action_type_loss

    delay_loss = criterion(action_pt.delay, action_gt.delay)
    loss += delay_loss

    queue_loss = criterion(action_pt.queue, action_gt.queue)
    loss += queue_loss

    units_loss = torch.tensor([0])
    if action_gt.units is not None and action_pt.units is not None:
        print('action_gt.units.shape:', action_gt.units.shape) if debug else None
        print('action_pt.units.shape:', action_pt.units.shape) if debug else None

        units_size = action_gt.units.shape[-1]

        units_loss = criterion(action_pt.units, action_gt.units)
        # units_loss = 0
        loss += units_loss

    target_unit_loss = torch.tensor([0])    
    if action_gt.target_unit is not None and action_pt.target_unit is not None:
        print('action_gt.target_unit.shape:', action_gt.target_unit.shape) if debug else None
        print('action_pt.target_unit.shape:', action_pt.target_unit.shape) if debug else None

        units_size = action_gt.target_unit.shape[-1]

        target_unit_loss = criterion(action_pt.target_unit, action_gt.target_unit)
        loss += target_unit_loss

    target_location_loss = torch.tensor([0])      
    if action_gt.target_location is not None and action_pt.target_location is not None:
        print('action_gt.target_location.shape:', action_gt.target_location.shape) if debug else None
        print('action_pt.target_location.shape:', action_pt.target_location.shape) if debug else None

        batch_size = action_gt.target_location.shape[0]

        target_location_loss = criterion(action_pt.target_location.reshape(batch_size, -1), action_gt.target_location.reshape(batch_size, -1))
        loss += target_location_loss

    # test, only return action_type_loss
    # return loss
    loss_list = [action_type_loss, delay_loss, queue_loss, units_loss, target_unit_loss, target_location_loss]
    return loss, loss_list
