#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for SL losses."

import traceback

import numpy as np

import torch
import torch.nn as nn

from pysc2.lib.actions import RAW_FUNCTIONS

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_utils as SU

__author__ = "Ruo-Ze Liu"

debug = False


def get_sl_loss(traj_batch, model, use_mask=True, use_eval=False):
    # criterion = nn.CrossEntropyLoss()
    # due to CrossEntropyLoss only accepts loss with lables.shape = [N]
    # we define a loss accept soft_target, which label.shape = [N, C]
    # for some rows, it should not be added to loss, so we also need a mask one
    def cross_entropy(soft_targets, pred, mask=None):
        # class is always in the last dim
        logsoftmax = nn.LogSoftmax(dim=-1)
        x_1 = - soft_targets * logsoftmax(pred)
        x_2 = torch.sum(x_1, -1)
        if mask is not None:
            x_2 = x_2 * mask
        x_4 = torch.mean(x_2)
        return x_4

    criterion = cross_entropy

    loss = 0
    feature_size = Feature.getSize()
    label_size = Label.getSize()

    print('traj_batch.shape:', traj_batch.shape) if debug else None
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
    print('state:', state) if debug else None

    action_gt.to(device)
    print('action_gt:', action_gt) if debug else None

    loss = torch.tensor([0.])
    loss_list = [0., 0., 0., 0., 0., 0.]
    acc_num_list = [0., 0., 0., 0., 0., 0.]    

    action_logits_pred, action_pred, _, select_units_num = model.forward(state, batch_size=batch_size, sequence_length=seq_len, return_logits=True)

    loss, loss_list = get_classify_loss(action_gt, action_logits_pred, criterion, device, use_mask=use_mask)    

    # if use_eval:
    acc_num_list = SU.get_accuracy(action_gt.action_type, action_pred.action_type, device)

    return loss, loss_list, acc_num_list


def get_classify_loss(action_gt, action_pred, criterion, device, use_mask=True):
    loss = 0

    # belwo is for test
    action_ground_truth = action_gt.action_type
    ground_truth_raw_action_id = torch.nonzero(action_ground_truth, as_tuple=True)[-1]
    mask_list = [] 
    print('ground_truth_raw_action_id', ground_truth_raw_action_id) if debug else None
    for raw_action_id in ground_truth_raw_action_id:
        mask_list.append(SU.get_mask_by_raw_action_id(raw_action_id.item()))
    print('mask_list', mask_list) if debug else None
    mask_tensor = torch.tensor(mask_list)
    print('mask_tensor:', mask_tensor) if debug else None
    mask_tensor = mask_tensor.to(device)
    print('mask_tensor:', mask_tensor) if debug else None

    action_type_loss = criterion(action_gt.action_type, action_pred.action_type)
    loss += action_type_loss

    # we don't consider delay loss now
    delay_loss = criterion(action_gt.delay, action_pred.delay)
    loss += delay_loss * 0

    queue_loss = action_pred.queue.sum()  # criterion(action_gt.queue, action_pred.queue)
    queue_loss_mask = criterion(action_gt.queue, action_pred.queue, mask=mask_tensor[:, 2].reshape(-1))

    if use_mask:
        queue_loss = queue_loss_mask
    loss += queue_loss

    def findNaN(x):
        return torch.isnan(x).any()

    units_loss = torch.tensor([0])
    if action_pred.units is not None and action_gt.units is not None:
        units_size = action_pred.units.shape[-1]
        select_size = action_pred.units.shape[1]

        if not findNaN(action_pred.units) and not findNaN(action_gt.units):

            units_loss = criterion(action_gt.units, action_pred.units)
            units_loss_split = criterion(action_gt.units.reshape(-1, units_size), action_pred.units.reshape(-1, units_size))

            units_mask = mask_tensor[:, 3]
            units_mask = units_mask.repeat(select_size, 1).transpose(1, 0)
            units_mask = units_mask.reshape(-1)

            units_loss_split_mask = criterion(action_gt.units.reshape(-1, units_size), action_pred.units.reshape(-1, units_size), mask=units_mask)

            print('units_loss', units_loss) if debug else None
            print('units_loss_split', units_loss_split) if debug else None
            print('units_loss_split_mask', units_loss_split_mask) if debug else None

            if use_mask:
                units_loss = units_loss_split_mask
            loss += units_loss

    target_unit_loss = torch.tensor([0])    
    if action_pred.target_unit is not None and action_gt.target_unit is not None:
        units_size = action_pred.target_unit.shape[-1]

        if not findNaN(action_pred.target_unit) and not findNaN(action_gt.target_unit):
            target_unit_loss = criterion(action_gt.target_unit, action_pred.target_unit)

            # target unit only has 1 unit, so we don't need to split it
            target_unit_loss_mask = criterion(action_gt.target_unit, action_pred.target_unit, mask=mask_tensor[:, 4].reshape(-1))

            print('target_unit_loss', target_unit_loss) if debug else None
            print('target_unit_loss_mask', target_unit_loss_mask) if debug else None

            if use_mask:
                target_unit_loss = target_unit_loss_mask
            loss += target_unit_loss

    target_location_loss = torch.tensor([0])
    if action_pred.target_location is not None and action_gt.target_location is not None:
        batch_size = action_pred.target_location.shape[0]

        if not findNaN(action_pred.target_location) and not findNaN(action_gt.target_location):

            target_location_loss = criterion(action_gt.target_location.reshape(batch_size, -1), action_pred.target_location.reshape(batch_size, -1))
            target_location_loss_mask = criterion(action_gt.target_location.reshape(batch_size, -1), action_pred.target_location.reshape(batch_size, -1), 
                                                  mask=mask_tensor[:, 5].reshape(-1))            

            print('target_location_loss', target_location_loss) if debug else None
            print('target_location_loss_mask', target_location_loss_mask) if debug else None

            if use_mask:
                target_location_loss = target_location_loss_mask
            loss += target_location_loss

    # test, only return action_type_loss
    # return loss
    loss_list = [action_type_loss, delay_loss, queue_loss, units_loss, target_unit_loss, target_location_loss]
    return loss, loss_list
