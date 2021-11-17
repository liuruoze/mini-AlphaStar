#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for SL losses for mutli-gpu."

import traceback

import numpy as np

import torch
import torch.nn as nn

from pysc2.lib.actions import RAW_FUNCTIONS

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import utils as SU


__author__ = "Ruo-Ze Liu"

debug = False


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


def get_sl_loss(traj_batch, model, use_mask=True, use_eval=False):
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

    # state.to(device)
    # print('state:', state) if debug else None

    # action_gt.to(device)
    # print('action_gt:', action_gt) if debug else None

    loss = torch.tensor([0.])
    loss_list = [0., 0., 0., 0., 0., 0.]
    acc_num_list = [0., 0., 0., 0., 0., 0.]    

    # we can't make them all into as a list or into ArgsActionLogits
    # if we do that, the pytorch DDP will cause a runtime error, just like the loss don't include all parameters
    # This error is strange, so we choose to use a specific loss writing schema for multi-gpu calculation.
    action_pred, action_type_logits, delay_logits, queue_logits, \
        units_logits, target_unit_logits, \
        target_location_logits = model.forward(state, batch_size=batch_size, 
                                               sequence_length=seq_len, multi_gpu_supvised_learning=True)

    print('action_pred.shape', action_pred.shape) if debug else None   

    loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, action_type_logits,
                                                             delay_logits, queue_logits, units_logits,
                                                             target_unit_logits, target_location_logits, criterion,
                                                             device)
    acc_num_list = SU.get_accuracy(action_gt.action_type, action_pred, device)

    print('loss', loss) if debug else None

    return loss, loss_list, acc_num_list


def get_sl_loss_for_tensor(features, labels, model, use_mask=True, use_eval=False):

    criterion = cross_entropy

    batch_size = features.shape[0]
    seq_len = features.shape[1]

    assert batch_size == labels.shape[0]
    assert seq_len == labels.shape[1]

    features = features.reshape(batch_size * seq_len, -1)
    labels = labels.reshape(batch_size * seq_len, -1)

    state = Feature.feature2state(features)
    print('state:', state) if debug else None

    action_gt = Label.label2action(labels)
    print('action_gt:', action_gt) if debug else None

    device = next(model.parameters()).device
    print("model.device:", device) if debug else None

    # state.to(device)
    # print('state:', state) if debug else None

    # action_gt.to(device)
    # print('action_gt:', action_gt) if debug else None

    loss = torch.tensor([0.])
    loss_list = [0., 0., 0., 0., 0., 0.]
    acc_num_list = [0., 0., 0., 0., 0., 0.]    

    # we can't make them all into as a list or into ArgsActionLogits
    # if we do that, the pytorch DDP will cause a runtime error, just like the loss don't include all parameters
    # This error is strange, so we choose to use a specific loss writing schema for multi-gpu calculation.
    action_pred, action_type_logits, delay_logits, queue_logits, \
        units_logits, target_unit_logits, \
        target_location_logits = model.forward(state, batch_size=batch_size, 
                                               sequence_length=seq_len, multi_gpu_supvised_learning=True)

    print('action_pred.shape', action_pred.shape) if debug else None   

    loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, action_type_logits,
                                                             delay_logits, queue_logits, units_logits,
                                                             target_unit_logits, target_location_logits, criterion,
                                                             device)
    acc_num_list = SU.get_accuracy(action_gt.action_type, action_pred, device)

    print('loss', loss) if debug else None

    return loss, loss_list, acc_num_list


def get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, action_type, delay, queue, units,
                                           target_unit, target_location, criterion, device):
    loss = 0.

    # consider using move camera weight
    move_camera_weight = SU.get_move_camera_weight_in_SL(action_gt.action_type, action_pred, device).reshape(-1)
    #move_camera_weight = None
    action_type_loss = criterion(action_gt.action_type, action_type, mask=move_camera_weight)
    loss += action_type_loss

    #mask_tensor = get_one_way_mask_in_SL(action_gt.action_type, device)
    mask_tensor = SU.get_two_way_mask_in_SL(action_gt.action_type, action_pred, device)

    # we don't consider delay loss now
    delay_loss = criterion(action_gt.delay, delay)
    loss += delay_loss * 0

    queue_loss = criterion(action_gt.queue, queue, mask=mask_tensor[:, 2].reshape(-1))
    loss += queue_loss

    select_size = action_gt.units.shape[1]
    units_size = action_gt.units.shape[-1]
    units_mask = mask_tensor[:, 3]
    units_mask = units_mask.repeat(select_size, 1).transpose(1, 0)
    units_mask = units_mask.reshape(-1)

    units_loss = criterion(action_gt.units.reshape(-1, units_size), units.reshape(-1, units_size), mask=units_mask)
    loss += units_loss

    target_unit_loss = criterion(action_gt.target_unit.squeeze(-2), target_unit.squeeze(-2), mask=mask_tensor[:, 4].reshape(-1))
    loss += target_unit_loss

    batch_size = action_gt.target_location.shape[0]
    target_location_loss = criterion(action_gt.target_location.reshape(batch_size, -1),
                                     target_location.reshape(batch_size, -1), mask=mask_tensor[:, 5].reshape(-1))
    loss += target_location_loss

    return loss, [action_type_loss.item(), delay_loss.item(), queue_loss.item(), units_loss.item(), target_unit_loss.item(), target_location_loss.item()]


def get_classify_loss_for_multi_gpu(action_gt, action_type, delay, queue, units, target_unit, target_location, criterion):
    loss = 0.

    action_type_loss = criterion(action_gt.action_type, action_type)
    loss += action_type_loss

    delay_loss = criterion(action_gt.delay, delay)
    loss += delay_loss

    queue_loss = criterion(action_gt.queue, queue)
    loss += queue_loss

    units_size = action_gt.units.shape[-1]
    units_loss = criterion(action_gt.units.reshape(-1, units_size), units.reshape(-1, units_size))
    loss += units_loss

    target_unit_loss = criterion(action_gt.target_unit.squeeze(-2), target_unit.squeeze(-2))
    loss += target_unit_loss

    batch_size = action_gt.target_location.shape[0]
    target_location_loss = criterion(action_gt.target_location.reshape(batch_size, -1), target_location.reshape(batch_size, -1))
    loss += target_location_loss

    return loss.item(), [action_type_loss.item(), delay_loss.item(), queue_loss.item(), units_loss.item(), target_unit_loss.item(), target_location_loss.item()]
