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


__author__ = "Ruo-Ze Liu"

debug = False


def get_sl_loss(traj_batch, model, use_eval=False):
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

    print('traj_batch.shape:', traj_batch.shape) if 1 else None
    batch_size = traj_batch.shape[0]
    seq_len = traj_batch.shape[1]

    feature = traj_batch[:, :, :feature_size].reshape(batch_size * seq_len, feature_size)
    label = traj_batch[:, :, feature_size:feature_size + label_size].reshape(batch_size * seq_len, label_size)
    is_final = traj_batch[:, :, -1:]

    state = Feature.feature2state(feature)
    print('state:', state) if 1 else None

    action_gt = Label.label2action(label)
    print('action_gt:', action_gt) if 1 else None

    device = next(model.parameters()).device
    print("model.device:", device) if 1 else None

    state.to(device)
    print('state:', state) if 0 else None

    action_gt.to(device)
    print('action_gt:', action_gt) if 0 else None

    def unroll(state, batch_size=None, sequence_length=None):
        action_logits_pred, action_pred, _ = model.forward(state, batch_size=batch_size, sequence_length=sequence_length, return_logits=True)
        return action_logits_pred, action_pred

    try:
        action_logits_pred, action_pred = unroll(state, batch_size=batch_size, sequence_length=seq_len)
        print('action_logits_pred:', action_logits_pred) if 1 else None

        loss, loss_list = get_classify_loss(action_gt, action_logits_pred, criterion, device)    

        # if use_eval:
        acc_num_list = get_accuracy(action_gt.action_type, action_pred.action_type, device)

    except Exception as e:
        print(traceback.format_exc())

        loss = torch.tensor([0.])
        loss_list = []

    # if use_eval:
    #     return loss, loss_list, acc_num_action_type, all_num

    return loss, loss_list, acc_num_list


def get_accuracy(ground_truth, predict, device):
    accuracy = 0.

    ground_truth_new = torch.nonzero(ground_truth, as_tuple=True)[-1]
    ground_truth_new = ground_truth_new.to(device)
    print('ground_truth', ground_truth_new)

    predict_new = predict.reshape(-1)
    print('predict_new', predict_new)

    # calculate how many move_camera? the id is 168 in raw_action
    MOVE_CAMERA_ID = 168
    #camera_num_action_type = torch.sum(MOVE_CAMERA_ID == ground_truth_new)
    move_camera_index = (ground_truth_new == MOVE_CAMERA_ID).nonzero(as_tuple=True)[0]
    non_camera_index = (ground_truth_new != MOVE_CAMERA_ID).nonzero(as_tuple=True)[0]

    print('move_camera_index', move_camera_index)    
    print('non_camera_index', non_camera_index)   

    print('for any type action')
    right_num, all_num = get_right_and_all_num(ground_truth_new, predict_new)

    print('for move_camera action')
    camera_ground_truth_new = ground_truth_new[move_camera_index]
    camera_predict_new = predict_new[move_camera_index]
    camera_right_num, camera_all_num = get_right_and_all_num(camera_ground_truth_new, camera_predict_new)

    print('for non-camera action')
    non_camera_ground_truth_new = ground_truth_new[non_camera_index]
    non_camera_predict_new = predict_new[non_camera_index]
    non_camera_right_num, non_camera_all_num = get_right_and_all_num(non_camera_ground_truth_new, non_camera_predict_new)

    return [right_num, all_num, camera_right_num, camera_all_num, non_camera_right_num, non_camera_all_num]


def get_right_and_all_num(gt, pred):
    acc_num_action_type = torch.sum(pred == gt)
    print('acc_num_action_type', acc_num_action_type)

    right_num = acc_num_action_type.item()
    print('right_num', right_num)

    all_num = gt.shape[0]
    print('all_num', all_num)

    accuracy = right_num / (all_num + 1e-9)
    print('accuracy', accuracy)

    return right_num, all_num


def get_classify_loss(action_gt, action_pred, criterion, device):
    loss = 0

    # belwo is for test
    action_ground_truth = action_gt.action_type
    ground_truth_raw_action_id = torch.nonzero(action_ground_truth, as_tuple=True)[-1]
    mask_list = []
    print('ground_truth_raw_action_id', ground_truth_raw_action_id) 
    for raw_action_id in ground_truth_raw_action_id:
        mask_list.append(get_mask_by_raw_action_id(raw_action_id.item()))
    print('mask_list', mask_list) if 0 else None
    mask_tensor = torch.tensor(mask_list)
    print('mask_tensor:', mask_tensor) if 0 else None
    mask_tensor = mask_tensor.to(device)
    print('mask_tensor:', mask_tensor) if 0 else None

    action_type_loss = criterion(action_gt.action_type, action_pred.action_type)
    loss += action_type_loss

    # we don't consider delay loss now
    delay_loss = criterion(action_gt.delay, action_pred.delay)
    #loss += delay_loss

    queue_loss = criterion(action_gt.queue, action_pred.queue)
    queue_loss_mask = criterion(action_gt.queue, action_pred.queue, mask=mask_tensor[:, 2].reshape(-1))

    print('queue_loss', queue_loss)
    print('queue_loss_mask', queue_loss_mask)

    queue_loss = queue_loss_mask
    loss += queue_loss_mask

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

            print('units_loss', units_loss)
            print('units_loss_split', units_loss_split)
            print('units_loss_split_mask', units_loss_split_mask)

            units_loss = units_loss_split_mask
            loss += units_loss_split_mask

    target_unit_loss = torch.tensor([0])    
    if action_pred.target_unit is not None and action_gt.target_unit is not None:
        units_size = action_pred.target_unit.shape[-1]

        if not findNaN(action_pred.target_unit) and not findNaN(action_gt.target_unit):
            target_unit_loss = criterion(action_gt.target_unit, action_pred.target_unit)

            # target unit only has 1 unit, so we don't need to split it
            target_unit_loss_mask = criterion(action_gt.target_unit, action_pred.target_unit, mask=mask_tensor[:, 4].reshape(-1))

            print('target_unit_loss', target_unit_loss)
            print('target_unit_loss_mask', target_unit_loss_mask)

            target_unit_loss = target_unit_loss_mask
            loss += target_unit_loss_mask

    target_location_loss = torch.tensor([0])
    if action_pred.target_location is not None and action_gt.target_location is not None:
        batch_size = action_pred.target_location.shape[0]

        if not findNaN(action_pred.target_location) and not findNaN(action_gt.target_location):

            target_location_loss = criterion(action_gt.target_location.reshape(batch_size, -1), action_pred.target_location.reshape(batch_size, -1))
            target_location_loss_mask = criterion(action_gt.target_location.reshape(batch_size, -1), action_pred.target_location.reshape(batch_size, -1), 
                                                  mask=mask_tensor[:, 5].reshape(-1))            

            print('target_location_loss', target_location_loss)
            print('target_location_loss_mask', target_location_loss_mask)

            target_location_loss = target_location_loss_mask
            loss += target_location_loss_mask

    # test, only return action_type_loss
    # return loss
    loss_list = [action_type_loss, delay_loss, queue_loss, units_loss, target_unit_loss, target_location_loss]
    return loss, loss_list


def get_mask_by_raw_action_id(raw_action_id):
    need_args = RAW_FUNCTIONS[raw_action_id].args

    # action type and delay is always enable
    action_mask = [1, 1, 0, 0, 0, 0]

    for arg in need_args:
        print("arg:", arg) if debug else None
        if arg.name == 'queued':
            action_mask[2] = 1
        elif arg.name == 'unit_tags':
            action_mask[3] = 1
        elif arg.name == 'target_unit_tag':
            action_mask[4] = 1
        elif arg.name == 'world':
            action_mask[5] = 1                       

    print('action_mask:', action_mask) if debug else None

    return action_mask
