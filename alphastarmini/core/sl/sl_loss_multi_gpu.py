#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for SL losses for mutli-gpu."

import traceback

import numpy as np

import torch
import torch.nn as nn

from pysc2.lib.actions import RAW_FUNCTIONS as F

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

from alphastarmini.lib import utils as L

__author__ = "Ruo-Ze Liu"

debug = False


# criterion = nn.CrossEntropyLoss()
# due to CrossEntropyLoss only accepts loss with lables.shape = [N]
# we define a loss accept soft_target, which label.shape = [N, C]
# for some rows, it should not be added to loss, so we also need a mask one
def cross_entropy(soft_targets, pred, mask=None, 
                  debug=False, outlier_remove=True, 
                  entity_nums=None, select_size=None,
                  select_units_num=None, use_select_size=True,
                  avg_type='None'):
    # class is always in the last dim
    logsoftmax = nn.LogSoftmax(dim=-1)
    x_1 = - soft_targets * logsoftmax(pred)

    if debug:
        for i, (t, p) in enumerate(zip(soft_targets, logsoftmax(pred))):
            # print('t', t)
            # print('t.shape', t.shape)
            # print('p', p)
            # print('p.shape', p.shape)
            value = - t * logsoftmax(p)
            # print('value', value)
            # print('value.shape', value.shape)
            m = mask[i].item()
            if value.sum() > 1e6 and m != 0:
                print('i', i)
                if entity_nums is not None:
                    if use_select_size and select_size is not None and select_units_num is not None:
                        i_1 = int(i / select_size)
                        print('i_1', i_1)
                        i_2 = i - i_1 * select_size
                        print('i_2', i_2)
                        print('entity_nums[i_1]', entity_nums[i_1])
                        print('select_units_num[i_1]', select_units_num[i_1])
                    else:
                        print('entity_nums[i]', entity_nums[i])

                print('find value large than 1e6')
                print('t', t)
                z = torch.nonzero(t, as_tuple=True)[-1]
                print('z', z)
                idx = z.item()
                print('p', p)
                q = p[idx]
                print('q', q)
                print('value', value)
                print('m', m)
                stop()

    print('x_1:', x_1) if debug else None
    print('x_1.shape:', x_1.shape) if debug else None

    x_2 = torch.sum(x_1, -1)
    print('x_2:', x_2) if debug else None
    print('x_2.shape:', x_2.shape) if debug else None

    # This mask is for each item's mask
    if mask is not None:
        x_2 = x_2 * mask

        if outlier_remove:
            outlier_mask = (x_2 >= 1e6)
            x_2 = x_2 * ~outlier_mask
        else:
            outlier_mask = (x_2 >= 1e6)
            if outlier_mask.any() > 0:
                stop()

    if use_select_size and select_size is not None and select_units_num is not None:
        x_2 = x_2.reshape(-1, select_size)
        x_2 = torch.sum(x_2, dim=-1, keepdim=True)

        avg_type = 'None'

        if avg_type == 'None':
            x_2 = x_2  # increase the multi unit weight
        elif avg_type == 'PrefSingle':
            x_2 / select_units_num.unsqueeze(dim=1)  # increase the single unit weight
        elif avg_type == 'Log':
            x_2 = x_2 / (torch.log(select_units_num.unsqueeze(dim=1).float()) + 1e-9)
        elif avg_type == 'Sqrt':
            x_2 = x_2 / torch.sqrt(select_units_num.unsqueeze(dim=1).float())
        else:
            x_2 = x_2

    x_4 = torch.mean(x_2)
    print('x_4:', x_4) if debug else None
    print('x_4.shape:', x_4.shape) if debug else None

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
    action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
        delay_logits, queue_logits, \
        units_logits, target_unit_logits, \
        target_location_logits, select_units_num = model.forward(state, batch_size=batch_size, 
                                                                 sequence_length=seq_len, multi_gpu_supvised_learning=True)

    print('action_pred.shape', action_pred.shape) if debug else None   

    loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, action_type_logits,
                                                             delay_logits, queue_logits, units_logits,
                                                             target_unit_logits, target_location_logits, 
                                                             select_units_num, criterion, device)

    acc_num_list = SU.get_accuracy(action_gt.action_type, action_pred, device)

    print('loss', loss) if debug else None

    return loss, loss_list, acc_num_list


def get_sl_loss_for_tensor(features, labels, model, decrease_smart_opertaion=False,
                           return_important=False, only_consider_small=False,
                           train=True, use_masked_loss=True):

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
    if True:
        if SCHP.map_name == 'Simple64':
            smart_action_gt = torch.zeros(1, LS.action_type_encoding, device=device).float()
            smart_action_gt[0, F.Smart_unit.id.value] = 1

            gather_action_gt = torch.zeros(1, LS.action_type_encoding, device=device).float()
            gather_action_gt[0, F.Harvest_Gather_unit.id.value] = 1

            action_gt.action_type[(action_gt.action_type == smart_action_gt).all(dim=-1).bool()] = gather_action_gt

        gt_units = action_gt.units
        units_size = gt_units.shape[-1]

        bias_action_gt = torch.zeros(1, 1, units_size, device=device).float()
        bias_action_gt[0, 0, -1] = 1

        gt_select_units_num = (~(action_gt.units == bias_action_gt).all(dim=-1)).sum(dim=-1)

        print('gt_select_units_num', gt_select_units_num) if debug else None
        print('gt_select_units_num.shape', gt_select_units_num.shape) if debug else None

        action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
            delay_logits, queue_logits, \
            units_logits, target_unit_logits, \
            target_location_logits, select_units_num, \
            hidden_state, unit_types_one = model.mimic_forward(state, 
                                                               action_gt, 
                                                               gt_select_units_num,
                                                               batch_size=batch_size, 
                                                               sequence_length=seq_len, 
                                                               multi_gpu_supvised_learning=True)

        if use_masked_loss:
            # masked loss
            loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, action_type_logits,
                                                                     delay_logits, queue_logits, units_logits,
                                                                     target_unit_logits, target_location_logits, 
                                                                     select_units_num, criterion, device, 
                                                                     decrease_smart_opertaion=decrease_smart_opertaion,
                                                                     only_consider_small=only_consider_small)    
        else:
            # noraml loss
            loss, loss_list = get_classify_loss_for_multi_gpu(action_gt, action_type_logits,
                                                              delay_logits, queue_logits, units_logits,
                                                              target_unit_logits, target_location_logits, 
                                                              criterion)

    if True:
        action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
            delay_logits, queue_logits, \
            units_logits, target_unit_logits, \
            target_location_logits, select_units_num = model.forward(state, 
                                                                     batch_size=batch_size, 
                                                                     sequence_length=seq_len, 
                                                                     multi_gpu_supvised_learning=True)

        acc_num_list, action_equal_mask = SU.get_accuracy(action_gt.action_type, action_pred, 
                                                          device, return_important=return_important)

        location_acc = SU.get_location_accuracy(action_gt.target_location, target_location, action_equal_mask, device, 
                                                strict_comparsion=True)
        selected_acc = SU.get_selected_units_accuracy(action_gt.units, units, action_gt.action_type, action_pred,
                                                      select_units_num, action_equal_mask,
                                                      device, unit_types_one, entity_nums, strict_comparsion=True)
        targeted_acc = SU.get_target_unit_accuracy(action_gt.target_unit, target_unit, action_equal_mask,
                                                   device, strict_comparsion=True)

        acc_num_list.extend(location_acc)
        acc_num_list.extend(selected_acc)
        acc_num_list.extend(targeted_acc)

    return loss, loss_list, acc_num_list


def get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, action_type_logits, 
                                           delay_logits, queue_logits, units_logits,
                                           target_unit_logits, target_location_logits, select_units_num,
                                           criterion, device, 
                                           decrease_smart_opertaion=False,
                                           only_consider_small=False,
                                           strict_comparsion=True, remove_none=True):
    loss = 0.

    # consider using move camera weight
    move_camera_weight = SU.get_move_camera_weight_in_SL(action_gt.action_type, 
                                                         action_pred, 
                                                         device, 
                                                         decrease_smart_opertaion=decrease_smart_opertaion,
                                                         only_consider_small=only_consider_small).reshape(-1)
    #move_camera_weight = None
    action_type_loss = criterion(action_gt.action_type, action_type_logits, mask=move_camera_weight)
    loss += action_type_loss

    #mask_tensor = get_one_way_mask_in_SL(action_gt.action_type, device)
    mask_tensor = SU.get_two_way_mask_in_SL(action_gt.action_type, action_pred, device, strict_comparsion=False)  # False

    # we now onsider delay loss
    delay_weight = 0.0
    if AHP.use_predict_step_mul:
        delay_weight = 1.0
    delay_loss = delay_weight * criterion(action_gt.delay, delay_logits)
    loss += delay_loss

    queue_loss = criterion(action_gt.queue, queue_logits, mask=mask_tensor[:, 2].reshape(-1))
    loss += queue_loss

    batch_size = action_gt.units.shape[0]
    select_size = action_gt.units.shape[1]
    units_size = action_gt.units.shape[-1]

    entity_nums = entity_nums
    print('entity_nums', entity_nums) if debug else None
    print('entity_nums.shape', entity_nums.shape) if debug else None

    # use extended select_size in SL for including EOF
    extended_select_size = select_size + 1

    units_mask = mask_tensor[:, 3]  # selected units is in the fourth position of units_mask
    units_mask = units_mask.unsqueeze(1).repeat(1, extended_select_size)
    units_mask = units_mask.reshape(-1)
    print('units_mask', units_mask) if debug else None
    print('units_mask.shape', units_mask.shape) if debug else None

    selected_mask = torch.arange(extended_select_size, device=device).float()
    selected_mask = selected_mask.repeat(batch_size, 1)

    # note, the select_units_num is actually gt_select_units_num here
    # we extend select_units_num by 1 to include the EFO
    selected_mask = selected_mask < (select_units_num + 1).unsqueeze(dim=1)
    selected_mask = selected_mask.reshape(-1)
    print('selected_mask', selected_mask) if debug else None
    print('selected_mask.shape', selected_mask.shape) if debug else None

    gt_units = action_gt.units.long()
    padding = torch.zeros(batch_size, 1, units_size, dtype=gt_units.dtype, device=gt_units.device)
    token = torch.tensor(AHP.max_entities - 1, dtype=padding.dtype, device=padding.device)
    padding[:, 0] = L.tensor_one_hot(token, units_size).reshape(-1)
    gt_units = torch.cat([gt_units, padding], dim=1)
    print('gt_units', gt_units) if debug else None
    print('gt_units.shape', gt_units.shape) if debug else None

    gt_units[torch.arange(batch_size), select_units_num] = L.tensor_one_hot(entity_nums, units_size).long()
    print('gt_units', gt_units) if debug else None
    print('gt_units.shape', gt_units.shape) if debug else None

    gt_units = gt_units.float()

    all_units_mask = units_mask * selected_mask  # * gt_units_mask

    # TODO: change to a proporate calculation of selected units
    # selected_units_weight = 10.
    # target_unit_weight = 1.
    # location_weight = 5.

    selected_units_weight = 1.
    target_unit_weight = 1.
    location_weight = 1.

    units_loss = selected_units_weight * criterion(gt_units.reshape(-1, units_size), units_logits.reshape(-1, units_size), 
                                                   mask=all_units_mask, debug=False, outlier_remove=True, entity_nums=entity_nums,
                                                   select_size=extended_select_size,
                                                   select_units_num=select_units_num + 1)
    loss += units_loss

    target_unit_loss = target_unit_weight * criterion(action_gt.target_unit.squeeze(-2), target_unit_logits.squeeze(-2), 
                                                      mask=mask_tensor[:, 4].reshape(-1), debug=False, outlier_remove=True, entity_nums=entity_nums)
    loss += target_unit_loss

    batch_size = action_gt.target_location.shape[0]

    target_location_loss = location_weight * criterion(action_gt.target_location.reshape(batch_size, -1),
                                                       target_location_logits.reshape(batch_size, -1), mask=mask_tensor[:, 5].reshape(-1))
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

    return loss, [action_type_loss.item(), delay_loss.item(), queue_loss.item(), units_loss.item(), target_unit_loss.item(), target_location_loss.item()]
