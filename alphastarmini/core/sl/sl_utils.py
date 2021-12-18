#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Help code for sl training "

import traceback

import numpy as np

import torch
import torch.nn as nn

from pysc2.lib.actions import RAW_FUNCTIONS as F

from alphastarmini.core.arch.agent import Agent

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

from alphastarmini.lib import utils as L

from alphastarmini.lib.sc2 import raw_actions_mapping_protoss as RAMP

debug = False

Smart_unit_weight_S64 = 0.5
Smart_pt_weight_S64 = 0.01
Important_weight_S64 = 2
Attack_pt_weight_S64 = 2
Other_weight_S64 = 1

Smart_unit_weight_AR = 0.5
Smart_pt_weight_AR = 0.1
Important_weight_AR = 2
Attack_pt_weight_AR = 2
Other_weight_AR = 1


def obs2feature(obs):
    s = Agent.get_state_and_action_from_pickle(obs)
    feature = Feature.state2feature(s)
    print("feature:", feature) if debug else None
    print("feature.shape:", feature.shape) if debug else None

    print("begin a:") if debug else None
    func_call = obs['func_call']
    action = Agent.func_call_to_action(func_call).toTenser()
    #tag_list = agent.get_tag_list(obs)
    print('action.get_shape:', action.get_shape()) if debug else None

    logits = action.toLogits()
    print('logits.shape:', logits) if debug else None
    label = Label.action2label(logits)
    print("label:", label) if debug else None
    print("label.shape:", label.shape) if debug else None
    return feature, label


def obs2feature_numpy(obs):
    s = Agent.get_state_and_action_from_pickle_numpy(obs)
    feature = Feature.state2feature_numpy(s)
    print("feature:", feature) if debug else None
    print("feature.shape:", feature.shape) if debug else None

    print("begin a:") if debug else None
    func_call = obs['func_call']
    action = Agent.func_call_to_action(func_call).toArray()
    #tag_list = agent.get_tag_list(obs)
    print('action.get_shape:', action.get_shape()) if debug else None

    logits = action.toLogits_numpy()
    print('logits.shape:', logits) if debug else None
    label = Label.action2label_numpy(logits)
    print("label:", label) if debug else None
    print("label.shape:", label.shape) if debug else None
    return feature, label


def obsToTensor(obs, final_index_list, seq_len):
    feature_list = []
    label_list = []
    for value in obs:
        feature, label = obs2feature(value)
        feature_list.append(feature)
        label_list.append(label)

    features = torch.cat(feature_list, dim=0)
    print("features.shape:", features.shape) if debug else None

    labels = torch.cat(label_list, dim=0)
    print("labels.shape:", labels.shape) if debug else None

    is_final = torch.zeros([features.shape[0], 1])

    # consider is_final
    print('begin', index) if debug else None
    print('end', index + seq_len) if debug else None
    for j in final_index_list:
        print('j', j) if debug else None
        if j >= index and j < index + seq_len:
            if debug:
                print('in it!') 
                print('begin', index)
                print('end', index + seq_len)
                print('j', j)
            is_final[j - index, 0] = 1
        else:
            pass

    one_traj = torch.cat([features, labels, is_final], dim=1)
    print("one_traj.shape:", one_traj.shape) if debug else None

    return one_traj


def get_mask_by_raw_action_id(raw_action_id):
    need_args = F[raw_action_id].args

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


def get_one_way_mask_in_SL(action_type_gt, device):
    # only consider the ground truth

    # the action_type_gt is one_hot embedding
    ground_truth_raw_action_id = torch.nonzero(action_type_gt, as_tuple=True)[-1]
    mask_list = [] 

    for raw_action_id in ground_truth_raw_action_id:
        mask_list.append(get_mask_by_raw_action_id(raw_action_id.item()))

    mask_tensor = torch.tensor(mask_list)
    mask_tensor = mask_tensor.to(device)

    return mask_tensor


def get_two_way_mask_in_SL(action_type_gt, action_pred, device, strict_comparsion=True):
    # consider the ground truth and the predicted
    ground_truth_raw_action_id = torch.nonzero(action_type_gt, as_tuple=True)[-1]
    action_pred = action_pred.reshape(-1)

    mask_list = [] 
    mask_list_2 = [] 

    print('ground_truth.shape', ground_truth_raw_action_id.shape) if debug else None
    print('ground_truth', ground_truth_raw_action_id) if debug else None
    print('action_pred.shape', action_pred.shape) if debug else None 
    print('action_pred', action_pred) if debug else None 

    for raw_action_id, action_id in zip(ground_truth_raw_action_id, action_pred):
        gt_aid = raw_action_id.item()
        predict_aid = action_id.item()

        mask_raw = get_mask_by_raw_action_id(gt_aid)
        mask_predict = get_mask_by_raw_action_id(predict_aid)

        mask_weight = 1

        Smart_pt_id = F.Smart_pt.id
        Smart_unit_id = F.Smart_unit.id
        Attack_pt_id = F.Attack_pt.id

        if SCHP.map_name == 'Simple64':
            if gt_aid == Smart_unit_id:
                mask_weight = Smart_unit_weight_S64
            elif gt_aid == Smart_pt_id:
                mask_weight = Smart_pt_weight_S64
            elif gt_aid == Attack_pt_id:
                mask_weight = Attack_pt_weight_S64
            elif gt_aid in RAMP.SMALL_LIST:
                mask_weight = Important_weight_S64
            else:
                mask_weight = Other_weight_S64
        elif SCHP.map_name == 'AbyssalReef':
            if gt_aid == Smart_unit_id:
                mask_weight = Smart_unit_weight_AR
            elif gt_aid == Smart_pt_id:
                mask_weight = Smart_pt_weight_AR
            elif gt_aid == Attack_pt_id:
                mask_weight = Attack_pt_weight_AR
            elif gt_aid in RAMP.MEDIUM_LIST:
                mask_weight = Important_weight_AR
            else:
                mask_weight = Other_weight_AR
        else:
            raise NotImplementedError('Unsupported map name!')

        if strict_comparsion:
            if raw_action_id.item() != action_id.item():
                zero_mask = [1, 1, 0, 0, 0, 0]
                mask_raw = zero_mask
                mask_predict = zero_mask            

        mask_raw = np.array(mask_raw)
        mask_predict = np.array(mask_predict)

        mask_raw = mask_raw * mask_weight

        mask_list.append(mask_raw)
        mask_list_2.append(mask_predict)

    mask_tensor = torch.tensor(mask_list)
    mask_tensor_2 = torch.tensor(mask_list_2)

    print('mask_tensor', mask_tensor) if debug else None 
    print('mask_tensor_2', mask_tensor_2) if debug else None 

    mask_tensor_return = mask_tensor  # * mask_tensor_2
    print('mask_tensor_return', mask_tensor_return) if debug else None 

    mask_tensor_return = mask_tensor_return.to(device)

    return mask_tensor_return


def get_move_camera_weight_in_SL(action_type_gt, action_pred, device, 
                                 decrease_smart_opertaion=False, only_consider_small=False):
    # consider the ground truth and the predicted
    ground_truth_raw_action_id = torch.nonzero(action_type_gt, as_tuple=True)[-1]
    mask_list = [] 

    MOVE_CAMERA_ID = F.raw_move_camera.id
    Smart_pt_id = F.Smart_pt.id
    Smart_unit_id = F.Smart_unit.id
    Attack_pt_id = F.Attack_pt.id

    print('ground_truth_raw_action_id', ground_truth_raw_action_id) if debug else None

    for raw_action_id in ground_truth_raw_action_id:
        aid = raw_action_id.item()

        if SCHP.map_name == 'Simple64':
            if aid == Smart_unit_id:
                mask_list.append([Smart_unit_weight_S64])
            elif aid == Smart_pt_id:
                mask_list.append([Smart_pt_weight_S64])
            elif aid == Attack_pt_id:
                mask_list.append([Attack_pt_weight_S64])
            elif aid in RAMP.SMALL_LIST:
                mask_list.append([Important_weight_S64])
            else:
                mask_list.append([Other_weight_S64])
        elif SCHP.map_name == 'AbyssalReef':
            if aid == Smart_unit_id:
                mask_list.append([Smart_unit_weight_AR])
            elif aid == Smart_pt_id:
                mask_list.append([Smart_pt_weight_AR])
            elif aid == Attack_pt_id:
                mask_list.append([Attack_pt_weight_AR])
            elif aid in RAMP.MEDIUM_LIST:
                mask_list.append([Important_weight_AR])
            else:
                mask_list.append([Other_weight_AR])
        else:
            raise NotImplementedError('Unsupported map name!')

    mask_tensor = torch.tensor(mask_list)

    print('mask_tensor', mask_tensor) if debug else None

    mask_tensor = mask_tensor.to(device)

    return mask_tensor


def get_selected_units_accuracy(ground_truth, predict, gt_action_type, pred_action_type,
                                select_units_num, action_equal_mask, 
                                device, unit_types_one=None, entity_nums=None,
                                strict_comparsion=True, use_strict_order=True):
    all_num, correct_num, gt_num, pred_num, type_correct_num = 0, 0, 1, 0, 0

    gt_action_type = torch.nonzero(gt_action_type.long(), as_tuple=True)[-1].unsqueeze(dim=1)
    print('gt_action_type.shape', gt_action_type.shape) if debug else None

    if strict_comparsion:
        action_equal_index = action_equal_mask.nonzero(as_tuple=True)[0]
        ground_truth = ground_truth[action_equal_index]
        predict = predict[action_equal_index]
        if unit_types_one is not None:
            unit_types_one = unit_types_one[action_equal_index]
        entity_nums = entity_nums[action_equal_index]

        gt_action_type = gt_action_type[action_equal_index]
        pred_action_type = pred_action_type[action_equal_index]

    units_nums_equal = 0
    batch_size = 0
    if ground_truth.shape[0] > 0:  
        batch_size = ground_truth.shape[0]
        NONE_INDEX = AHP.max_entities - 1

        for i in range(batch_size):
            ground_truth_sample = ground_truth[i]
            ground_truth_sample = torch.nonzero(ground_truth_sample, as_tuple=True)[-1]
            ground_truth_sample = ground_truth_sample.cpu().detach().numpy().tolist()
            print('ground_truth_sample', ground_truth_sample) if debug else None

            predict_sample = predict[i].reshape(-1)
            print('predict_sample units', predict_sample) if debug else None

            if unit_types_one is not None:
                unit_types_one_sample = unit_types_one[i].reshape(-1)
                print('unit_types_one_sample', unit_types_one_sample) if debug else None

            entity_num = entity_nums[i].reshape(-1)

            ground_truth_units_num_i = 0
            for gt in ground_truth_sample:
                if gt != NONE_INDEX and gt != entity_num.item():  # the last index is the None index
                    ground_truth_units_num_i += 1
            print('ground_truth_units_num_i', ground_truth_units_num_i) if debug else None         

            select_units_num_i = select_units_num[i].item()
            print('select_units_num_i', select_units_num_i) if debug else None

            if ground_truth_units_num_i == select_units_num_i:
                units_nums_equal += 1

            gt_action_type_sample = gt_action_type[i].item()        
            pred_action_type_sample = pred_action_type[i].item()

            for j in range(select_units_num_i):
                pred = predict_sample[j].item()
                gt = ground_truth_sample[j]
                if gt != NONE_INDEX:  # the last index is the None index
                    gt_num += 1

                if unit_types_one_sample is not None:
                    pred_type = unit_types_one_sample[pred].item()
                    gt_type = unit_types_one_sample[gt].item()

                    if pred_type == gt_type:
                        type_correct_num += 1

                if use_strict_order:
                    if pred == gt and pred != NONE_INDEX:
                        correct_num += 1
                else:
                    if pred in ground_truth_new and pred != NONE_INDEX:
                        correct_num += 1
                pred_num += 1

            all_num += AHP.max_selected

    ret = {}
    ret['correct_num'] = correct_num
    ret['gt_num'] = gt_num
    ret['pred_num'] = pred_num
    ret['all_num'] = all_num
    ret['units_nums_equal'] = units_nums_equal
    ret['batch_size'] = batch_size

    print('get_selected_units_accuracy', [correct_num, gt_num, type_correct_num, pred_num, units_nums_equal, batch_size]) if debug else None

    return [correct_num, gt_num, type_correct_num, pred_num, units_nums_equal, batch_size]


def get_target_unit_accuracy(ground_truth, predict, action_equal_mask, device, 
                             strict_comparsion=True, remove_none=True):
    right_num, all_num = 0, 0

    if strict_comparsion:
        action_equal_index = action_equal_mask.nonzero(as_tuple=True)[0]
        ground_truth = ground_truth[action_equal_index]
        predict = predict[action_equal_index]

    if ground_truth.shape[0] > 0:  
        print('ground_truth target_unit', ground_truth) if debug else None

        ground_truth_new = torch.nonzero(ground_truth, as_tuple=True)[-1]
        ground_truth_new = ground_truth_new.to(device)
        print('ground_truth_new target_unit', ground_truth_new) if debug else None

        predict_new = predict.reshape(-1)
        print('predict_new target_unit', predict_new) if debug else None

        NONE_ID = AHP.max_entities - 1
        if remove_none:
            effect_index = (ground_truth_new != NONE_ID).nonzero(as_tuple=True)[0]
            ground_truth_new = ground_truth_new[effect_index]
            predict_new = predict_new[effect_index]

        right_num, all_num = get_right_and_all_num(ground_truth_new, predict_new)

    print('get_target_unit_accuracy', [right_num, all_num]) if debug else None

    return [right_num, all_num]


def get_location_accuracy(ground_truth, predict, action_equal_mask, device, strict_comparsion=True):
    all_nums = ground_truth.shape[0]

    effect_nums = 0  # when the location argument applied both in ground_truth and predict
    correct_nums = 0
    distance_loss = 0.

    if strict_comparsion:
        action_equal_index = action_equal_mask.nonzero(as_tuple=True)[0]
        ground_truth = ground_truth[action_equal_index]
        predict = predict[action_equal_index]

    if ground_truth.shape[0] > 0:    

        ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)
        ground_truth_new = torch.nonzero(ground_truth, as_tuple=True)[-1]
        ground_truth_new = ground_truth_new.to(device)
        print('ground_truth location', ground_truth_new) if debug else None

        output_map_size = SCHP.world_size

        for i, idx in enumerate(ground_truth_new):
            row_number = idx // output_map_size
            col_number = idx - output_map_size * row_number

            gt_location_y = row_number
            gt_location_x = col_number
            print("gt_location_y, gt_location_x", gt_location_y, gt_location_x) if debug else None

            [predict_x, predict_y] = predict[i] 
            print("predict_x, predict_y", predict_x, predict_y) if debug else None

            x_diff_square = (predict_x.item() - gt_location_x.item()) ** 2
            y_diff_square = (predict_y.item() - gt_location_y.item()) ** 2

            print('x_diff_square', x_diff_square) if debug else None
            print('y_diff_square', y_diff_square) if debug else None

            # pos(output_map_size-1, output_map_size-1) isconsidered a flag meaning this arugment is not applied for this action;
            # e.g., we hardly will choose or see a point of pos(output_map_size-1, output_map_size-1)
            if not (gt_location_y.item() == output_map_size - 1 and gt_location_x.item() == output_map_size - 1):  # the last index is the None index
                if not (predict_x.item() == 0 and predict_y.item() == 0):
                    effect_nums += 1

                    diff_square = x_diff_square + y_diff_square
                    distance_loss += diff_square

                    if diff_square == 0:
                        correct_nums += 1

    print('get_location_accuracy', [correct_nums, effect_nums, all_nums, distance_loss]) if debug else None

    return [correct_nums, effect_nums, all_nums, distance_loss]


def get_accuracy(ground_truth, predict, device, return_important=False):
    accuracy = 0.

    ground_truth_new = torch.nonzero(ground_truth, as_tuple=True)[-1]
    ground_truth_new = ground_truth_new.to(device)
    print('ground_truth action_type', ground_truth_new) if debug else None

    predict_new = predict.reshape(-1)
    print('predict_new', predict_new) if debug else None  

    # shape: [batch_size]
    action_equal_mask = (ground_truth_new == predict_new)

    # calculate how many move_camera? the id is 168 in raw_action
    MOVE_CAMERA_ID = 168
    #camera_num_action_type = torch.sum(MOVE_CAMERA_ID == ground_truth_new)
    move_camera_index = (ground_truth_new == MOVE_CAMERA_ID).nonzero(as_tuple=True)[0]
    non_camera_index = (ground_truth_new != MOVE_CAMERA_ID).nonzero(as_tuple=True)[0]

    short_important_list = []
    for j in RAMP.SMALL_MAPPING.keys():
        aid = F[j].id.value
        print('aid', aid) if debug else None
        short_index = (ground_truth_new == aid).nonzero(as_tuple=True)[0]
        print('short_index', short_index) if debug else None
        short_important_list.append(short_index)

    short_important_index = torch.cat(short_important_list)
    print('short_important_index', short_important_index) if debug else None  

    print('move_camera_index', move_camera_index) if debug else None    
    print('non_camera_index', non_camera_index) if debug else None  

    print('for any type action') if debug else None
    right_num, all_num = get_right_and_all_num(ground_truth_new, predict_new)

    print('for move_camera action') if debug else None
    camera_ground_truth_new = ground_truth_new[move_camera_index]
    camera_predict_new = predict_new[move_camera_index]
    camera_right_num, camera_all_num = get_right_and_all_num(camera_ground_truth_new, camera_predict_new)

    print('for non-camera action') if debug else None
    non_camera_ground_truth_new = ground_truth_new[non_camera_index]
    non_camera_predict_new = predict_new[non_camera_index]
    non_camera_right_num, non_camera_all_num = get_right_and_all_num(non_camera_ground_truth_new, non_camera_predict_new)

    print('for short-important action') if debug else None
    short_important_ground_truth_new = ground_truth_new[short_important_index]
    short_important_predict_new = predict_new[short_important_index]
    short_important_right_num, short_important_all_num = get_right_and_all_num(short_important_ground_truth_new, short_important_predict_new)

    acc_list = [right_num, all_num, camera_right_num, camera_all_num, non_camera_right_num, 
                non_camera_all_num, short_important_right_num, short_important_all_num]

    return acc_list, action_equal_mask


def get_right_and_all_num(gt, pred):
    acc_num_action_type = torch.sum(pred == gt)
    print('acc_num_action_type', acc_num_action_type) if debug else None

    right_num = acc_num_action_type.item()
    print('right_num', right_num) if debug else None

    all_num = gt.shape[0]
    print('all_num', all_num) if debug else None

    accuracy = right_num / (all_num + 1e-9)
    print('accuracy', accuracy) if debug else None

    return right_num, all_num 
