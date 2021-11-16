#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Help code for sl training "

import traceback

import numpy as np

import torch
import torch.nn as nn

from pysc2.lib.actions import RAW_FUNCTIONS

from alphastarmini.core.arch.agent import Agent

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

from alphastarmini.lib.hyper_parameters import Label_Size as LS


debug = False


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


def get_two_way_mask_in_SL(action_type_gt, action_pred, device):
    # consider the ground truth and the predicted
    ground_truth_raw_action_id = torch.nonzero(action_type_gt, as_tuple=True)[-1]
    mask_list = [] 

    print('ground_truth_raw_action_id.shape', ground_truth_raw_action_id.shape) if debug else None

    for raw_action_id in ground_truth_raw_action_id:
        mask_list.append(get_mask_by_raw_action_id(raw_action_id.item()))

    mask_tensor = torch.tensor(mask_list)

    mask_list_2 = [] 

    print('action_pred.shape', action_pred.shape) if debug else None  
    for action_id in action_pred:
        mask_list_2.append(get_mask_by_raw_action_id(action_id.item()))

    mask_tensor_2 = torch.tensor(mask_list_2)

    mask_tensor_return = mask_tensor * mask_tensor_2
    mask_tensor_return = mask_tensor_return.to(device)

    return mask_tensor_return


def get_move_camera_weight_in_SL(action_type_gt, action_pred, device):
    # consider the ground truth and the predicted
    ground_truth_raw_action_id = torch.nonzero(action_type_gt, as_tuple=True)[-1]
    mask_list = [] 

    MOVE_CAMERA_ID = 168
    # Note, in SC2, move_camera resides as 50% actions in all actions
    # we assume every other action has the same happenning rate, so 
    # assume move_camera weight is 1.
    # the non_move_camera weight is MAX_ACTIONS /2. / alpha
    # alpha set to 10
    MOVE_CAMERA_WEIGHT = 1.  # 1. / LS.action_type_encoding * 2.
    alpha = 40.
    NON_MOVE_CAMERA_WEIGHT = LS.action_type_encoding / 2. / alpha

    for raw_action_id in ground_truth_raw_action_id:
        if raw_action_id.item() == MOVE_CAMERA_ID:
            mask_list.append([MOVE_CAMERA_WEIGHT])
        else:
            mask_list.append([NON_MOVE_CAMERA_WEIGHT])
    mask_tensor = torch.tensor(mask_list)

    # also use predict value to weight
    # not used first
    if False:
        mask_list_2 = [] 
        for action_id in action_pred:
            if action_id.item() == MOVE_CAMERA_ID:
                mask_list_2.append([MOVE_CAMERA_WEIGHT])
            else:
                mask_list_2.append([NON_MOVE_CAMERA_WEIGHT])
        mask_tensor_2 = torch.tensor(mask_list_2)
        mask_tensor = mask_tensor * mask_tensor_2

    mask_tensor = mask_tensor.to(device)

    return mask_tensor


def get_accuracy(ground_truth, predict, device):
    accuracy = 0.

    ground_truth_new = torch.nonzero(ground_truth, as_tuple=True)[-1]
    ground_truth_new = ground_truth_new.to(device)
    print('ground_truth', ground_truth_new) if debug else None

    predict_new = predict.reshape(-1)
    print('predict_new', predict_new) if debug else None

    # calculate how many move_camera? the id is 168 in raw_action
    MOVE_CAMERA_ID = 168
    #camera_num_action_type = torch.sum(MOVE_CAMERA_ID == ground_truth_new)
    move_camera_index = (ground_truth_new == MOVE_CAMERA_ID).nonzero(as_tuple=True)[0]
    non_camera_index = (ground_truth_new != MOVE_CAMERA_ID).nonzero(as_tuple=True)[0]

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

    return [right_num, all_num, camera_right_num, camera_all_num, non_camera_right_num, non_camera_all_num]


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
