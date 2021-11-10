#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Some useful lib functions "

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pysc2.lib import actions

from alphastarmini.lib import config as C
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

from pysc2.lib.units import Neutral, Protoss, Terran, Zerg

__author__ = "Ruo-Ze Liu"

debug = False


def unit_tpye_to_unit_type_index(unit_type):
    ''' 
    transform unique unit type in SC2 to unit index in one hot represent in mAS.
    '''
    unit_tpye_name, race = get_unit_tpye_name_and_race(unit_type)
    print('unit_tpye_name, race:', unit_tpye_name, race) if debug else None   

    unit_type_index = get_unit_tpye_index(unit_tpye_name, race)
    print('unit_type_index:', unit_type_index) if debug else None

    return unit_type_index


def get_unit_tpye_name_and_race(unit_type):
    for race in (Neutral, Protoss, Terran, Zerg):
        try:
            return race(unit_type), race
        except ValueError:
            pass  # Wrong race.


def get_unit_tpye_index(unit_type_name, race):
    begin_index = 0
    if race == Neutral:
        begin_index = 0
    elif race == Protoss:
        begin_index = len(Neutral)
    elif race == Terran:
        begin_index = len(Neutral) + len(Protoss)
    elif race == Zerg:
        begin_index = len(Neutral) + len(Protoss) + len(Terran)

    for i, e in enumerate(list(race)):
        if e == unit_type_name:
            return i + begin_index
    return -1


def unpackbits_for_largenumber(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def calculate_unit_counts_bow(obs):
    unit_counts = obs["unit_counts"] 
    print('unit_counts:', unit_counts) if debug else None
    unit_counts_bow = torch.zeros(1, SFS.unit_counts_bow)
    for u_c in unit_counts:
        unit_type = u_c[0]
        unit_count = u_c[1]
        assert unit_type >= 0
        # the unit_count can not be negetive number
        assert unit_count >= 0

        # the unit_type should not be more than the SFS.unit_counts_bow
        # if it is, make it to be 0 now. (0 means nothing now)
        # the most impact one is ShieldBattery = 1910        
        # find a better way to do it: transform it to unit_type_index!
        unit_type_index = unit_tpye_to_unit_type_index(unit_type)
        if unit_type_index >= SFS.unit_counts_bow:
            unit_type_index = 0

        unit_counts_bow[0, unit_type_index] = unit_count
    return unit_counts_bow


def calculate_build_order(previous_bo, obs, next_obs):
    # calculate the build order
    ucb = calculate_unit_counts_bow(obs)
    next_ucb = calculate_unit_counts_bow(next_obs)
    diff = next_ucb - ucb

    # the probe, drone, and SCV are not counted in build order
    worker_type_list = [84, 104, 45] 
    # the pylon, drone, and supplypot are not counted in build order
    supply_type_list = [60, 106, 19] 
    diff[0, worker_type_list] = 0
    diff[0, supply_type_list] = 0

    diff_count = torch.sum(diff).item()
    print("diff between unit_counts_bow", diff_count) if debug else None
    if diff_count == 1.0:
        diff_numpy = diff.numpy()
        index_list = np.where(diff_numpy >= 1.0)
        print("index_list:", index_list) if debug else None
        index = index_list[1][0]
        if index not in worker_type_list and index not in supply_type_list:
            previous_bo.append(index)

    return previous_bo


def load_latest_model(model_type, path):
    models = list(filter(lambda x: model_type in x, os.listdir(path)))
    if len(models) == 0:
        print("No models are found!")
        return None

    models.sort()    
    model_path = os.path.join(path, models[-1])
    print("load model from {}".format(model_path))
    model = torch.load(model_path)

    return model


def load_the_model(model_path):
    model = torch.load(model_path)
    return model


def show_map_data_test(obs, map_width=128, show_original=True, show_resacel=True):
    use_small_map = False
    small_map_width = 32

    resize_type = np.uint8
    save_type = np.float16

    # note, in pysc2-1.2, obs["feature_minimap"]["height_map"] can be shown straight,
    # however, in pysc-3.0, that can not be show straight, must be transformed to numpy arrary firstly;
    height_map = np.array(obs["feature_minimap"]["height_map"])
    if show_original:
        imgplot = plt.imshow(height_map)
        plt.show()

    visibility_map = np.array(obs["feature_minimap"]["visibility_map"])
    if show_original:
        imgplot = plt.imshow(visibility_map)
        plt.show()

    creep = np.array(obs["feature_minimap"]["creep"])
    if show_original:
        imgplot = plt.imshow(creep)
        plt.show()

    player_relative = np.array(obs["feature_minimap"]["player_relative"])
    if show_original:
        imgplot = plt.imshow(player_relative)
        plt.show()

    # the below three maps are all zero, this may due to we connnect to a 3.16.1 version SC2,
    # may be different when we connect to 4.10 version SC2.
    alerts = np.array(obs["feature_minimap"]["alerts"])
    if show_original:
        imgplot = plt.imshow(alerts)
        plt.show()

    pathable = np.array(obs["feature_minimap"]["pathable"])
    if show_original:
        imgplot = plt.imshow(pathable)
        plt.show()

    buildable = np.array(obs["feature_minimap"]["buildable"])
    if show_original:
        imgplot = plt.imshow(buildable)
        plt.show()

    return None


def show_numpy_image(numpy_image):
    """
    """
    imgplot = plt.imshow(numpy_image)
    plt.show()
    return None


def np_one_hot(targets, nb_classes):
    """This is for numpy array
    https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    """

    print('nb_classes', nb_classes) if debug else None
    print('targets', targets) if debug else None

    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]

    return res.reshape(list(targets.shape) + [nb_classes])


def tensor_one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    cuda_check = labels.is_cuda
    if cuda_check:
        get_cuda_device = labels.get_device()

    y = torch.eye(num_classes)

    if cuda_check:
        y = y.to(get_cuda_device)

    return y[labels]


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    print('y', y) if debug else None
    cuda_check = y.is_cuda
    print('cuda_check', cuda_check) if debug else None

    if cuda_check:
        get_cuda_device = y.get_device()
        print('get_cuda_device', get_cuda_device) if debug else None

    y_tensor = y.data if isinstance(y, Variable) else y
    print('y_tensor', y_tensor) if debug else None
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    print('y_tensor', y_tensor) if debug else None

    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    if cuda_check:
        y_one_hot = y_one_hot.to(get_cuda_device)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def action_can_be_queued(action_type):
    """
    test the action_type whether can be queued

    Inputs: action_type, int
    Outputs: true or false
    """
    need_args = actions.RAW_FUNCTIONS[action_type].args
    result = False
    for arg in need_args:
        if arg.name == 'queued':
            result = True
            break
    return result 


def action_can_be_queued_mask(action_types):
    """
    test the action_type whether can be queued

    Inputs: action_types
    Outputs: mask
    """
    mask = torch.zeros_like(action_types)
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()
        print('i:', i, 'action_type_index:', action_type_index) if debug else None
        mask[i] = action_can_be_queued(action_type_index)

    return mask


def action_can_apply_to_entity_types(action_type):
    """
    find the entity_types which the action_type can be applied to
    TAG: TODO

    Inputs: action_type
    Outputs: mask of applied entity_types
    """
    mask = torch.ones(1, SCHP.max_unit_type)

    # note: this can be done when we know which action_type can apply
    # to certain unit_types which need strong prior knowledge, at present
    # I don't find there is such an api in pysc2
    # Thus now we only return a mask means all unit_types accept the action_type
    return mask


def action_can_apply_to_entity_types_mask(action_types):
    """
    find the entity_types which the action_type can be applied to

    Inputs: batch of action_type
    Outputs: mask
    """
    mask_list = []
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask = action_can_apply_to_entity_types(action_type_index)
        mask_list.append(mask)

    batch_mask = torch.cat(mask_list, dim=0)

    return batch_mask


def action_can_apply_to_entity(action_type):
    """
    find the entity_types which the action_type can be applied to
    TAG: TODO

    Inputs: action_type
    Outputs: the list of applied entity_types
    """
    if action_type % 2 == 0:
        return [0, 2, 4]
    else:
        return [1, 3, 7, 11]


def action_involve_selecting_units(action_type):
    """
    test the action_type whether involve selecting units

    Inputs: action_type
    Outputs: true or false
    """

    need_args = actions.RAW_FUNCTIONS[action_type].args
    result = False
    for arg in need_args:
        if arg.name == 'unit_tags':
            result = True
            break
    return result 


def action_involve_selecting_units_mask(action_types):
    """
    test the action_type whether involve selecting units

    Inputs: batch action_types
    Outputs: mask
    """

    mask = torch.zeros_like(action_types)
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_selecting_units(action_type_index)

    return mask


def action_involve_targeting_units(action_type):
    """
    test the action_type whether involve targeting units

    Inputs: action_type
    Outputs: true or false
    """
    need_args = actions.RAW_FUNCTIONS[action_type].args
    result = False
    for arg in need_args:
        if arg.name == 'target_unit_tag':
            result = True
            break
    return result 


def action_involve_targeting_units_mask(action_types):
    """
    test the action_type whether involve targeting units

    Inputs: batch action_types
    Outputs: mask
    """

    mask = torch.zeros_like(action_types)
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_targeting_units(action_type_index)

    return mask


def action_involve_targeting_location(action_type):
    """
    test the action_type whether involve targeting location
    Inputs: action_type
    Outputs: true or false
    """
    need_args = actions.RAW_FUNCTIONS[action_type].args
    result = False
    for arg in need_args:
        if arg.name == 'world':
            result = True
            break
    return result 


def action_involve_targeting_location_mask(action_types):
    """
    test the action_type whether involve targeting location

    Inputs: batch action_types
    Outputs: mask
    """

    mask = torch.zeros_like(action_types)
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_targeting_location(action_type_index)

    return mask


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def test():

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
