#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Some useful lib functions "

import os

from time import time, clock

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg

from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False
speed = True


def unit_tpye_to_unit_type_index(unit_type):
    ''' 
    transform unique unit type in SC2 to unit index in one hot represent in mAS.
    '''
    unit_type_index = get_unit_tpye_index_fast(unit_type)

    return unit_type_index


def get_unit_tpye_name_and_race(unit_type):
    for race in (Neutral, Protoss, Terran, Zerg):
        try:
            return race(unit_type), race
        except ValueError:
            pass  # Wrong race.


n = [item.value for item in Neutral]
p = [item.value for item in Protoss]
t = [item.value for item in Terran]
z = [item.value for item in Zerg]

all_list = n + p + t + z
all_dict = dict(zip(all_list, range(0, len(all_list))))


def get_unit_tpye_index_fast(item):
    index = all_dict[item]

    #index = all_list.index(item)

    return index


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


def calculate_unit_counts_bow_numpy(obs):
    unit_counts = obs["unit_counts"] 
    print('unit_counts:', unit_counts) if debug else None
    unit_counts_bow = np.zeros((1, SFS.unit_counts_bow))
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


def calculate_build_order_numpy(previous_bo, obs, next_obs):
    # calculate the build order
    ucb = calculate_unit_counts_bow_numpy(obs)
    next_ucb = calculate_unit_counts_bow_numpy(next_obs)
    diff = next_ucb - ucb

    # the probe, drone, and SCV are not counted in build order
    worker_type_list = [84, 104, 45] 
    # the pylon, drone, and supplypot are not counted in build order
    supply_type_list = [60, 106, 19] 
    diff[0, worker_type_list] = 0
    diff[0, supply_type_list] = 0

    diff_count = np.sum(diff).item()
    print("diff between unit_counts_bow", diff_count) if debug else None
    if diff_count == 1.0:
        index_list = np.where(diff >= 1.0)
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

    model = torch.load(model_path, map_location=torch.device(device))

    return model


def load_the_model(model_path):
    # we use new ways
    model = torch.load(model_path)
    return model


def initial_model_state_dict(model_type, path, model):
    models = list(filter(lambda x: model_type in x, os.listdir(path)))
    if len(models) == 0:
        print("No models are found!")
        return None

    models.sort()    
    model_path = os.path.join(path, models[-1])
    print("load model from {}".format(model_path))

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False) 

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


def np_one_hot_fast(targets, nb_classes):
    """

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
    mask = torch.zeros_like(action_types).bool()
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()
        print('i:', i, 'action_type_index:', action_type_index) if debug else None
        mask[i] = action_can_be_queued(action_type_index)

    return mask


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

    mask = torch.zeros_like(action_types).bool()
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_selecting_units(action_type_index)

    return mask


def action_involve_targeting_unit(action_type):
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


def action_involve_targeting_unit_mask(action_types):
    """
    test the action_type whether involve targeting units

    Inputs: batch action_types
    Outputs: mask
    """

    mask = torch.zeros_like(action_types).bool()
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_targeting_unit(action_type_index)

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

    mask = torch.zeros_like(action_types).bool()
    action_types = action_types.cpu().detach().numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_targeting_location(action_type_index)

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


def get_location_mask(mask):
    # mask shape [batch_size, output_map_size x output_map_size]
    map_name = P.map_name
    mask = mask.reshape(mask.shape[0], SCHP.world_size, SCHP.world_size)

    map_size = (AAIFP.raw_resolution, AAIFP.raw_resolution)

    mask[:, :map_size[1], :map_size[0]] = 1. 
    print('mask[0]', mask[0]) if debug else None 
    print('mask[0].sum()', mask[0].sum()) if debug else None

    mask = mask.reshape(mask.shape[0], -1)

    return mask


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    # from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def positional_encoding(max_position, embedding_size, add_batch_dim=False):
    # from https://github.com/metataro/sc2_imitation_learning in spatial_decoder in utils.py
    # has modification
    positions = np.arange(max_position)
    angle_rates = 1 / np.power(10000, (2 * (np.arange(embedding_size) // 2)) / np.float32(embedding_size))
    angle_rads = positions[:, np.newaxis] * angle_rates[np.newaxis, :]

    # note: "A::B" means from A every intervel of B, 0::5 is 0, 5, 10... ]
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    if add_batch_dim:
        # before: [max_position x embedding_size]
        # after: [1 x max_position x embedding_size]
        angle_rads = angle_rads[np.newaxis, ...]

    #return_tensor = torch.tensor(angle_rads, dtype=torch.float)

    return angle_rads


def test():

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
