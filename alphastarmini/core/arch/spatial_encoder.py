#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Spatial Encoder."

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from alphastarmini.core.arch.entity_encoder import EntityEncoder
from alphastarmini.core.arch.entity_encoder import Entity

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import MiniStar_Arch_Hyper_Parameters as MAHP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False


class SpatialEncoder(nn.Module):
    '''    
    Inputs: map, entity_embeddings
    Outputs:
        embedded_spatial - A 1D tensor of the embedded map
        map_skip - Tensors of the outputs of intermediate computations
    '''

    def __init__(self, n_resblocks=4, original_32=AHP.original_32,
                 original_64=AHP.original_64,
                 original_128=AHP.original_128,
                 original_256=AHP.original_256,
                 original_512=AHP.original_512):
        super().__init__()
        self.use_improved_one = True

        self.project_inplanes = AHP.map_channels
        self.project_inplanes_scatter = AHP.map_channels + AHP.original_32 * AHP.scatter_channels
        self.project = nn.Conv2d(self.project_inplanes, original_32, kernel_size=1, stride=1,
                                 padding=0, bias=True)
        self.project_scatter = nn.Conv2d(self.project_inplanes_scatter, original_32, kernel_size=1, stride=1,
                                         padding=0, bias=True)
        # ds means downsampling
        self.ds_1 = nn.Conv2d(original_32, original_64, kernel_size=4, stride=2,
                              padding=1, bias=True)
        self.ds_2 = nn.Conv2d(original_64, original_128, kernel_size=4, stride=2,
                              padding=1, bias=True)
        self.ds_3 = nn.Conv2d(original_128, original_128, kernel_size=4, stride=2,
                              padding=1, bias=True)
        self.resblock_stack = nn.ModuleList([
            ResBlock(inplanes=original_128, planes=original_128, stride=1)
            for _ in range(n_resblocks)])

        if AHP == MAHP:
            # note: in mAS, we replace 128x128 to 64x64, and the result 16x16 also to 8x8
            self.fc = nn.Linear(8 * 8 * original_128, original_256)
        else:
            self.fc = nn.Linear(16 * 16 * original_128, original_256)  # position-wise

        self.conv1 = nn.Conv1d(original_256, original_32, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.map_width = AHP.minimap_size

    @classmethod
    def preprocess_numpy(cls, obs, entity_pos_list=None):
        map_data = cls.get_map_data(obs, entity_pos_list=entity_pos_list)
        return map_data

    def scatter(self, entity_embeddings, entity_x_y):
        # `entity_embeddings` are embedded through a size 32 1D convolution, followed by a ReLU,
        # [batch_size x entity_size x embedding_size]
        print("entity_embeddings.shape:", entity_embeddings.shape) if debug else None

        # [batch_size x entity_size x reduced_embedding_size(e.g. 16)]
        reduced_entity_embeddings = F.relu(self.conv1(entity_embeddings.transpose(1, 2))).transpose(1, 2)
        print("reduced_entity_embeddings.shape:", reduced_entity_embeddings.shape) if debug else None

        # then scattered into a map layer so that the size 32 vector at a specific 
        # location corresponds to the units placed there.

        def bits2value(bits):
            # change from the bits to dec values.
            l = len(bits)
            v = 0
            g = 1
            for i in range(l - 1, -1, -1):               
                v += bits[i] * g
                g *= 2
            return v

        # shape [batch_size x entity_size x embedding_size]
        batch_size = reduced_entity_embeddings.shape[0]
        entity_size = reduced_entity_embeddings.shape[1]

        device = next(self.parameters()).device
        scatter_map = torch.zeros(batch_size, AHP.original_32, self.map_width, self.map_width, device=device)
        print("scatter_map.shape:", scatter_map.shape) if debug else None

        # This operation consumes much time.
        # Do we have a good efficient one?
        for i in range(batch_size):
            for j in range(entity_size):
                # can not be masked entity
                print('entity_x_y[i, j, 0]', entity_x_y[i, j, 0]) if debug else None
                if entity_x_y[i, j, 0] != EntityEncoder.bias_value:
                    x = entity_x_y[i, j, :8]
                    y = entity_x_y[i, j, 8:]
                    x = bits2value(x)
                    y = bits2value(y)
                    print('x', x) if debug else None
                    print('y', y) if debug else None

                    # Note: the x and y from obs["raw_data"] is actually minimap position!
                    # because the pysc2 has transformed the world pos to minimap pos in the raw data 
                    # However, the minimap size is not set by feature_dimensions.minimap but
                    # the raw_resolution! If the raw_resolution is none, it will use the map_size,
                    # making the x and y beyond 64 !
                    if P.map_name == 'AbyssalReef':
                        x = int(x / 4)
                        y = int(y / 4)
                    scatter_map[i, :, y, x] = reduced_entity_embeddings[i, j, :]

        return scatter_map   

    def forward(self, x, entity_embeddings=None, entity_x_y=None):
        # 
        # scatter_map may cause a NaN bug in SL training, now we don't use it
        if entity_embeddings is not None and entity_x_y is not None:
            scatter_map = self.scatter(entity_embeddings, entity_x_y)
            x = torch.cat([scatter_map, x], dim=1)

        # After preprocessing, the planes are concatenated, projected to 32 channels 
        # by a 2D convolution with kernel size 1, passed through a ReLU
        if AHP.scatter_channels:
            x = F.relu(self.project_scatter(x))
        else:
            x = F.relu(self.project(x))

        # then downsampled from 128x128 to 16x16 through 3 2D convolutions and ReLUs 
        # with channel size 64, 128, and 128 respectively. 
        # The kernel size for those 3 downsampling convolutions is 4, and the stride is 2.
        # note: in mAS, we replace 128x128 to 64x64, and the result 16x16 also to 8x8
        # note: here we should add a relu after each conv2d
        x = F.relu(self.ds_1(x))
        x = F.relu(self.ds_2(x))
        x = F.relu(self.ds_3(x))

        if not self.use_improved_one:
            # 4 ResBlocks with 128 channels and kernel size 3 and applied to the downsampled map, 
            # with the skip connections placed into `map_skip`.
            map_skip = x
            for resblock in self.resblock_stack:
                x = resblock(x)

                # note if we add the follow line, it will output "can not comput gradient error"
                # map_skip += x
                # so we try to change to the follow line, which will not make a in-place operation
                map_skip = map_skip + x
        else:
            # Referenced mostly from "sc2_imitation_learning" project in spatial_decoder
            map_skip = [x]
            for resblock in self.resblock_stack:
                x = resblock(x)
                map_skip.append(x)

        # Compared to AS, we a relu, referred from "sc2_imitation_learning"
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        # The ResBlock output is embedded into a 1D tensor of size 256 by a linear layer 
        # and a ReLU, which becomes `embedded_spatial`.
        x = self.fc(x)
        embedded_spatial = F.relu(x)

        return map_skip, embedded_spatial

    @classmethod
    def get_map_data(cls, obs, entity_pos_list=None, map_width=AHP.minimap_size, verbose=False):
        '''
        default map_width is 64
        '''
        feature_minimap = obs["feature_minimap"] if "feature_minimap" in obs else obs
        save_type = np.float32

        # we consider the most 4 entities in the same position
        scatter_map = np.zeros((1, map_width, map_width, 4), dtype=save_type)
        if entity_pos_list is not None:
            # make the scatter_map has the index of entity in the entity's position (matrix format)      
            for i, (x, y) in enumerate(entity_pos_list):
                for j in range(4):
                    if not scatter_map[0, y, x, j]:
                        scatter_map[0, y, x, j] = i
                        break

            if False:  # a simple test
                for j in range(5):
                    r = random.randint(0, 10)
                    if r < len(entity_pos_list):
                        (x, y) = entity_pos_list[r]
                        print('r', r)
                        print('x', x)
                        print('y', y)
                        print('scatter_map[0, y, x]', scatter_map[0, y, x])
                        assert r in scatter_map[0, y, x].tolist()

        # A: camera: One-hot with maximum 2 of whether a location is within the camera, this refers to mimimap
        camera_map = L.np_one_hot(feature_minimap["camera"].reshape(-1, map_width, map_width), 2).astype(save_type)
        print('camera_map:', camera_map) if verbose else None
        print('camera_map.shape:', camera_map.shape) if verbose else None

        # A: height_map: Float of (height_map / 255.0)
        height_map = np.expand_dims(feature_minimap["height_map"].reshape(-1, map_width, map_width) / 255.0, -1).astype(save_type)
        print('height_map:', height_map) if verbose else None
        print('height_map.shape:', height_map.shape) if verbose else None

        # A: visibility: One-hot with maximum 4
        visibility = L.np_one_hot(feature_minimap["visibility_map"].reshape(-1, map_width, map_width), 4).astype(save_type)
        print('visibility:', visibility) if verbose else None
        print('visibility.shape:', visibility.shape) if verbose else None

        # A: creep: One-hot with maximum 2
        creep = L.np_one_hot(feature_minimap["creep"].reshape(-1, map_width, map_width), 2).astype(save_type)
        print('creep:', creep) if verbose else None

        # A: entity_owners: One-hot with maximum 5
        entity_owners = L.np_one_hot(feature_minimap["player_relative"].reshape(-1, map_width, map_width), 5).astype(save_type)
        print('entity_owners:', entity_owners) if verbose else None

        # the bottom 3 maps are missed in pysc1.2 and pysc2.0
        # however, the 3 maps can be found on s2clientprotocol/spatial.proto
        # actually, the 3 maps can be found on pysc3.0

        # A: alerts: One-hot with maximum 2
        alerts = L.np_one_hot(feature_minimap["alerts"].reshape(-1, map_width, map_width), 2).astype(save_type)
        print('alerts:', alerts) if verbose else None

        # A: pathable: One-hot with maximum 2
        pathable = L.np_one_hot(feature_minimap["pathable"].reshape(-1, map_width, map_width), 2).astype(save_type)
        print('pathable:', pathable) if verbose else None

        # A: buildable: One-hot with maximum 2
        buildable = L.np_one_hot(feature_minimap["buildable"].reshape(-1, map_width, map_width), 2).astype(save_type)
        print('buildable:', buildable) if verbose else None

        out_channels = 1 + 2 + 1 + 4 + 2 + 5 + 2 + 2 + 2

        map_data = np.concatenate([scatter_map, camera_map, height_map, visibility, creep, entity_owners, 
                                   alerts, pathable, buildable], axis=-1)  # the channel is at the last axis

        # NWHC to NCHW
        map_data = np.transpose(map_data, [0, 3, 1, 2])
        print('map_data.shape:', map_data.shape) if verbose else None

        return map_data


class ResBlock(nn.Module):
    # without batchnorm
    # referenced from https://github.com/metataro/sc2_imitation_learning in conv.py
    # also referenced from https://github.com/liuruoze/Thought-SC2 in ops.py

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + z


class ResBlock_BN(nn.Module):

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResBlockImproved(nn.Module):

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlockImproved, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    '''From paper Identity Mappings in Deep Residual Networks'''

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x = x + residual
        return x


class ResBlock1D(nn.Module):

    def __init__(self, inplanes, planes, seq_len, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.ln1 = nn.LayerNorm([planes, seq_len])
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.ln2 = nn.LayerNorm([planes, seq_len])

    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(x))
        x = self.conv1(x)
        x = F.relu(self.ln2(x))
        x = self.conv2(x)
        x = x + residual
        return x


def test():
    spatial_encoder = SpatialEncoder()
    batch_size = 2
    # dummy map list
    map_list = []
    map_data_1 = torch.zeros(batch_size, 1, AHP.minimap_size, AHP.minimap_size)
    map_data_1_one_hot = L.to_one_hot(map_data_1, 2)
    print('map_data_1_one_hot.shape:', map_data_1_one_hot.shape) if debug else None

    map_list.append(map_data_1)
    map_data_2 = torch.zeros(batch_size, 17, AHP.minimap_size, AHP.minimap_size)
    map_list.append(map_data_2)
    map_data = torch.cat(map_list, dim=1)

    map_skip, embedded_spatial = spatial_encoder.forward(map_data)

    print('map_skip:', map_skip) if debug else None
    print('embedded_spatial:', embedded_spatial) if debug else None

    print('map_skip.shape:', map_skip.shape) if debug else None
    print('embedded_spatial.shape:', embedded_spatial.shape) if debug else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()
