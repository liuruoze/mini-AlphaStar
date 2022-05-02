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
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

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
    scatter_volume = 4

    def __init__(self, n_resblocks=4, original_32=AHP.original_32,
                 original_64=AHP.original_64,
                 original_128=AHP.original_128,
                 original_256=AHP.original_256,
                 original_512=AHP.original_512):
        super().__init__()
        self.use_improved_one = True

        self.project_inplanes = AHP.map_channels
        self.project_inplanes_scatter = AHP.map_channels + AHP.original_32 * AHP.scatter_channels - self.scatter_volume
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

    def scatter(self, scatter_index, entity_embeddings, same_pos_handle='add'):
        # Note, though this command is called scatter, it actually use the gather function in PyTorch,
        # while 'gather' is an opposite operation of 'scatter'

        # `entity_embeddings` are embedded through a size 32 1D convolution, followed by a ReLU,
        # [batch_size x entity_size x embedding_size]

        # [batch_size x entity_size x reduced_embedding_size(e.g. 16)]
        reduced_entity_embeddings = F.relu(self.conv1(entity_embeddings.transpose(1, 2))).transpose(1, 2)
        del entity_embeddings

        # then scattered into a map layer so that the size 32 vector at a specific 
        # location corresponds to the units placed there.
        # shape [batch_size x entity_size x embedding_size]
        batch_size = reduced_entity_embeddings.shape[0]
        entity_size = reduced_entity_embeddings.shape[1]
        embed_size = reduced_entity_embeddings.shape[2]

        device = next(self.parameters()).device
        zero_bias = torch.zeros(batch_size, 1, embed_size, device=device)
        reduced_entity_embeddings = torch.cat([zero_bias, reduced_entity_embeddings[:, 1:, :]], dim=1)
        print('reduced_entity_embeddings.shape', reduced_entity_embeddings.shape) if debug else None

        # [batch_size x 4 x AHP.minimap_size x AHP.minimap_size]
        scatter_index = scatter_index.reshape(batch_size, -1)
        scatter_index = scatter_index.unsqueeze(-1).repeat(1, 1, AHP.original_32)
        # [batch_size x 4 * AHP.minimap_size * AHP.minimap_size x AHP.original_32]

        # Question: This has a problem, the first element of index 0 will be averaged everywhere
        # Solution: use zero_bias to remove
        scatter_mid = reduced_entity_embeddings.gather(1, scatter_index.long())
        del reduced_entity_embeddings, scatter_index, zero_bias
        print('scatter_mid', scatter_mid[0, :16, :4]) if debug else None

        scatter_mid = scatter_mid.reshape(batch_size, self.scatter_volume, 
                                          self.map_width, self.map_width, AHP.original_32)

        if same_pos_handle == 'add':
            scatter_result = torch.sum(scatter_mid, dim=1)
        elif same_pos_handle == 'mean':
            scatter_result = torch.mean(scatter_mid, dim=1)
        else:
            scatter_result = torch.sum(scatter_mid, dim=1)

        scatter_result = scatter_result.permute(0, 3, 1, 2)
        del scatter_mid

        return scatter_result 

    def forward(self, x, entity_embeddings=None):
        device = next(self.parameters()).device

        # scatter_map may cause a NaN bug in SL training, now we don't use it
        if entity_embeddings is not None:
            channels = x.shape[1]

            # the first 4 channels are scatter map
            scatter_map, reduced = torch.split(x, [self.scatter_volume, channels - self.scatter_volume], dim=1)
            scatter_entity = self.scatter(scatter_map, entity_embeddings)

            batch_size = scatter_entity.shape[0]

            if P.handle_cuda_error:  # and batch_size != 1:
                reduced = reduced.to('cpu')
                scatter_entity = scatter_entity.to('cpu')

            x = torch.cat([scatter_entity, reduced], dim=1)

            if P.handle_cuda_error:  # and batch_size != 1:
                x = x.to(device)

            del scatter_map, scatter_entity, reduced

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

        del x

        return map_skip, embedded_spatial

    @classmethod
    def get_map_data(cls, obs, entity_pos_list=None, map_width=AHP.minimap_size, verbose=False):
        '''
        default map_width is 64
        '''
        feature_minimap = obs["feature_minimap"] if "feature_minimap" in obs else obs
        save_type = np.float32

        # we consider the most 4 entities in the same position
        scatter_map = np.zeros((1, map_width, map_width, cls.scatter_volume), dtype=save_type)
        if entity_pos_list is not None:
            # make the scatter_map has the index of entity in the entity's position (matrix format)      
            for i, (x, y) in enumerate(entity_pos_list):
                scale_factor = AAIFP.raw_resolution / map_width
                x = int(x / scale_factor)
                y = int(y / scale_factor)

                for j in range(cls.scatter_volume):
                    if not scatter_map[0, y, x, j]:
                        scatter_map[0, y, x, j] = min(i + 1, AHP.max_entities - 1)
                        break

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

        del scatter_map, camera_map, height_map, visibility, creep, entity_owners, alerts, pathable, buildable

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
        # Batch norm is applied in the channels, C of (N, C, H, W)
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
        del identity
        return out


class ResBlockImproved(nn.Module):

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlockImproved, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # Batch norm is applied in the channels, C of (N, C, H, W)
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
        del residual
        return x


class ResBlock1D(nn.Module):

    def __init__(self, inplanes, planes, seq_len, 
                 stride=1, downsample=None, norm_type='prev'):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # Layer norm is applied in the not N channels, 
        # e.g., E of (N, S, E) in NLP,
        # and [C, H, W] of (N, C, H, W) in CV.
        # Note, because Layer norm doesn't average over dim of batch (N), so
        # it is the same in training and evaluating.
        # For Batch Norm, this is not the case.
        self.ln1 = nn.LayerNorm([planes, seq_len])
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.ln2 = nn.LayerNorm([planes, seq_len])
        self.normtype = norm_type

    def forward(self, x):
        if self.normtype == 'prev':
            residual = x
            x = F.relu(self.ln1(x))
            x = self.conv1(x)
            x = F.relu(self.ln2(x))
            x = self.conv2(x)
            x = x + residual
            del residual
            return x
        elif self.normtype == 'post':
            residual = x
            x = F.relu(self.conv1(x))
            x = self.ln1(x)
            x = F.relu(self.conv2(x))
            x = self.ln2(x)
            x = x + residual
            del residual
            return x
        else:
            raise KeyError('unspported normtype!')     


def test():
    spatial_encoder = SpatialEncoder()
    batch_size = 2
    # dummy map list
    map_list = []
    map_data_1 = torch.zeros(batch_size, 18, AHP.minimap_size, AHP.minimap_size)
    map_data_1_one_hot = L.to_one_hot(map_data_1, 2)
    print('map_data_1_one_hot.shape:', map_data_1_one_hot.shape) if debug else None

    map_list.append(map_data_1)
    map_data_2 = torch.zeros(batch_size, 18, AHP.minimap_size, AHP.minimap_size)
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
