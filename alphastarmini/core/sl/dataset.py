#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Dataset for sc2 replays: "

import os
import time
import traceback
from typing import Tuple

from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset

from alphastarmini.lib.hyper_parameters import DATASET_SPLIT_RATIO
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False


class SC2ReplayData(object):
    '''
        Note: The feature here is actually feature+label
    '''

    def __init__(self):
        super().__init__()
        self.initialed = False
        self.feature_size = None
        self.label_size = None
        self.replay_length_list = []

    def get_trainable_data(self, path):
        out_features = None

        print('data path:', path)
        replay_files = os.listdir(path)

        print('length of replay_files:', len(replay_files))
        replay_files.sort()

        traj_list = []
        for i, replay_file in enumerate(replay_files):
            try:
                if i > 15:
                    break
                replay_path = path + replay_file
                print('replay_path:', replay_path)

                m = torch.load(replay_path)
                features = m['features']
                labels = m['labels']
                assert len(features) == len(labels)

                if self.feature_size:
                    assert self.feature_size == features.shape[1]
                    assert self.label_size == labels.shape[1]  
                else:
                    self.feature_size = features.shape[1]
                    self.label_size = labels.shape[1]

                print('self.feature_size:', self.feature_size)
                print('self.label_size:', self.label_size)

                is_final = torch.zeros([features.shape[0], 1])
                is_final[features.shape[0] - 1, 0] = 1

                one_traj = torch.cat([features, labels, is_final], dim=1)

                traj_list.append(one_traj)
                self.replay_length_list.append(one_traj.shape[0])

            except Exception as e:
                traceback.print_exc()

        print("end")
        print("self.replay_length_list:", self.replay_length_list)

        # note: do not do below line because memory can not affort thig big tensor
        #out_features = torch.cat(out_features_list, dim=0)

        self.initialed = True
        return traj_list

    @staticmethod
    def filter_data(feature, label):
        tmp_feature = None

        return tmp_feature

    @staticmethod
    def get_training_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        print('training_size:', training_size)
        return trajs[0:training_size] 

    @staticmethod
    def get_val_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        print('val_size:', val_size)
        return trajs[training_size:training_size + val_size]

    @staticmethod
    def get_test_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        test_size = int(len(trajs) * DATASET_SPLIT_RATIO.test)
        print('test_size:', test_size)
        return trajs[-test_size:]

    @staticmethod
    def get_training_for_val_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        print('training_size:', training_size)
        return trajs[0:training_size]

    @staticmethod
    def get_training_for_test_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        print('training_for_test_size:', training_size + val_size)
        return trajs[0:training_size + val_size]

    @staticmethod
    def get_training_for_deploy_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        test_size = int(len(trajs) * DATASET_SPLIT_RATIO.test)
        print('training_for_deploy_size:', training_size + val_size + test_size)
        return trajs[0:training_size + val_size + test_size]


class SC2ReplayDataset(Dataset):

    def __init__(self, traj_list, seq_length, training=False):
        super().__init__()

        self.traj_list = traj_list
        self.seq_length = seq_length
        #print("self.seq_length:", self.seq_length)

    def __getitem__(self, index):
        old_start = 0
        begin = index * self.seq_length
        end = (index + 1) * self.seq_length

        for i, one_traj in enumerate(self.traj_list):
            new_start = old_start + one_traj.shape[0]
            #print('old_start:', old_start)
            #print('new_start:', new_start)             

            if begin >= new_start:
                pass
            else:
                index_begin = begin - old_start
                if end < new_start:            
                    index_end = end - old_start
                    return one_traj[index_begin:index_end, :]
                elif i < len(self.traj_list) - 1:
                    next_traj = self.traj_list[i + 1]

                    first_part = one_traj[index_begin:, :]
                    second_part = next_traj[:self.seq_length - len(first_part), :]
                    return torch.cat([first_part, second_part], dim=0)

            old_start = new_start

    def __len__(self):
        max_len = 0
        for one_traj in self.traj_list:
            max_len += one_traj.shape[0]
        max_len -= self.seq_length

        return int(max_len / self.seq_length)


class ReplayTensorDataset(TensorDataset):

    # seq_len=AHP.sequence_length
    def __init__(self, *tensors: Tensor, seq_len) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.seq_len = seq_len

    def __getitem__(self, index):
        return tuple(tensor[index:index + self.seq_len] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0) - self.seq_len + 1


def test():
    pass
