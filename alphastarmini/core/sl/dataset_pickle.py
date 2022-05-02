#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Dataset for sc2 replays, newly for pickle data"

import os
import time
import traceback
import pickle
import random

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.hyper_parameters import DATASET_SPLIT_RATIO
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False


class OneReplayDataset(Dataset):

    def __init__(self, traj_dict, seq_length=AHP.sequence_length):
        super().__init__()

        self.traj_dict = traj_dict

        # if we use "self.keys =traj_dict.keys()", this will cause a shallow copy problem, refer
        # to https://discuss.pytorch.org/t/dataloader-multiprocessing-error-cant-pickle-odict-
        # keys-objects-when-num-workers-0/43951/4
        self.keys = list(traj_dict.keys())
        self.seq_len = seq_length

    def __getitem__(self, index):

        key_list = self.keys[index:index + self.seq_len]
        feature_list = []
        label_list = []
        obs_list = []

        for key in key_list:
            obs = self.traj_dict[key]

            feature, label = SU.obs2feature(obs)
            feature_list.append(feature)
            label_list.append(label)

        features = torch.cat(feature_list, dim=0)
        print("features.shape:", features.shape) if 0 else None

        labels = torch.cat(label_list, dim=0)
        print("labels.shape:", labels.shape) if 0 else None

        is_final = torch.zeros([features.shape[0], 1])

        one_traj = torch.cat([features, labels, is_final], dim=1)
        print("one_traj.shape:", one_traj.shape) if debug else None

        return one_traj

    def __len__(self):
        return len(self.keys) - self.seq_len


class AllReplayDataset(Dataset):

    def __init__(self, traj_loader_list=None):
        super().__init__()
        self.traj_loader_list = traj_loader_list

    def _get_random_trajectory(self, index):
        traj_loader = self.traj_loader_list[index]

        # note: don't use "for traj in traj_loader" which is too slow
        # we only need one item from traj_loader now
        the_item = next(iter(traj_loader))

        return the_item

    def __getitem__(self, index):
        replay = self._get_random_trajectory(index)
        print('replay.shape:', replay.shape) if 0 else None
        replay = replay.squeeze(0)
        print('replay.shape:', replay.shape) if 0 else None

        return replay

    def __len__(self):
        return len(self.traj_loader_list)

    @staticmethod
    def get_trainable_data(replay_data_path, max_file_size=None, shuffle=False):
        replay_files = os.listdir(replay_data_path)
        print('length of replay_files:', len(replay_files))

        replay_files.sort()
        if shuffle:
            random.shuffle(replay_files)

        traj_loader_list = []
        for i, replay_file in enumerate(tqdm(replay_files)):
            try:
                if max_file_size is not None:
                    if i >= max_file_size:
                        break

                replay_path = replay_data_path + replay_file
                print('replay_path:', replay_path) if debug else None

                with open(replay_path, 'rb') as handle:
                    traj_dict = pickle.load(handle)                  
                    traj_dataset = OneReplayDataset(traj_dict=traj_dict)
                    traj_loader = DataLoader(traj_dataset, batch_size=1, shuffle=True)
                    traj_loader_list.append(traj_loader)

            except Exception as e:
                traceback.print_exc()
        return traj_loader_list

    @staticmethod
    def get_training_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        print('training_size:', training_size)
        return trajs[0:training_size] 

    @staticmethod
    def get_val_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        if val_size == 0 and len(trajs) >= 5:
            val_size = 1
        print('val_size:', val_size)
        return trajs[-val_size:]

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
        if training_size == 0 and len(trajs) == 1:
            training_size = 1
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


class FullDataset(Dataset):

    def __init__(self, replay_data_path, val=False, max_file_size=None, shuffle=False, seq_length=AHP.sequence_length):
        super().__init__()

        self.seq_len = seq_length

        replay_files = os.listdir(replay_data_path)
        print('length of replay_files:', len(replay_files)) if debug else None
        replay_files.sort()

        max_file_size = max_file_size if max_file_size < len(replay_files) else len(replay_files)
        print('max_file_size', max_file_size) if debug else None

        training_size = int(max_file_size * 0.95)
        if val:
            replay_files = replay_files[training_size:max_file_size]
            print('length for val files:', len(replay_files)) if debug else None
        else:
            replay_files = replay_files[0:training_size]
            print('length for train files:', len(replay_files)) if debug else None

        if shuffle:
            random.shuffle(replay_files)

        self.obs_list = []
        obs_index = 0
        self.final_index_list = []

        for i, replay_file in enumerate(tqdm(replay_files)):
            try:
                replay_path = replay_data_path + replay_file
                print('replay_path:', replay_path) if debug else None

                with open(replay_path, 'rb') as handle:
                    traj_dict = pickle.load(handle)                  
                    #key_list = list(traj_dict.keys())

                    #print('len(key_list):', len(key_list)) if debug else None
                    for key, value in traj_dict.items():
                        print('key', key) if debug else None
                        # feature, label = SU.obs2feature(value)
                        # self.feature_list.append(feature)
                        # self.label_list.append(label)
                        self.obs_list.append(value)
                        obs_index = obs_index + 1

                    self.final_index_list.append(obs_index - 1)

            except Exception as e:
                traceback.print_exc()

    def __getitem__(self, index):
        #t = self.get_tensor_item(index)

        result = self.get_array_item(index)

        return result

    def get_array_item(self, index):
        obs = self.obs_list[index:index + self.seq_len]

        feature_list = []
        label_list = []
        for value in obs:
            feature, label = SU.obs2feature_numpy(value)
            feature_list.append(feature)
            label_list.append(label)

        features = np.concatenate(feature_list, axis=0)
        print("features.shape:", features.shape) if debug else None

        labels = np.concatenate(label_list, axis=0)
        print("labels.shape:", labels.shape) if debug else None

        is_final = np.zeros([features.shape[0], 1])

        # consider is_final
        # for speed, we don't use it now
        if False:
            print('begin', index) if debug else None
            print('end', index + self.seq_len) if debug else None
            for j in self.final_index_list:
                print('j', j) if debug else None
                if j >= index and j < index + self.seq_len:
                    if debug:
                        print('in it!') 
                        print('begin', index)
                        print('end', index + self.seq_len)
                        print('j', j)
                    is_final[j - index, 0] = 1
                else:
                    pass

        one_traj = np.concatenate([features, labels, is_final], axis=1)

        del features, labels, is_final

        print("one_traj.shape:", one_traj.shape) if debug else None
        print("one_traj:", one_traj) if debug else None

        return one_traj

    def get_tensor_item(self, index):
        obs = self.obs_list[index:index + self.seq_len]

        # note, we must not convert the obs to tensor, otherwise this will cause a memory leak of shared memory
        # in PyTorch's dataloader, this is a bug reported:
        # https://discuss.pytorch.org/t/training-crashes-due-to-insufficient-shared-memory-shm-nn-dataparallel/26396/18

        feature_list = []
        label_list = []
        for value in obs:
            feature, label = SU.obs2feature(value)
            feature_list.append(feature)
            label_list.append(label)

        features = torch.cat(feature_list, dim=0)
        print("features.shape:", features.shape) if debug else None

        labels = torch.cat(label_list, dim=0)
        print("labels.shape:", labels.shape) if debug else None

        is_final = torch.zeros([features.shape[0], 1])

        # consider is_final
        print('begin', index) if debug else None
        print('end', index + self.seq_len) if debug else None
        for j in self.final_index_list:
            print('j', j) if debug else None
            if j >= index and j < index + self.seq_len:
                if debug:
                    print('in it!') 
                    print('begin', index)
                    print('end', index + self.seq_len)
                    print('j', j)
                is_final[j - index, 0] = 1
            else:
                pass

        one_traj = torch.cat([features, labels, is_final], dim=1)
        print("one_traj.shape:", one_traj.shape) if debug else None

        # The elegant processing method should be to directly generate numpy array, 
        # but this may require a lot of changes. The stupid but simple approach taken 
        # here is to directly convert tensor to numpy.
        one_traj = one_traj.cpu().detach().numpy()
        print("one_traj:", one_traj) if debug else None

        return one_traj

    def __len__(self):
        return len(self.obs_list) - self.seq_len + 1


def test():
    pass
