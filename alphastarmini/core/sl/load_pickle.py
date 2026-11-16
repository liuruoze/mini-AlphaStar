#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Load features and labels from the replay .pickle files"

import os
import sys
import traceback
import pickle

import numpy as np

import torch

from torch.utils.data import TensorDataset

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_utils as SU

__author__ = "Ruo-Ze Liu"

debug = False

FLAGS = flags.FLAGS
flags.DEFINE_string("replay_data_path", "./data/replay_data/", "path to replays_save replay data")
flags.DEFINE_string("save_tensor_path", "./data/replay_data_tensor/", "path to replays_save replay data tensor")
FLAGS(sys.argv)

ON_SERVER = False


def print_tensor_list(tensor_list):
    for l in tensor_list:
        if isinstance(l, list):
            print_tensor_list(l)
        else:
            print('l.shape', l.shape)


def from_pickle_to_tensor(pickle_path, tensor_path, from_index=0, end_index=None):
    replay_files = os.listdir(pickle_path)
    print('length of replay_files:', len(replay_files))
    replay_files.sort()

    replay_length_list = []
    traj_list = []
    for i, replay_file in enumerate(replay_files):
        try:
            do_write = False
            if i >= from_index:
                if end_index is None:
                    do_write = True
                elif end_index is not None and i < end_index:
                    do_write = True

            if not do_write:
                continue    

            replay_path = pickle_path + replay_file
            print('replay_path:', replay_path)

            feature_list = []
            label_list = []

            with open(replay_path, 'rb') as handle:
                traj_dict = pickle.load(handle)

                j = 0
                for key, value in traj_dict.items():
                    # if j > 10:
                    #     break
                    feature, label = SU.obs2feature_numpy(value)
                    feature_list.append(feature)
                    label_list.append(label)
                    del value, feature, label
                    j += 1 

                del traj_dict

            features = np.concatenate(feature_list, axis=0)
            print("features.shape:", features.shape) if debug else None
            del feature_list

            labels = np.concatenate(label_list, axis=0)
            print("labels.shape:", labels.shape) if debug else None
            del label_list

            features = torch.tensor(features)
            labels = torch.tensor(labels)

            m = (features, labels)

            if not os.path.exists(tensor_path):
                os.mkdir(tensor_path)
            file_name = tensor_path + replay_file.replace('.pickle', '') + '.pt'
            torch.save(m, file_name)

        except Exception as e:
            traceback.print_exc()    

    print("end")
    print("replay_length_list:", replay_length_list)


def test(on_server=False):
    from_pickle_to_tensor(FLAGS.replay_data_path, FLAGS.save_tensor_path, 15, 20)
