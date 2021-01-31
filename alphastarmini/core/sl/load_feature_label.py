#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Load features and labels from the replay .pt files"

import os
import sys
import traceback

import torch

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

__author__ = "Ruo-Ze Liu"

debug = False

FLAGS = flags.FLAGS
flags.DEFINE_string("replay_data_path", "./data/replay_data/", "path to replays_save replay data")
FLAGS(sys.argv)

ON_SERVER = False


def print_tensor_list(tensor_list):
    for l in tensor_list:
        if isinstance(l, list):
            print_tensor_list(l)
        else:
            print(l.shape)


def test():
    DATA_PATH = FLAGS.replay_data_path
    print('data path:', DATA_PATH)
    replay_files = os.listdir(DATA_PATH)
    print('length of replay_files:', len(replay_files))
    replay_files.sort()

    agent = Agent()
    j = 0
    replay_length_list = []
    traj_list = []
    for replay_file in tqdm(replay_files):
        try:
            replay_path = DATA_PATH + replay_file
            print('replay_path:', replay_path)

            traj = []
            m = torch.load(replay_path)
            features = m['features']
            labels = m['labels']

            assert len(features) == len(labels)

            for i in range(len(features)):
                feature = features[i:i + 1, :]
                label = labels[i:i + 1, :]
                isfinal = 0
                if i == len(features) - 1:
                    isfinal = 1

                print('feature.shape:', feature.shape)
                print('label.shape:', label.shape)

                state = Feature.feature2state(feature)
                action_gt = Label.label2action(label)

                print("action_gt:", action_gt)
                print('state:', state)
                print('isfinal:', isfinal)
                #action_logits_prdict = agent.action_logits_by_state(state)
                #print("action_logits_prdict:", action_logits_prdict)

                traj.append([state, action_gt, isfinal])

            '''TODO: change to batch version
                states = Feature.feature2state(features)
                print("states:", states)
                action_gts = Label.label2action(labels)
                print("action_gts:", action_gts)
                action_prdicts = agent.action_by_state(states)
                print("action_prdicts:", action_prdicts)
            '''

        except Exception as e:
            traceback.print_exc()    

    print("end")
    print("replay_length_list:", replay_length_list)
