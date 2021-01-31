#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Load features and labels from the replay .pickle files"

import os
import sys
import traceback
import pickle

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
    for replay_file in replay_files:
        try:
            replay_path = DATA_PATH + replay_file
            print('replay_path:', replay_path)

            with open(replay_path, 'rb') as handle:
                b = pickle.load(handle)
                keys = b.keys()
                #print('keys:', keys)
                for key in keys:
                    obs = b[key]
                    s = agent.get_state_and_action_from_pickle(obs)
                    feature = Feature.state2feature(s)
                    print("feature:", feature) if debug else None
                    print("feature.shape:", feature.shape) if debug else None

                    print("begin a:") if debug else None
                    func_call = obs['func_call']
                    action = agent.func_call_to_action(func_call).toTenser()
                    #tag_list = agent.get_tag_list(obs)
                    print('action.get_shape:', action.get_shape()) if debug else None

                    logits = action.toLogits()
                    print('logits.shape:', logits) if debug else None
                    label = Label.action2label(logits)
                    print("label:", label) if debug else None
                    print("label.shape:", label.shape) if debug else None

        except Exception as e:
            traceback.print_exc()    

    print("end")
    print("replay_length_list:", replay_length_list)
