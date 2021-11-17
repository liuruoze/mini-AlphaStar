#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay .pt files through pytorch tensor"

import os

USED_DEVICES = "7"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import time
import traceback
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, RMSprop

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.agent import Agent

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl.dataset import SC2ReplayData, SC2ReplayDataset

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP

__author__ = "Ruo-Ze Liu"

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="./data/replay_data/", help="The path where data stored")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument("-t", "--type", choices=["val", "test", "deploy"], default="val", help="Train type")
parser.add_argument("-m", "--model", choices=["lstm", "gru", "fc"], default="lstm", help="Choose policy network")
parser.add_argument("-n", "--norm", choices=[True, False], default=False, help="Use norm for data")
args = parser.parse_args()

# training paramerters
PATH = args.path
MODEL = args.model
TYPE = args.type
RESTORE = args.restore
NORM = args.norm

# hyper paramerters
BATCH_SIZE = AHP.batch_size
SEQ_LEN = AHP.sequence_length
NUM_EPOCHS = SLTHP.num_epochs
LEARNING_RATE = SLTHP.learning_rate
WEIGHT_DECAY = SLTHP.weight_decay
CLIP = SLTHP.clip

# set random seed
torch.manual_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
torch.autograd.set_detect_anomaly(True)

# model path
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))


def train_for_val(feature, replay_data):
    train_feature = SC2ReplayData.get_training_for_val_data(feature)
    val_feature = SC2ReplayData.get_val_data(feature)

    train_set = SC2ReplayDataset(train_feature, seq_length=SEQ_LEN, training=NORM)
    val_set = SC2ReplayDataset(val_feature, seq_length=SEQ_LEN, training=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

    #model = load_latest_model() if RESTORE else choose_model(MODEL)
    # model.to(DEVICE)

    agent = Agent()
    agent.to(DEVICE)

    optimizer = Adam(agent.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loss = 0
    for epoch in range(NUM_EPOCHS):
        agent.model.train()

        loss_sum = 0.0
        i = 0
        for traj in train_loader:
            traj = traj.to(DEVICE).float()

            with torch.autograd.detect_anomaly():
                loss = agent.get_sl_loss(traj, replay_data)
                optimizer.zero_grad()
                loss.backward()  # note, we don't need retain_graph=True if we set hidden_state.detach()

                # add a grad clip
                parameters = [p for p in agent.model.parameters() if p is not None and p.requires_grad]
                torch.nn.utils.clip_grad_norm_(parameters, CLIP)

                optimizer.step()
                loss_sum += loss.item()
            i += 1

        train_loss = loss_sum / (i + 1e-9)
        #val_loss = eval(model, criterion, val_loader, train_set, val_set)
        print("Train loss: {:.6f}.".format(train_loss))
        #print("Train loss: {:.6f}, Val loss: {:.6f}.".format(train_loss, val_loss))

    torch.save(agent.model, SAVE_PATH + "_val" + ".pkl")


def eval(model, criterion, data_loader, train_set, val_set):
    model.eval()

    n_samples = len(val_set)
    loss_sum = 0.0

    for feature, target in data_loader:
        feature = feature.to(DEVICE).float()
        target = target.to(DEVICE).float()

        output = agent.unroll(feature)

        if debug:
            print("feature.size(): ", feature.size())
            print("target.size(): ", target.size())
            print("output.size(): ", output.size())
            break

        loss = criterion(output, target)
        loss_sum += output.size(0) * loss.item()

    return loss_sum / n_samples


def test(on_server):
    # get all the data
    # Note: The feature here is actually feature+label
    replay_data = SC2ReplayData()
    features = replay_data.get_trainable_data(PATH)

    #print('out_features:', features)

    if TYPE == 'val':
        train_for_val(features, replay_data)  # select the best hyper-parameters
    elif TYPE == 'test':
        train_for_test(features, replay_data)  # for test the performance in real life
    elif TYPE == 'deploy':
        train_for_deploy(features, replay_data)  # only used for production
    else:
        train_for_val(features, replay_data)
