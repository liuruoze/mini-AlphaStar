#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay files through python pickle file"

import os

USED_DEVICES = "0"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import time
import traceback
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, RMSprop

from tensorboardX import SummaryWriter

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.agent import Agent

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_loss as Loss
from alphastarmini.core.sl.dataset_pickle import OneReplayDataset, AllReplayDataset

from alphastarmini.lib.utils import load_latest_model
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP

__author__ = "Ruo-Ze Liu"

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="./data/replay_data/", help="The path where data stored")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument("-t", "--type", choices=["val", "test", "deploy"], default="val", help="Train type")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
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
print('BATCH_SIZE:', BATCH_SIZE)
SEQ_LEN = AHP.sequence_length
print('SEQ_LEN:', SEQ_LEN)
NUM_EPOCHS = SLTHP.num_epochs
LEARNING_RATE = SLTHP.learning_rate
WEIGHT_DECAY = SLTHP.weight_decay
CLIP = SLTHP.clip

NUM_ITERS = 10  # 100
FILE_SIZE = 100

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


def train_for_val(replays, replay_data, agent):
    now = datetime.now()
    summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(summary_path)

    train_replays = AllReplayDataset.get_training_for_val_data(replays)
    val_replays = AllReplayDataset.get_val_data(replays)

    train_set = AllReplayDataset(train_replays)
    val_set = AllReplayDataset(val_replays)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    if RESTORE:
        agent.model = load_latest_model(model_type=MODEL, path=MODEL_PATH)

    print('torch.cuda.device_count():', torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        agent.model = nn.DataParallel(agent.model)

    agent.model.to(DEVICE)
    # agent.model.cuda()

    optimizer = Adam(agent.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loss = 0
    for epoch in range(NUM_EPOCHS):
        agent.model.train()

        loss_sum = 0.0
        i = 0
        last_traj = None

        for j in range(NUM_ITERS):
            traj = next(iter(train_loader))

            print('traj.shape:', traj.shape) if debug else None

            traj = traj.to(DEVICE).float()

            if last_traj is not None:
                print("traj == last_traj ?", torch.equal(traj, last_traj)) if debug else None

            # with torch.autograd.detect_anomaly():
            loss = Loss.get_sl_loss(traj, agent.model)
            optimizer.zero_grad()
            loss.backward()  # note, we don't need retain_graph=True if we set hidden_state.detach()

            # add a grad clip
            parameters = [p for p in agent.model.parameters() if p is not None and p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, CLIP)

            optimizer.step()
            loss_sum += loss.item()

            last_traj = traj.clone().detach()
            i += 1

        train_loss = loss_sum / (i + 1e-9)
        val_loss = eval(agent, val_loader)

        print("Train loss: {:.6f}.".format(train_loss))
        writer.add_scalar('Train/Loss', train_loss, epoch)
        print("Val loss: {:.6f}.".format(val_loss))
        writer.add_scalar('Val/Loss', val_loss, epoch)

        print("beign to save model in " + SAVE_PATH)
        torch.save(agent.model, SAVE_PATH + "" + ".pkl")


def eval(agent, val_loader):
    agent.model.eval()

    loss_sum = 0.0
    i = 0
    for traj in val_loader:
        traj = traj.to(DEVICE).float()

        loss = Loss.get_sl_loss(traj, agent.model)
        loss_sum += loss.item()
        i += 1

    val_loss = loss_sum / (i + 1e-9)
    return val_loss


def test(on_server):
    # get all the data
    # Note: The feature here is actually feature+label

    agent = Agent()

    replays = AllReplayDataset.get_trainable_data(replay_data_path=PATH, agent=agent, max_file_size=FILE_SIZE)

    '''
    for replay in replays:
        print('replay', replay)
        for tensor in replay:
            print('tensor:', tensor)
    '''

    if TYPE == 'val':
        train_for_val(replays, None, agent)  # select the best hyper-parameters
    elif TYPE == 'test':
        train_for_test(replays, None, agent)  # for test the performance in real life
    elif TYPE == 'deploy':
        train_for_deploy(replays, None, agent)  # only used for production
    else:
        train_for_val(replays, None, agent)
