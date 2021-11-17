#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay files through python pickle file"

import os
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

from alphastarmini.lib.utils import load_latest_model, initial_model_state_dict
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
print('BATCH_SIZE:', BATCH_SIZE) if debug else None
SEQ_LEN = AHP.sequence_length
print('SEQ_LEN:', SEQ_LEN) if debug else None

NUM_EPOCHS = 100  # SLTHP.num_epochs
LEARNING_RATE = 1e-4  # SLTHP.learning_rate
WEIGHT_DECAY = 1e-5  # SLTHP.weight_decay
CLIP = 0.5  # SLTHP.clip

NUM_ITERS = 100  # 100
FILE_SIZE = 100  # 100

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
    if RESTORE:
        # agent.model = load_latest_model(model_type=MODEL, path=MODEL_PATH)
        # use state dict to replace
        initial_model_state_dict(model_type=MODEL, path=MODEL_PATH, model=agent.model)

    print('torch.cuda.device_count():', torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     used_gpu_nums = 2  # torch.cuda.device_count()
    #     print("Let's use", used_gpu_nums, "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     agent.model = nn.DataParallel(agent.model, device_ids=[0, 1])

    agent.model.to(DEVICE)
    # agent.model.cuda()

    now = datetime.now()
    summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(summary_path)

    train_replays = AllReplayDataset.get_training_for_val_data(replays)
    val_replays = AllReplayDataset.get_val_data(replays)

    train_set = AllReplayDataset(train_replays)
    val_set = AllReplayDataset(val_replays)

    print('len(train_set)', len(train_set))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    print('len(train_loader)', len(train_loader))

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = Adam(agent.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loss = 0
    batch_iter = 0
    for epoch in range(NUM_EPOCHS):
        agent.model.train()

        loss_sum = 0.0
        i = 0
        last_traj = None

        for j in range(NUM_ITERS):
            traj = next(iter(train_loader))

            print('traj.shape:', traj.shape) if debug else None

            traj = traj.to(DEVICE).float()

            # if last_traj is not None:
            #     print("traj == last_traj ?", torch.equal(traj, last_traj)) if debug else None

            # with torch.autograd.detect_anomaly():
            try:
                loss, loss_list, acc_num_list = Loss.get_sl_loss(traj, agent.model)
                optimizer.zero_grad()
                loss.backward()  # note, we don't need retain_graph=True if we set hidden_state.detach()

                action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)
                move_camera_accuracy = acc_num_list[2] / (acc_num_list[3] + 1e-9)
                non_camera_accuracy = acc_num_list[4] / (acc_num_list[5] + 1e-9)

                # add a grad clip
                parameters = [p for p in agent.model.parameters() if p is not None and p.requires_grad]
                torch.nn.utils.clip_grad_norm_(parameters, CLIP)

                optimizer.step()
                loss_sum += loss.item()

                print("One batch loss: {:.6f}.".format(loss.item()))
                writer.add_scalar('OneBatch/Loss', loss.item(), batch_iter)

                if True:
                    print("One batch action_type_loss loss: {:.6f}.".format(loss_list[0].item()))
                    writer.add_scalar('OneBatch/action_type_loss', loss_list[0].item(), batch_iter)

                    print("One batch delay_loss loss: {:.6f}.".format(loss_list[1].item()))
                    writer.add_scalar('OneBatch/delay_loss', loss_list[1].item(), batch_iter)

                    print("One batch queue_loss loss: {:.6f}.".format(loss_list[2].item()))
                    writer.add_scalar('OneBatch/queue_loss', loss_list[2].item(), batch_iter)

                    print("One batch units_loss loss: {:.6f}.".format(loss_list[3].item()))
                    writer.add_scalar('OneBatch/units_loss', loss_list[3].item(), batch_iter)

                    print("One batch target_unit_loss loss: {:.6f}.".format(loss_list[4].item()))
                    writer.add_scalar('OneBatch/target_unit_loss', loss_list[4].item(), batch_iter)

                    print("One batch target_location_loss loss: {:.6f}.".format(loss_list[5].item()))
                    writer.add_scalar('OneBatch/target_location_loss', loss_list[5].item(), batch_iter)

                    print("One batch action_accuracy: {:.6f}.".format(action_accuracy))
                    writer.add_scalar('OneBatch/action_accuracy', action_accuracy, batch_iter)

                    print("One batch move_camera_accuracy: {:.6f}.".format(move_camera_accuracy))
                    writer.add_scalar('OneBatch/move_camera_accuracy', move_camera_accuracy, batch_iter)

                    print("One batch non_camera_accuracy: {:.6f}.".format(non_camera_accuracy))
                    writer.add_scalar('OneBatch/non_camera_accuracy', non_camera_accuracy, batch_iter)

            #last_traj = traj.clone().detach()
            except Exception as e:               
                print(traceback.format_exc())

            i += 1
            batch_iter += 1

        train_loss = loss_sum / (i + 1e-9)
        val_loss, val_acc = eval(agent, val_loader)

        print("Train loss: {:.6f}.".format(train_loss))
        writer.add_scalar('Train/Loss', train_loss, epoch)
        print("Val loss: {:.6f}.".format(val_loss))
        writer.add_scalar('Val/Loss', val_loss, epoch)

        # for accuracy of actions in val
        print("Val action acc: {:.6f}.".format(val_acc[0]))
        writer.add_scalar('Val/action Acc', val_acc[0], epoch)
        print("Val move_camera acc: {:.6f}.".format(val_acc[1]))
        writer.add_scalar('Val/move_camera Acc', val_acc[1], epoch)
        print("Val non_camera acc: {:.6f}.".format(val_acc[2]))
        writer.add_scalar('Val/non_camera Acc', val_acc[2], epoch)

        print("beign to save model in " + SAVE_PATH)

        # torch.save(agent.model, SAVE_PATH + "" + ".pkl")
        # we use new save ways, only save the state_dict, and the extension changes to pt
        torch.save(agent.model.state_dict(), SAVE_PATH + "" + ".pth")


def eval(agent, val_loader):
    agent.model.eval()

    loss_sum = 0.0
    i = 0

    action_acc_num = 0.
    action_all_num = 0.

    move_camera_action_acc_num = 0.
    move_camera_action_all_num = 0.

    non_camera_action_acc_num = 0.
    non_camera_action_all_num = 0.

    for traj in val_loader:
        traj = traj.to(DEVICE).float()

        loss, _, acc_num_list = Loss.get_sl_loss(traj, agent.model, use_eval=True)
        loss_sum += loss.item()

        action_acc_num += acc_num_list[0]
        action_all_num += acc_num_list[1]

        move_camera_action_acc_num += acc_num_list[2]
        move_camera_action_all_num += acc_num_list[3]

        non_camera_action_acc_num += acc_num_list[4]
        non_camera_action_all_num += acc_num_list[5]

        i += 1

    val_loss = loss_sum / (i + 1e-9)

    action_accuracy = action_acc_num / (action_all_num + 1e-9)
    move_camera_accuracy = move_camera_action_acc_num / (move_camera_action_all_num + 1e-9)
    non_camera_accuracy = non_camera_action_acc_num / (non_camera_action_all_num + 1e-9)

    return val_loss, [action_accuracy, move_camera_accuracy, non_camera_accuracy]


def test(on_server):
    # get all the data
    # Note: The feature here is actually feature+label

    agent = Agent()

    replays = AllReplayDataset.get_trainable_data(replay_data_path=PATH,
                                                  max_file_size=FILE_SIZE, shuffle=True)

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
