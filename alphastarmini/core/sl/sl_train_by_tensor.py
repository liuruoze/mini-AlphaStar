#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay files through python tensor (pt) file"

import os
import gc

import sys
import time
import traceback
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, RMSprop

from tensorboardX import SummaryWriter

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.arch_model import ArchModel

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_loss_multi_gpu as Loss
from alphastarmini.core.sl.dataset import ReplayTensorDataset
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.utils import load_latest_model, initial_model_state_dict
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP

import param as P

__author__ = "Ruo-Ze Liu"

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="./data/replay_data_tensor_new/", help="The path where data stored")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument('--num_workers', type=int, default=1, help='')


args = parser.parse_args()

# training paramerters
PATH = args.path
MODEL = args.model
RESTORE = args.restore
NUM_WORKERS = args.num_workers

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
RESTORE_PATH = MODEL_PATH + 'sl_21-11-23_22-25-27.pth' 

SIMPLE_TEST = not P.on_server
if SIMPLE_TEST:
    TRAIN_FROM = 0
    TRAIN_NUM = 2

    VAL_FROM = 0
    VAL_NUM = 1
else:
    TRAIN_FROM = 0
    TRAIN_NUM = 12

    VAL_FROM = 12
    VAL_NUM = 3


# hyper paramerters
BATCH_SIZE = AHP.batch_size
print('BATCH_SIZE:', BATCH_SIZE) if debug else None
SEQ_LEN = AHP.sequence_length
print('SEQ_LEN:', SEQ_LEN) if debug else None

NUM_EPOCHS = 100  # SLTHP.num_epochs
LEARNING_RATE = 1e-3  # SLTHP.learning_rate
WEIGHT_DECAY = 1e-5  # SLTHP.weight_decay
CLIP = 0.5  # SLTHP.clip

NUM_ITERS = 100  # 100

# use too many Files may cause the following problem: 
# ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
FILE_SIZE = 25  # 100

EVAL_INTERFEVL = 200
EVAL_NUM = 50

# set random seed
# is is actually effective
torch.manual_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)


def getReplayData(path, replay_files, from_index=0, end_index=None):
    td_list = []
    for i, replay_file in enumerate(tqdm(replay_files)):
        try:
            replay_path = path + replay_file
            print('replay_path:', replay_path) if debug else None

            do_write = False
            if i >= from_index:
                if end_index is None:
                    do_write = True
                elif end_index is not None and i < end_index:
                    do_write = True

            if not do_write:
                continue 

            features, labels = torch.load(replay_path)
            print('features.shape:', features.shape) if debug else None
            print('labels.shape::', labels.shape) if debug else None

            td_list.append(ReplayTensorDataset(features, labels))

        except Exception as e:
            traceback.print_exc() 

    return td_list   


def main_worker(device):

    print('==> Making model..')
    net = ArchModel()

    if RESTORE:
        # use state dict to restore
        net.load_state_dict(torch.load(RESTORE_PATH, map_location=device), strict=False)

    net = net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')

    replay_files = os.listdir(PATH)
    print('length of replay_files:', len(replay_files)) if debug else None
    replay_files.sort()

    train_list = getReplayData(PATH, replay_files, from_index=TRAIN_FROM, end_index=TRAIN_FROM + TRAIN_NUM)
    val_list = getReplayData(PATH, replay_files, from_index=VAL_FROM, end_index=VAL_FROM + VAL_NUM)

    train_set = ConcatDataset(train_list)
    val_set = ConcatDataset(val_list)

    print('len(train_set)', len(train_set))
    print('len(val_set)', len(val_set))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=False)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=False)   

    print('len(train_loader)', len(train_loader))
    print('len(val_loader)', len(val_loader))

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(net, optimizer, train_set, train_loader, device, val_set, val_loader)


def train(net, optimizer, train_set, train_loader, device, val_set, val_loader=None):

    now = datetime.datetime.now()
    summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(summary_path)

    # model path
    SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))

    epoch_start = time.time()
    batch_iter = 0

    for epoch in range(NUM_EPOCHS):

        loss_sum = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            # put model in train mode
            net.train()
            start = time.time()

            feature_tensor = features.to(device).float()
            labels_tensor = labels.to(device).float()
            del features, labels

            loss, loss_list, \
                acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                           labels_tensor, net, 
                                                           decrease_smart_opertaion=True,
                                                           return_important=True,
                                                           only_consider_small=False,
                                                           include_location_accuracy=True,
                                                           include_selected_units_accuracy=True)
            del feature_tensor, labels_tensor

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            # add a grad clip
            parameters = [p for p in net.parameters() if p is not None and p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, CLIP)

            loss_value = float(loss.item())

            loss_sum += loss_value
            del loss

            action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)
            move_camera_accuracy = acc_num_list[2] / (acc_num_list[3] + 1e-9)
            non_camera_accuracy = acc_num_list[4] / (acc_num_list[5] + 1e-9)
            short_important_accuracy = acc_num_list[6] / (acc_num_list[7] + 1e-9)

            location_accuracy = acc_num_list[8] / (acc_num_list[9] + 1e-9)
            location_distance = acc_num_list[11] / (acc_num_list[9] + 1e-9)

            selected_units_accuracy = acc_num_list[12] / (acc_num_list[13] + 1e-9)
            selected_units_coverage = acc_num_list[14] / (acc_num_list[13] + 1e-9)

            batch_time = time.time() - start

            batch_iter += 1
            print('batch_iter', batch_iter)

            if batch_iter % EVAL_INTERFEVL == 0:
                print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                    batch_iter, epoch, loss_sum / (batch_iter + 1), action_accuracy, batch_time))

                # torch.save(net, SAVE_PATH + "" + ".pkl")
                # we use new save ways, only save the state_dict, and the extension changes to pt
                torch.save(net.state_dict(), SAVE_PATH + "" + ".pth")

            write(writer, loss_value, loss_list, action_accuracy, 
                  move_camera_accuracy, non_camera_accuracy, short_important_accuracy,
                  location_accuracy, location_distance, selected_units_accuracy,
                  selected_units_coverage,
                  batch_iter)

            gc.collect()

        if True:
            print('eval begin')
            val_loss, val_acc = eval(net, val_set, val_loader, device)
            print('eval end')

            print("Val loss: {:.6f}.".format(val_loss))
            writer.add_scalar('Val/Loss', val_loss, batch_iter)

            # for accuracy of actions in val
            print("Val action acc: {:.6f}.".format(val_acc[0]))
            writer.add_scalar('Val/action Acc', val_acc[0], batch_iter)
            print("Val move_camera acc: {:.6f}.".format(val_acc[1]))
            writer.add_scalar('Val/move_camera Acc', val_acc[1], batch_iter)
            print("Val non_camera acc: {:.6f}.".format(val_acc[2]))
            writer.add_scalar('Val/non_camera Acc', val_acc[2], batch_iter)
            print("Val short_important acc: {:.6f}.".format(val_acc[3]))
            writer.add_scalar('Val/short_important Acc', val_acc[3], batch_iter)
            print("Val location_accuracy acc: {:.6f}.".format(val_acc[4]))
            writer.add_scalar('Val/location_accuracy Acc', val_acc[4], batch_iter)
            print("Val location_distance acc: {:.6f}.".format(val_acc[5]))
            writer.add_scalar('Val/location_distance Acc', val_acc[5], batch_iter)
            print("Val selected_units_accuracy acc: {:.6f}.".format(val_acc[6]))
            writer.add_scalar('Val/selected_units_accuracy Acc', val_acc[6], batch_iter)
            print("Val selected_units_coverage acc: {:.6f}.".format(val_acc[7]))
            writer.add_scalar('Val/selected_units_coverage Acc', val_acc[7], batch_iter)
            del val_loss, val_acc

            gc.collect()

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


def eval(model, val_set, val_loader, device):
    model.eval()

    # should we use it ？
    # model.core.lstm.train()

    loss_sum = 0.0
    i = 0

    action_acc_num = 0.
    action_all_num = 0.

    move_camera_action_acc_num = 0.
    move_camera_action_all_num = 0.

    non_camera_action_acc_num = 0.
    non_camera_action_all_num = 0.

    short_important_action_acc_num = 0.
    short_important_action_all_num = 0.

    location_acc_num = 0.
    location_effect_num = 0.
    location_dist = 0.

    selected_units_correct_num, selected_units_gt_num, selected_units_pred_num = 0, 0, 0

    for i, (features, labels) in enumerate(val_loader):

        if False and i > EVAL_NUM:
            break

        feature_tensor = features.to(device).float()
        labels_tensor = labels.to(device).float()
        del features, labels

        with torch.no_grad():
            loss, _, acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                                labels_tensor, model, 
                                                                decrease_smart_opertaion=True,
                                                                return_important=True,
                                                                only_consider_small=False,
                                                                include_location_accuracy=True,
                                                                include_selected_units_accuracy=True)
            del feature_tensor, labels_tensor

        print('eval i', i, 'loss', loss) if debug else None

        loss_sum += loss.item()

        action_acc_num += acc_num_list[0]
        action_all_num += acc_num_list[1]

        move_camera_action_acc_num += acc_num_list[2]
        move_camera_action_all_num += acc_num_list[3]

        non_camera_action_acc_num += acc_num_list[4]
        non_camera_action_all_num += acc_num_list[5]

        short_important_action_acc_num += acc_num_list[6]
        short_important_action_all_num += acc_num_list[7]

        location_acc_num += acc_num_list[8]
        location_effect_num += acc_num_list[9]
        location_dist += acc_num_list[11]

        selected_units_correct_num += acc_num_list[12]
        selected_units_gt_num += acc_num_list[13]
        selected_units_pred_num += acc_num_list[14]

        del loss, acc_num_list

        gc.collect()

    val_loss = loss_sum / (i + 1e-9)

    action_accuracy = action_acc_num / (action_all_num + 1e-9)
    move_camera_accuracy = move_camera_action_acc_num / (move_camera_action_all_num + 1e-9)
    non_camera_accuracy = non_camera_action_acc_num / (non_camera_action_all_num + 1e-9)
    short_important_accuracy = short_important_action_acc_num / (short_important_action_all_num + 1e-9)

    location_accuracy = location_acc_num / (location_effect_num + 1e-9)
    location_distance = location_dist / (location_effect_num + 1e-9)

    selected_units_accuracy = selected_units_correct_num / (selected_units_gt_num + 1e-9)
    selected_units_coverage = selected_units_pred_num / (selected_units_gt_num + 1e-9)

    return val_loss, [action_accuracy, move_camera_accuracy, non_camera_accuracy, 
                      short_important_accuracy, location_accuracy, location_distance,
                      selected_units_accuracy, selected_units_coverage]


def write(writer, loss, loss_list, action_accuracy, move_camera_accuracy, 
          non_camera_accuracy, short_important_accuracy, 
          location_accuracy, location_distance, 
          selected_units_accuracy, selected_units_coverage,
          batch_iter):

    print("One batch loss: {:.6f}.".format(loss))
    writer.add_scalar('OneBatch/Loss', loss, batch_iter)

    if True:
        print("One batch action_type_loss loss: {:.6f}.".format(loss_list[0]))
        writer.add_scalar('OneBatch/action_type_loss', loss_list[0], batch_iter)

        print("One batch delay_loss loss: {:.6f}.".format(loss_list[1]))
        writer.add_scalar('OneBatch/delay_loss', loss_list[1], batch_iter)

        print("One batch queue_loss loss: {:.6f}.".format(loss_list[2]))
        writer.add_scalar('OneBatch/queue_loss', loss_list[2], batch_iter)

        print("One batch units_loss loss: {:.6f}.".format(loss_list[3]))
        writer.add_scalar('OneBatch/units_loss', loss_list[3], batch_iter)

        print("One batch target_unit_loss loss: {:.6f}.".format(loss_list[4]))
        writer.add_scalar('OneBatch/target_unit_loss', loss_list[4], batch_iter)

        print("One batch target_location_loss loss: {:.6f}.".format(loss_list[5]))
        writer.add_scalar('OneBatch/target_location_loss', loss_list[5], batch_iter)

        print("One batch action_accuracy: {:.6f}.".format(action_accuracy))
        writer.add_scalar('OneBatch/action_accuracy', action_accuracy, batch_iter)

        print("One batch move_camera_accuracy: {:.6f}.".format(move_camera_accuracy))
        writer.add_scalar('OneBatch/move_camera_accuracy', move_camera_accuracy, batch_iter)

        print("One batch non_camera_accuracy: {:.6f}.".format(non_camera_accuracy))
        writer.add_scalar('OneBatch/non_camera_accuracy', non_camera_accuracy, batch_iter)

        print("One batch short_important_accuracy: {:.6f}.".format(short_important_accuracy))
        writer.add_scalar('OneBatch/short_important_accuracy', short_important_accuracy, batch_iter)

        print("One batch location_accuracy: {:.6f}.".format(location_accuracy))
        writer.add_scalar('OneBatch/location_accuracy', location_accuracy, batch_iter)

        print("One batch location_distance: {:.6f}.".format(location_distance))
        writer.add_scalar('OneBatch/location_distance', location_distance, batch_iter)

        print("One batch selected_units_accuracy: {:.6f}.".format(selected_units_accuracy))
        writer.add_scalar('OneBatch/selected_units_accuracy', selected_units_accuracy, batch_iter)

        print("One batch selected_units_coverage: {:.6f}.".format(selected_units_coverage))
        writer.add_scalar('OneBatch/selected_units_coverage', selected_units_coverage, batch_iter)


def test(on_server):
    # gpu setting
    ON_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")

    if ON_GPU:
        if torch.backends.cudnn.is_available():
            print('cudnn available')
            print('cudnn version', torch.backends.cudnn.version())
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    main_worker(DEVICE)
