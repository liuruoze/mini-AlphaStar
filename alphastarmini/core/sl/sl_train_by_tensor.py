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
from torch.optim.lr_scheduler import StepLR

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
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

import param as P

__author__ = "Ruo-Ze Liu"

debug = True

parser = argparse.ArgumentParser()
parser.add_argument("-p1", "--path1", default="./data/replay_data_tensor_new_small/", help="The path where data stored")
parser.add_argument("-p2", "--path2", default="./data/replay_data_tensor_new_small_AR/", help="The path where data stored")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument("-c", "--clip", action="store_true", default=False, help="whether to use clipping")
parser.add_argument('--num_workers', type=int, default=2, help='')


args = parser.parse_args()

# training paramerters
if SCHP.map_name == 'Simple64':
    PATH = args.path1
elif SCHP.map_name == 'AbyssalReef':
    PATH = args.path2
else:
    raise Exception

MODEL = args.model
RESTORE = args.restore
CLIP = args.clip
NUM_WORKERS = args.num_workers

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

MODEL_PATH_TRAIN = "./model_train/"
if not os.path.exists(MODEL_PATH_TRAIN):
    os.mkdir(MODEL_PATH_TRAIN)

RESTORE_NAME = 'sl_21-12-21_09-11-12'
RESTORE_PATH = MODEL_PATH + RESTORE_NAME + '.pth' 
RESTORE_PATH_TRAIN = MODEL_PATH_TRAIN + RESTORE_NAME + '.pkl'

SAVE_STATE_DICT = True
SAVE_ALL_PKL = False
SAVE_CHECKPOINT = True

LOAD_STATE_DICT = False
LOAD_ALL_PKL = False
LOAD_CHECKPOINT = True

SIMPLE_TEST = not P.on_server
if SIMPLE_TEST:
    TRAIN_FROM = 0
    TRAIN_NUM = 1

    VAL_FROM = 0
    VAL_NUM = 1
else:
    TRAIN_FROM = 0  # 20
    TRAIN_NUM = 90  # 60

    VAL_FROM = 0
    VAL_NUM = 5

# hyper paramerters
# use the same as in RL
# BATCH_SIZE = AHP.batch_size
# SEQ_LEN = AHP.sequence_length

# important: use larger batch_size and smaller seq_len in SL!
BATCH_SIZE = 3 * AHP.batch_size
SEQ_LEN = int(AHP.sequence_length * 0.5)

print('BATCH_SIZE:', BATCH_SIZE) if debug else None
print('SEQ_LEN:', SEQ_LEN) if debug else None

NUM_EPOCHS = 10 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

CLIP_VALUE = 0.5  # SLTHP.clip
STEP_SIZE = 50
GAMMA = 0.2

torch.manual_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)


def getReplayData(path, replay_files, from_index=0, end_index=None):
    td_list = []
    for i, replay_file in enumerate(tqdm(replay_files)):
        try:
            replay_path = path + replay_file
            print('replay_path:', replay_path) if 1 else None

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

            td_list.append(ReplayTensorDataset(features, labels, seq_len=SEQ_LEN))

        except Exception as e:
            traceback.print_exc() 

    return td_list   


def main_worker(device):
    print('==> Making model..')
    net = ArchModel()
    checkpoint = None
    if RESTORE:
        if LOAD_STATE_DICT:
            # use state dict to restore
            net.load_state_dict(torch.load(RESTORE_PATH, map_location=device), strict=False)

        if LOAD_ALL_PKL:
            # use all to restore
            net = torch.load(RESTORE_PATH_TRAIN, map_location=device)

        if LOAD_CHECKPOINT:
            # use checkpoint to restore
            checkpoint = torch.load(RESTORE_PATH_TRAIN, map_location=device)
            net.load_state_dict(checkpoint['model'], strict=False)
    net = net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Making optimizer and scheduler..')

    optimizer, scheduler = None, None
    batch_iter, epoch = 0, 0

    if RESTORE and LOAD_CHECKPOINT:
        # use checkpoint to restore other
        optimizer = Adam(net.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler = StepLR(optimizer, step_size=STEP_SIZE)
        scheduler.load_state_dict(checkpoint['scheduler'])

        batch_iter = checkpoint['batch_iter']
        print('batch_iter is', batch_iter)
        epoch = checkpoint['epoch']
        print('epoch is', epoch)

        ckpt = torch.load(RESTORE_PATH_TRAIN)
        np.random.set_state(ckpt['numpy_random_state'])
        torch.random.set_rng_state(ckpt['torch_random_state'])
    else:
        optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print('==> Preparing data..')

    replay_files = os.listdir(PATH)
    print('length of replay_files:', len(replay_files)) if debug else None
    replay_files.sort()

    train_list = getReplayData(PATH, replay_files, from_index=TRAIN_FROM, end_index=TRAIN_FROM + TRAIN_NUM)
    val_list = getReplayData(PATH, replay_files, from_index=VAL_FROM, end_index=VAL_FROM + VAL_NUM)

    print('len(train_list)', len(train_list)) if debug else None
    print('len(val_list)', len(val_list)) if debug else None

    train_set = ConcatDataset(train_list)
    val_set = ConcatDataset(val_list)

    print('len(train_set)', len(train_set)) if debug else None
    print('len(val_set)', len(val_set)) if debug else None

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=False)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=False)   

    print('len(train_loader)', len(train_loader)) if debug else None
    print('len(val_loader)', len(val_loader)) if debug else None

    train(net, optimizer, scheduler, train_set, train_loader, device, 
          val_set, batch_iter, epoch, val_loader)


def train(net, optimizer, scheduler, train_set, train_loader, device, 
          val_set, batch_iter, epoch, val_loader=None):

    now = datetime.datetime.now()
    summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(summary_path)

    time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time_str)
    SAVE_PATH_TRAIN = os.path.join(MODEL_PATH_TRAIN, MODEL + "_" + time_str)

    epoch_start = time.time()

    for ep in range(NUM_EPOCHS):
        loss_sum = 0
        epoch += 1 

        # put model in train mode
        net.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
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
                                                           train=True)
            del feature_tensor, labels_tensor

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            # add a grad clip
            if CLIP:
                parameters = [p for p in net.parameters() if p is not None and p.requires_grad]
                torch.nn.utils.clip_grad_norm_(parameters, CLIP_VALUE)

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
            selected_units_type_right = acc_num_list[14] / (acc_num_list[15] + 1e-9)
            selected_units_num_right = acc_num_list[16] / (acc_num_list[17] + 1e-9)

            target_unit_accuracy = acc_num_list[18] / (acc_num_list[19] + 1e-9)

            batch_time = time.time() - start

            batch_iter += 1
            print('batch_iter', batch_iter)

            write(writer, loss_value, loss_list, action_accuracy, 
                  move_camera_accuracy, non_camera_accuracy, short_important_accuracy,
                  location_accuracy, location_distance, selected_units_accuracy,
                  selected_units_type_right, selected_units_num_right, target_unit_accuracy,
                  batch_iter)

            gc.collect()

            print('Batch/Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_iter, epoch, loss_value, action_accuracy, batch_time))

        if SAVE_STATE_DICT:
            save_path = SAVE_PATH + ".pth"
            print('Save model state_dict to', save_path)
            torch.save(net.state_dict(), save_path)

        if SAVE_ALL_PKL:
            save_path = SAVE_PATH_TRAIN + ".pkl"
            print('Save model all to', save_path)
            torch.save(net, save_path)

        if SAVE_CHECKPOINT:
            save_path = SAVE_PATH_TRAIN + ".pkl"
            save_dict = {'batch_iter': batch_iter,
                         'epoch': epoch,
                         'model': net.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'loss': loss_value,
                         'numpy_random_state': np.random.get_state(),
                         'torch_random_state': torch.random.get_rng_state(),
                         }

            print('Save model checkpoint to', save_path)
            torch.save(save_dict, save_path)

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
            print("Val selected_units_type_right acc: {:.6f}.".format(val_acc[7]))
            writer.add_scalar('Val/selected_units_type_right Acc', val_acc[7], batch_iter)
            print("Val selected_units_num_right acc: {:.6f}.".format(val_acc[8]))
            writer.add_scalar('Val/selected_units_num_right Acc', val_acc[8], batch_iter)
            print("Val target_unit_accuracy acc: {:.6f}.".format(val_acc[9]))
            writer.add_scalar('Val/target_unit_accuracy Acc', val_acc[9], batch_iter)            

            del val_loss, val_acc

        gc.collect()

        scheduler.step()

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

    selected_units_correct_num, selected_units_gt_num = 0, 0
    selected_units_type_correct_num, selected_units_pred_num = 0, 0
    selected_units_nums_equal, selected_units_batch_size = 0, 0

    target_unit_correct_num, target_unit_all_num = 0, 0

    for i, (features, labels) in enumerate(val_loader):

        feature_tensor = features.to(device).float()
        labels_tensor = labels.to(device).float()
        del features, labels

        with torch.no_grad():
            loss, _, acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                                labels_tensor, model, 
                                                                decrease_smart_opertaion=True,
                                                                return_important=True,
                                                                only_consider_small=False,
                                                                train=False)
            del feature_tensor, labels_tensor

        print('eval i', i, 'loss', loss) if 1 else None

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

        selected_units_type_correct_num += acc_num_list[14]
        selected_units_pred_num += acc_num_list[15]

        selected_units_nums_equal += acc_num_list[16]
        selected_units_batch_size += acc_num_list[17]

        target_unit_correct_num += acc_num_list[18]
        target_unit_all_num += acc_num_list[19]

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
    selected_units_type_right = selected_units_type_correct_num / (selected_units_pred_num + 1e-9)
    selected_units_num_right = selected_units_nums_equal / (selected_units_batch_size + 1e-9)

    target_unit_accuracy = target_unit_correct_num / (target_unit_all_num + 1e-9)

    return val_loss, [action_accuracy, move_camera_accuracy, non_camera_accuracy, 
                      short_important_accuracy, location_accuracy, location_distance,
                      selected_units_accuracy, selected_units_type_right, selected_units_num_right, target_unit_accuracy]


def write(writer, loss, loss_list, action_accuracy, move_camera_accuracy, 
          non_camera_accuracy, short_important_accuracy, 
          location_accuracy, location_distance, 
          selected_units_accuracy, selected_units_type_right, selected_units_num_right, target_unit_accuracy,
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

        print("One batch selected_units_type_right: {:.6f}.".format(selected_units_type_right))
        writer.add_scalar('OneBatch/selected_units_type_right', selected_units_type_right, batch_iter)

        print("One batch selected_units_num_right: {:.6f}.".format(selected_units_num_right))
        writer.add_scalar('OneBatch/selected_units_num_right', selected_units_num_right, batch_iter)

        print("One batch target_unit_accuracy: {:.6f}.".format(target_unit_accuracy))
        writer.add_scalar('OneBatch/target_unit_accuracy', target_unit_accuracy, batch_iter)


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
