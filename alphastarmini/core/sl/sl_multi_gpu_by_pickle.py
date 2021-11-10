#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train （multi-gpu） from the replay files through python pickle file"

# reference most from https://github.com/dnddnjs/pytorch-multigpu/blob/master/dist_parallel/train.py

import os

import sys
import time
import traceback
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, RMSprop

# for multi-process gpu training
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

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
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument('--num_workers', type=int, default=4, help='')

# multi-gpu parameters
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8888', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

args = parser.parse_args()

# training paramerters
PATH = args.path
MODEL = args.model

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
FILE_SIZE = 30  # 100

# set random seed
torch.manual_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    agent = Agent()
    net = agent.model
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)

    #args.batch_size = int(args.batch_size / ngpus_per_node)

    args.num_workers = int(args.num_workers / ngpus_per_node)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')

    replays = AllReplayDataset.get_trainable_data(replay_data_path=PATH,
                                                  max_file_size=FILE_SIZE, shuffle=True)
    train_replays = AllReplayDataset.get_training_for_val_data(replays)
    train_set = AllReplayDataset(train_replays)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=args.num_workers, 
                              sampler=train_sampler)
    print('len(train_loader)', len(train_loader))

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = None
    train(net, criterion, optimizer, train_loader, args.gpu, args.rank)


def train(net, criterion, optimizer, train_loader, device, rank):
    if rank == 0:
        now = datetime.datetime.now()
        summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        writer = SummaryWriter(summary_path)

        # model path
        MODEL_PATH = "./model/"
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))

    net.train()

    loss_sum = 0
    epoch_start = time.time()

    batch_iter = 0

    for batch_idx in range(NUM_ITERS):

        traj = next(iter(train_loader))

        start = time.time()
        print('batch_idx', batch_idx)
        print('traj', traj) if debug else None
        traj = traj.cuda(device)
        print('traj', traj) if debug else None
        loss, loss_list, acc_num_list = Loss.get_sl_loss(traj, net)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        # add a grad clip
        parameters = [p for p in net.parameters() if p is not None and p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, CLIP)

        loss_sum += loss.item()

        action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)
        move_camera_accuracy = acc_num_list[2] / (acc_num_list[3] + 1e-9)
        non_camera_accuracy = acc_num_list[4] / (acc_num_list[5] + 1e-9)

        batch_time = time.time() - start

        if batch_idx % 20 == 0:
            print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, NUM_ITERS, loss_sum / (batch_idx + 1), 0, batch_time))

        batch_iter += 1

        print('rank', rank)
        # if rank == 0:
        #    write(writer, loss, loss_list, action_accuracy, move_camera_accuracy, non_camera_accuracy, batch_iter)

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))

    if rank == 0:
        torch.save(net, SAVE_PATH + "" + ".pkl")


def write(writer, loss, loss_list, action_accuracy, move_camera_accuracy, non_camera_accuracy, batch_iter):

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


def test(on_server):
    # initial for gpu settings
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node) if debug else None

    args.world_size = ngpus_per_node * args.world_size
    print('args.world_size', args.world_size) if debug else None

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
