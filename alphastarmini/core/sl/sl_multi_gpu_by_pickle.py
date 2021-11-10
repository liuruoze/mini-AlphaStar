#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train （multi-gpu） from the replay files through python pickle file"

# reference most from https://github.com/dnddnjs/pytorch-multigpu/blob/master/dist_parallel/train.py

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
FILE_SIZE = 100  # 100

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
    replays = AllReplayDataset.get_trainable_data(replay_data_path=PATH, agent=agent, 
                                                  max_file_size=10, shuffle=False)

    train_replays = AllReplayDataset.get_training_for_val_data(replays)

    train_set = AllReplayDataset(train_replays)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=args.num_workers, 
                              sampler=train_sampler)

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = None
    train(net, criterion, optimizer, train_loader, args.gpu)


def train(net, criterion, optimizer, train_loader, device):
    net.train()

    loss_sum = 0

    epoch_start = time.time()
    for batch_idx, traj in enumerate(train_loader):
        start = time.time()
        print('batch_idx', batch_idx)
        print('traj', traj)
        traj = traj.cuda(device)
        print('traj', traj)
        loss, _, _ = Loss.get_sl_loss(traj, net)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        loss_sum += loss.item()

        batch_time = time.time() - start

        if batch_idx % 20 == 0:
            print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(train_loader), loss_sum / (batch_idx + 1), 0, batch_time))

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


def test(on_server):
    # initial for gpu settings
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)

    args.world_size = ngpus_per_node * args.world_size
    print('args.world_size', args.world_size)

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
