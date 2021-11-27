#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train （multi-gpu） from the replay files through python pickle file"

# reference most from https://github.com/dnddnjs/pytorch-multigpu/blob/master/dist_parallel/train.py

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

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, RMSprop

# for multi-process gpu training
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.arch_model import ArchModel

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_loss_multi_gpu as Loss
from alphastarmini.core.sl.dataset_pickle import OneReplayDataset, AllReplayDataset, FullDataset
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.utils import load_latest_model, initial_model_state_dict
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP

__author__ = "Ruo-Ze Liu"

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="./data/replay_data/", help="The path where data stored")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument("-r", "--restore", action="store_true", default=True, help="whether to restore model or not")
parser.add_argument('--num_workers', type=int, default=2, help='')

# multi-gpu parameters
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8818', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

args = parser.parse_args()

# training paramerters
PATH = args.path
MODEL = args.model
RESTORE = args.restore

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
RESTORE_PATH = MODEL_PATH + 'sl_21-11-16_13-30-24.pth'

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

EVAL_INTERFEVL = 5
EVAL_NUM = 50

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
    net = ArchModel()

    if RESTORE:
        # use state dict to restore
        net.load_state_dict(torch.load(RESTORE_PATH, map_location=torch.device(args.rank)), strict=False)

    torch.cuda.set_device(args.gpu)
    net = net.cuda(args.gpu)

    #args.batch_size = int(args.batch_size / ngpus_per_node)

    args.num_workers = int(args.num_workers / ngpus_per_node)
    net = DDP(net, device_ids=[args.gpu], find_unused_parameters=True)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')

    train_set = FullDataset(replay_data_path=PATH, max_file_size=FILE_SIZE, shuffle=False)
    val_set = FullDataset(replay_data_path=PATH, val=True, max_file_size=FILE_SIZE, shuffle=False)

    print('len(train_set)', len(train_set))
    print('len(val_set)', len(val_set))

    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(val_set)

    # when you face this error：
    # RuntimeError: DataLoader worker (pid 53617) is killed by signal: Bus error. It is possible that dataloader's 
    # workers are out of shared memory. Please try to raise your shared memory limit.
    # You can try to set num_workers=0, it may slove it.
    # https://github.com/pytorch/pytorch/issues/5040
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=(train_sampler is None), 
                              num_workers=args.num_workers, sampler=train_sampler)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=(val_sampler is None), 
                            num_workers=args.num_workers, sampler=val_sampler)   

    print('len(train_loader)', len(train_loader))
    print('len(val_loader)', len(val_loader))

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = None
    train(net, criterion, optimizer, train_set, train_loader, args.gpu, args.rank, val_set, val_loader)


def train(net, criterion, optimizer, train_set, train_loader, device, rank, val_set, val_loader=None):
    if rank == 0:
        now = datetime.datetime.now()
        summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        writer = SummaryWriter(summary_path)

        # model path
        SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))
    dist.barrier()

    epoch_start = time.time()
    batch_iter = 0

    for epoch in range(NUM_EPOCHS):
        # let all processes sync up before starting with a new epoch of training
        # dist.barrier()
        loss_sum = 0

        for batch_idx, traj in enumerate(train_loader):
            # put model in train mode
            net.train()
            start = time.time()

            traj_tensor = traj.cuda(device).float()
            del traj

            loss, loss_list, acc_num_list = Loss.get_sl_loss(traj_tensor, net)
            del traj_tensor

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

            batch_time = time.time() - start

            batch_iter += 1
            print('batch_iter', batch_iter)

            if batch_iter % EVAL_INTERFEVL == 0:
                print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                    batch_iter, epoch, loss_sum / (batch_iter + 1), action_accuracy, batch_time))

                if rank == 0:    
                    # torch.save(net, SAVE_PATH + "" + ".pkl")
                    # we use new save ways, only save the state_dict, and the extension changes to pt
                    torch.save(net.state_dict(), SAVE_PATH + "" + ".pth")

                dist.barrier()

            if rank == 0:
                write(writer, loss_value, loss_list, action_accuracy, move_camera_accuracy, non_camera_accuracy, batch_iter)
            dist.barrier()

            gc.collect()

        if False:
            print('eval begin')
            val_loss, val_acc = eval(net, val_set, val_loader, device)
            print('eval end')

            if rank == 0:
                print("Val loss: {:.6f}.".format(val_loss))
                writer.add_scalar('Val/Loss', val_loss, batch_iter)

                # for accuracy of actions in val
                print("Val action acc: {:.6f}.".format(val_acc[0]))
                writer.add_scalar('Val/action Acc', val_acc[0], batch_iter)
                print("Val move_camera acc: {:.6f}.".format(val_acc[1]))
                writer.add_scalar('Val/move_camera Acc', val_acc[1], batch_iter)
                print("Val non_camera acc: {:.6f}.".format(val_acc[2]))
                writer.add_scalar('Val/non_camera Acc', val_acc[2], batch_iter)
            dist.barrier()  

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


def eval(model, val_set, val_loader, device):
    model.eval()

    loss_sum = 0.0
    i = 0

    action_acc_num = 0.
    action_all_num = 0.

    move_camera_action_acc_num = 0.
    move_camera_action_all_num = 0.

    non_camera_action_acc_num = 0.
    non_camera_action_all_num = 0.

    for traj in val_loader:

        if i > EVAL_NUM:
            break

        traj_tensor = traj.cuda(device).float()
        del traj

        loss, _, acc_num_list = Loss.get_sl_loss(traj_tensor, model)
        del traj_tensor

        print('eval i', i, 'loss', loss) if debug else None

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


def write(writer, loss, loss_list, action_accuracy, move_camera_accuracy, non_camera_accuracy, batch_iter):

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


def test(on_server):
    # initial for gpu settings
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node) if debug else None

    args.world_size = ngpus_per_node * args.world_size
    print('args.world_size', args.world_size) if debug else None

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
