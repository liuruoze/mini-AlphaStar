import os
USED_DEVICES = "-1"  # if your want to use CPU in a server with GPU, change "0" to "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import alphastarmini
import torch

from alphastarmini.core.arch import entity_encoder
from alphastarmini.core.arch import scalar_encoder
from alphastarmini.core.arch import spatial_encoder
from alphastarmini.core.arch import arch_model
from alphastarmini.core.arch import action_type_head
from alphastarmini.core.arch import selected_units_head
from alphastarmini.core.arch import target_unit_head
from alphastarmini.core.arch import delay_head
from alphastarmini.core.arch import queue_head
from alphastarmini.core.arch import location_head
from alphastarmini.core.arch import agent
from alphastarmini.core.arch import baseline

from alphastarmini.core.sl import load_pickle

from alphastarmini.core.rl import action
from alphastarmini.core.rl import env_utils
from alphastarmini.core.rl import actor
from alphastarmini.core.rl import against_computer
from alphastarmini.core.rl import pseudo_reward

import param as P

if __name__ == '__main__':
    # if we don't add this line, it may cause running time error while in Windows
    # torch.multiprocessing.freeze_support()

    print("run init")

    # ------------------------

    # 1. we transform the replays to pickle
    from alphastarmini.core.sl import transform_replay_data
    transform_replay_data.test(on_server=P.on_server)

    print('run over')
