import os
USED_DEVICES = "0,1,2,3,4,5,6,7"  # if your want to use CPU in a server with GPU, change "0" to "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import numpy as np
import random

seed = 0  # use the fixed seed for the full program

# must use
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

# optional use
# torch.set_deterministic(True)
# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(seed)

import alphastarmini

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

import param as P

if __name__ == '__main__':
    print("run init")

    # ------------------------

    # # 1. we transform the replays to pickle
    # from alphastarmini.core.sl import transform_replay_data
    # transform_replay_data.test(on_server=P.on_server)

    # 2. we use tensor to do supervised learning
    from alphastarmini.core.sl import sl_train_by_tensor
    sl_train_by_tensor.test(on_server=P.on_server)

    # # 3. we use RL environment to evaluate SL model
    # from alphastarmini.core.rl import rl_eval_sl
    # rl_eval_sl.test(on_server=P.on_server)

    # # 4. we use SL model to do reinforcement learning against computer
    # from alphastarmini.core.rl import rl_vs_inner_bot_mp
    # rl_vs_inner_bot_mp.test(on_server=P.on_server, replay_path=P.replay_path)

    # # 5. we use RL environment to evaluate SL model
    # from alphastarmini.core.rl import rl_eval_rl
    # rl_eval_rl.test(on_server=P.on_server)

    # # 6. we use SL model and replays to do reinforcement learning
    # from alphastarmini.core.rl import rl_train_with_replay
    # rl_train_with_replay.test(on_server=P.on_server, replay_path=P.replay_path)

    # ------------------------
    #
    # below is optional to use

    # transform pickles data to tensor data for supervised learning
    # from alphastarmini.core.sl import load_pickle
    # load_pickle.test(on_server=False)

    # we can use pickle to do supervised learning
    # from alphastarmini.core.sl import sl_train_by_pickle
    # sl_train_by_pickle.test(on_server=P.on_server)

    print('run over')
