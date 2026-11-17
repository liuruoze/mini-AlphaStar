import os
USED_DEVICES = "1"  # if your want to use CPU in a server with GPU, change "0" to "-1"
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

from alphastarmini.core.rl import action
# ---- 2026-11-16 RL 技术债 D2：三个已不存在的模块不再 import ----
# env_utils / against_computer / mp_rl_eval_rl 在全仓均无对应文件（历史遗留 import）。
# 原样保留会让本脚本在 import 阶段就 ImportError，连"该用哪个入口"的提示都看不到。
# 这里不假装修复：直接给出明确指引，活入口是 run_rl_eval.py（单机）和 mp_rl_eval_sl.py（server）。
from alphastarmini.core.rl import actor
from alphastarmini.core.rl import pseudo_reward

import param as P

if __name__ == '__main__':
    # ---- 2026-11-16 RL 技术债 D2：死入口改为明确指引，不静默 ImportError ----
    # mp_rl_eval_rl 模块在全仓不存在，本入口按现状永远跑不起来。
    # 真正的评测入口：
    #   单机/诊断：python run_rl_eval.py  → rl_eval_sl.test()
    #   server 批量：mp_rl_eval_sl.test()
    # 保留本文件只是兼容旧引用，不再伪装成可用入口。
    raise SystemExit(
        "eval_rl.py 是历史死入口（mp_rl_eval_rl 模块不存在）。\n"
        "请用：python run_rl_eval.py（单机评测）或 mp_rl_eval_sl.test()（server 评测）。")
