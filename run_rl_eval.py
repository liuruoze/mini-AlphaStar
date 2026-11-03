#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用训练好的 SL 模型在 SC2 里跑一局（RL 评估），用于验证 RL 链路。

    MAS_ON_SERVER=0 CUDA_VISIBLE_DEVICES=0 python run_rl_eval.py

注意 rl_eval_sl 的 SIMPLE_TEST 分支只在 on_server=False 时可用，
on_server=True 会直接 raise NotImplementedError，所以这里固定传 False。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import param as P

P.on_server = False

# pysc2 3.0 的 run_configs.get() 会读 FLAGS.sc2_run_config，必须先解析 absl flags；
# rl_eval_sl.py 自己没有调用 FLAGS(...)，所以在这里补上。
from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)

# rl_eval_sl 会往 ./outputs/eval_sl.txt 追加结果，目录不存在会抛 FileNotFoundError
os.makedirs("./outputs", exist_ok=True)

from alphastarmini.core.rl import rl_eval_sl

if __name__ == '__main__':
    print("== rl eval (SL model vs built-in bot) ==")
    print("torch.cuda.is_available():", __import__("torch").cuda.is_available())
    rl_eval_sl.test(on_server=False)
    print("== done ==")
