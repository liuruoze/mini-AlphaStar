#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RL 训练的独立入口（rl_vs_inner_bot_mp，多进程 A3C 打内置 bot）。

    MAS_ON_SERVER=0 CUDA_VISIBLE_DEVICES=0 python run_rl.py

不需要 replay 也能跑（NO_REPLAY_LEARN=True，只用胜负/得分奖励）。

关于 FLAGS：pysc2 3.0 的 run_configs.get() 会读 FLAGS.sc2_run_config，
而 rl_vs_inner_bot_mp 自己没解析 flags；同时它用 mp.spawn 起子进程，
子进程会重新 import 本模块，重复 FLAGS(sys.argv) 可能出错。
所以这里解析失败就退回 mark_as_parsed()，让 FLAGS 有默认值即可。
"""

import os
import sys

# 多进程 RL 必设（与 rl_vs_inner_bot_mp.py 顶部同因）：
# nproc=128 时每个进程的 BLAS/OMP 默认开 128 个线程，8 个 actor = 1024 线程抢 128 核，
# 实测把 preprocess 从 10 ms 拖到 200 ms、吞吐从 85 掉到 14 step/s。
# 这里再设一次，确保不依赖后续 import 顺序。
for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

from absl import flags

FLAGS = flags.FLAGS
try:
    FLAGS(sys.argv)
except Exception as e:
    print("FLAGS parse skipped:", type(e).__name__, e)
FLAGS.mark_as_parsed()

os.makedirs("./log", exist_ok=True)
os.makedirs("./model", exist_ok=True)

import param as P

ON_SERVER = os.environ.get("MAS_ON_SERVER", "0") == "1"
P.on_server = ON_SERVER

from alphastarmini.core.rl import rl_vs_inner_bot_mp as RL

if __name__ == '__main__':
    print("== rl training (vs built-in bot) ==")
    print("on_server:", ON_SERVER, "| PARALLEL:", RL.PARALLEL, "| MAX_EPISODES:", RL.MAX_EPISODES)
    print("torch.cuda.is_available():", __import__("torch").cuda.is_available())
    RL.test(on_server=ON_SERVER, replay_path=P.replay_path)
    print("== done ==")
