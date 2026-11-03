#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Replay -> tensor 的独立入口。

不改动 run.py（保持原仓库可对照），用法：

    # 冒烟测试：2 个 replay，每个最多 1000 帧
    MAS_ON_SERVER=0 python run_transform.py \
        --no_server_replay_path=data/Replays/filtered_replays_1/

    # 全量：遍历 filtered_replays_1 下所有 replay
    MAS_ON_SERVER=1 python run_transform.py

为什么要在 import transform 之前设 param.on_server：
transform_replay_data 在模块级用 `SIMPLE_TEST = not P.on_server` 决定处理的 replay 数量
（False -> 2 个，True -> 90 个），导入后再改就晚了。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import param as P

ON_SERVER = os.environ.get("MAS_ON_SERVER", "0") == "1"
P.on_server = ON_SERVER

# pysc2 的 run_config.replay_data() 会把非绝对路径拼到 $SC2PATH/Replays/ 下面，
# 于是相对路径 "data/Replays/xxx" 会变成 "$SC2PATH/Replays/data/Replays/xxx" 而找不到文件。
# 这里统一转成绝对路径（cwd 需为项目根目录）。
def _abs_dir(path):
    """转绝对路径并保留尾部斜杠。

    transform_replay_data 里是 `replay_path = REPLAY_PATH + replay_file` 的字符串拼接，
    依赖尾部斜杠；os.path.abspath 会把斜杠吃掉，导致路径粘成一个错误的名字。
    """
    path = os.path.abspath(path)
    if not path.endswith(os.sep):
        path += os.sep
    return path


P.replay_path = _abs_dir(P.replay_path)

from alphastarmini.core.sl import transform_replay_data as TRD

TRD.FLAGS.no_server_replay_path = _abs_dir(TRD.FLAGS.no_server_replay_path)

# 简易模式默认只跑 2 个 replay、每个最多 1000 个 game step（约 45 秒游戏时间），
# 会把一局 replay 从中间截断。做参数对比实验时需要放开，用环境变量覆盖。
if os.environ.get("MAS_MAX_STEPS"):
    TRD.SMALL_MAX_STEPS = int(os.environ["MAS_MAX_STEPS"])
if os.environ.get("MAS_DATA_NUM"):
    TRD.DATA_NUM = int(os.environ["MAS_DATA_NUM"])
if os.environ.get("MAS_DATA_FROM"):
    TRD.DATA_FROM = int(os.environ["MAS_DATA_FROM"])

if __name__ == '__main__':
    print("== transform replay -> tensor ==")
    print("on_server:", ON_SERVER)
    print("replay_path(P):", P.replay_path)
    print("replay_path(flag):", TRD.FLAGS.no_server_replay_path)
    print("max_steps_of_replay:", TRD.FLAGS.max_steps_of_replay if ON_SERVER else TRD.SMALL_MAX_STEPS)
    print("data range: [%d, %d)" % (TRD.DATA_FROM, TRD.DATA_FROM + TRD.DATA_NUM))
    print("keep rate: no_op=%.3f camera=%.3f Smart_pt=%.2f Smart_unit=%.2f Harvest=%.2f Attack=%.2f"
          % (TRD.FLAGS.no_op_threshold, TRD.FLAGS.move_camera_threshold,
             TRD.FLAGS.Smart_pt_threshold, TRD.FLAGS.Smart_unit_threshold,
             TRD.FLAGS.Harvest_Gather_unit_threshold, TRD.FLAGS.Attack_pt_threshold))
    print("argv:", sys.argv)
    TRD.test(on_server=ON_SERVER)
    print("== done ==")
