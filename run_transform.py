#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Replay -> tensor 的独立入口。

不改动 run.py（保持原仓库可对照），用法：

    # 冒烟测试：2 个 replay，每个最多 1000 帧
    MAS_ON_SERVER=0 MAS_DATA_SMOKE=1 python run_transform.py \
        --no_server_replay_path=data/Replays/filtered_replays_1/

    # 全量：默认 90 局，可用 MAS_DATA_NUM 覆盖
    MAS_ON_SERVER=1 python run_transform.py

`MAS_ON_SERVER` 决定 replay 路径与帧上限；处理数量不再随它静默变化。
默认 90 局（训练纪律）；冒烟用 `MAS_DATA_SMOKE=1`（2 局），也可用 `MAS_DATA_NUM` 覆盖。
转换结束会比对实际写出数与预期数，不一致时入口以非零码退出。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import param as P

# ---- 可复跑：转换器内部有 6 处 random.uniform 做阈值丢弃，但原文件没有 seed。
# 不固定种子时同一局两次转换结果不同（实测 1,050 vs 1,019 帧）。
# 默认不 seed（保持历史行为）；采集/重转时显式传 MAS_TRANSFORM_SEED。
import os as _os_r, random as _random_r
if _os_r.environ.get("MAS_TRANSFORM_SEED", "") != "":
    _random_r.seed(int(_os_r.environ["MAS_TRANSFORM_SEED"]))


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


if os.environ.get("MAS_REPLAY_PATH"):
    P.replay_path = _abs_dir(os.environ["MAS_REPLAY_PATH"])
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
    written, expected = TRD.test(on_server=ON_SERVER)
    print("== done ==")
    if written != expected:
        print("[TRANSFORM-ERROR] written=%d expected=%d; refusing to report success"
              % (written, expected))
        sys.exit(1)
