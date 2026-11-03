#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""监督学习（SL）的独立入口，不改动 run.py。

    # 冒烟：只读 1 个训练 + 1 个验证 tensor 文件
    MAS_ON_SERVER=0 CUDA_VISIBLE_DEVICES=0 python run_sl.py

    # 全量：训练集 = 文件列表除末尾 5 个以外的全部，验证集 = 末尾 5 个
    #       （两者不重叠；具体区间启动后会打印 "data split: ..."）
    MAS_ON_SERVER=1 CUDA_VISIBLE_DEVICES=0 python run_sl.py

sl_train_by_tensor 用 `SIMPLE_TEST = not P.on_server` 在模块级决定读多少数据，
所以必须在 import 之前设置 param.on_server。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import param as P

ON_SERVER = os.environ.get("MAS_ON_SERVER", "0") == "1"
P.on_server = ON_SERVER

from alphastarmini.core.sl import sl_train_by_tensor as SL

if __name__ == '__main__':
    print("== supervised learning ==")
    print("on_server:", ON_SERVER)
    print("torch.cuda.is_available():", __import__("torch").cuda.is_available())
    print("data path (--path1):", SL.PATH)
    SL.test(on_server=ON_SERVER)
    print("== done ==")
