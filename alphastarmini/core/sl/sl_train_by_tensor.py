#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay files through python tensor (pt) file"

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

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR

from tensorboardX import SummaryWriter

from absl import flags
from absl import app
from tqdm import tqdm

from alphastarmini.core.arch.arch_model import ArchModel

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_loss_multi_gpu as Loss
from alphastarmini.core.sl.dataset import ReplayTensorDataset
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.utils import load_latest_model, initial_model_state_dict

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

import param as P

__author__ = "Ruo-Ze Liu"

debug = True

parser = argparse.ArgumentParser()
parser.add_argument("-p1", "--path1", default="./data/replay_data_tensor_new_small/", help="The path where data stored")
parser.add_argument("-p2", "--path2", default="./data/replay_data_tensor_new_small_AR/", help="The path where data stored")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument("-c", "--clip", action="store_true", default=False, help="whether to use clipping")
parser.add_argument('--num_workers', type=int, default=16, help='')


args = parser.parse_args()

# training paramerters
if SCHP.map_name == 'Simple64':
    PATH = args.path1
elif SCHP.map_name == 'AbyssalReef':
    PATH = args.path2
else:
    raise Exception

MODEL = args.model
RESTORE = os.environ.get("MAS_SL_RESTORE", "1" if os.environ.get("MAS_SL_LOAD_STATE", "0") == "1" else "0") == "1"
CLIP = args.clip
NUM_WORKERS = int(os.environ.get("MAS_SL_WORKERS", str(args.num_workers)))
# 数据整份常驻内存（60 文件约 66 GB），fork 多个 worker 有内存翻倍风险；
# 训练时通常用 MAS_SL_WORKERS=0（batch 只是内存切片，不需要 worker 预取）。

MODEL_PATH = os.environ.get("MAS_SL_MODEL_PATH", "./model/")
if not MODEL_PATH.endswith("/"):
    MODEL_PATH = MODEL_PATH + "/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

MODEL_PATH_TRAIN = "./model_train/"
if not os.path.exists(MODEL_PATH_TRAIN):
    os.mkdir(MODEL_PATH_TRAIN)

RESTORE_NAME = 'sl_21-12-21_09-11-12'
# 第 83 轮（DS，#45）：支持"从指定权重继续训练 + 覆盖学习率"。
# 用途：从多版本平均（soup）得到的权重出发，用更小的 lr 再训 1 轮 ——
# "先平均、再重新特化"，是集成之外唯一没试过的组合。
RESTORE_PATH = os.environ.get("MAS_SL_RESTORE_PATH",
                              MODEL_PATH + RESTORE_NAME + '.pth')
RESTORE_PATH_TRAIN = MODEL_PATH_TRAIN + RESTORE_NAME + '.pkl'

SAVE_STATE_DICT = True
SAVE_ALL_PKL = False
SAVE_CHECKPOINT = True

LOAD_STATE_DICT = os.environ.get("MAS_SL_LOAD_STATE", "0") == "1"
LOAD_ALL_PKL = False
# 第 83 轮（DS，#46b）：用 MAS_SL_LOAD_STATE 指定权重续训时，必须关掉默认的
# "从 model_train/*.pkl 恢复" 分支，否则会去找不存在的 pkl 直接报错。
LOAD_CHECKPOINT = not (os.environ.get("MAS_SL_LOAD_STATE", "0") == "1")

# ---- 第 78 轮（DS，#23）：静默降级的根因修复 ----
# 原实现 `SIMPLE_TEST = not P.on_server`：只要没设 MAS_ON_SERVER=1，
# 就会在**没有任何提示**的情况下只读 1 个训练 + 1 个验证 tensor 文件。
# 实测后果：2026-11-02 那 2.5 小时、5 个 checkpoint 的 SL 训练，
# train_set 只有 120 个序列（1 个 replay），collected 只有官方模型的 28%。
#
# 改法：冒烟必须是**显式**的（MAS_SL_SMOKE=1），默认全量。
# 这样"忘记设环境变量"只影响设备选择，不会悄悄把训练集砍成 1/60。
SIMPLE_TEST = os.environ.get("MAS_SL_SMOKE", "0") == "1"

# 数据划分：训练集在前，验证集取「文件列表末尾」的 VAL_NUM 个，两者不重叠。
# 原实现是 TRAIN_FROM=0 / TRAIN_NUM=90 配 VAL_FROM=0 / VAL_NUM=5，
# 验证集就是训练集的前 5 个回放（数据泄漏）；改为从尾部切分。
# 训练集大小不再写死，由「总数 - VAL_NUM」自适应，数据量增减无需改这里。
TRAIN_FROM = 0                      # 训练集起始下标（若开头有几个坏样本可改这里跳过）
TRAIN_NUM = None                    # None = 训练到验证集起点为止
VAL_NUM = (1 if SIMPLE_TEST else
           int(os.environ.get("MAS_SL_VAL_NUM", "5")))   # 验证集文件数（从末尾取，默认 5 不变）


def split_indices(total):
    """按「训练集在前 / 验证集在后」计算 [from, end) 区间，保证不重叠。

    返回 ((train_from, train_end), (val_from, val_end))。
    只有一个文件时无法切分，退化为同一份（仅供冒烟测试使用）。
    """
    if total <= 1:
        return (0, total), (0, total)

    val_num = min(VAL_NUM, total - 1)
    val_from = total - val_num

    train_end = val_from
    if TRAIN_NUM is not None:
        train_end = min(train_end, TRAIN_FROM + TRAIN_NUM)
    return (TRAIN_FROM, train_end), (val_from, total)

# hyper paramerters
# use the same as in RL
# BATCH_SIZE = AHP.batch_size
# SEQ_LEN = AHP.sequence_length

# important: use larger batch_size and smaller seq_len in SL!
BATCH_SIZE = int(os.environ.get("MAS_SL_BATCH", str(3 * AHP.batch_size)))

# ---- 第 79 轮（DS，#27）：SEQ_LEN 可配，用于对齐"官方 21 年模型训练时的设置" ----
# 时间线（与 #26 的 Smart 替换同源）：
#   · 官方 sl_21-12-29 模型训练于 2021-12-29；
#   · `SEQ_LEN = AHP.sequence_length * 0.5` 与 Smart_unit 替换是**同一个提交**
#     （54aa123, 2022-05-02，git blame 显示是 boundary commit，即当前历史起点）;
#   · 也就是说 2021-12 时很可能是完整的 `AHP.sequence_length` = 8。
# 时序长度直接决定 LSTM 能看到多长的因果链："造兵 -> 集结 -> 进攻" 跨越几十秒，
# 4 步（step_mul=8，约 1.4 秒回传窗口）可能不足以学到 combat 的时机。
# 默认保持 4（不改历史行为），设 MAS_SL_SEQ_LEN=8 做对照。
SEQ_LEN = int(os.environ.get("MAS_SL_SEQ_LEN", str(int(AHP.sequence_length * 0.5))))

print('BATCH_SIZE:', BATCH_SIZE) if debug else None
print('SEQ_LEN:', SEQ_LEN) if debug else None

# 第 78 轮（DS，#25）：epochs 可配（按时间预算控制）。
# 10 是官方配方的值（readme: lr=1e-4, 10 epochs）；改成可配只为在有限时间里做数据量消融。
NUM_EPOCHS = int(os.environ.get("MAS_SL_EPOCHS", "10"))

LEARNING_RATE = float(os.environ.get("MAS_SL_LR", "1e-4"))
L2SP = float(os.environ.get("MAS_SL_L2SP", "0"))
_L2SP_ANCHOR = None
WEIGHT_DECAY = float(os.environ.get("MAS_SL_WEIGHT_DECAY", "1e-5"))

CLIP_VALUE = 0.5  # SLTHP.clip
STEP_SIZE = int(os.environ.get("MAS_SL_STEP_SIZE", "50"))
GAMMA = float(os.environ.get("MAS_SL_GAMMA", "0.2"))

_SEED = int(os.environ.get("MAS_SL_SEED", str(SLTHP.seed)))
torch.manual_seed(_SEED)
np.random.seed(_SEED)
try:
    import random as _random
    _random.seed(_SEED)
except Exception:
    pass
try:
    torch.cuda.manual_seed_all(_SEED)
except Exception:
    pass
print("[SL-seed] %d" % _SEED)

# ---- 确定性开关（DS 2026-11-07）：MAS_SL_DETERMINISTIC=1 时强制可复现 ----
_det_mode = os.environ.get("MAS_SL_DETERMINISTIC", "0")
if _det_mode in ("1", "strict", "soft"):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("[SL-det] mode=%s cudnn(deterministic=True, benchmark=False)" % _det_mode)
    _det_ok = False
    try:
        torch.use_deterministic_algorithms(True)
        _det_ok = True
        print("[SL-det] use_deterministic_algorithms(True)")
    except AttributeError:
        pass
    except Exception as _e:
        print("[SL-det] use_deterministic_algorithms failed: %s" % _e)
    if not _det_ok and _det_mode != "soft":
        try:
            torch.set_deterministic(True)
            _det_ok = True
            print("[SL-det] set_deterministic(True)")
        except AttributeError:
            print("[SL-det] old torch: only cudnn.deterministic is available")


def getReplayData(path, replay_files, from_index=0, end_index=None):
    td_list = []
    # R7 (2026-11-14): 原来 except 里只 traceback.print_exc()，坏文件会被**静默跳过** ——
    # 训练照常开跑，但用的是更少的数据，日志里也看不出来。现在收集失败并在结束时拒绝启动
    # （MAS_SL_ALLOW_BAD_FILES=1 可显式放行）。
    _bad = []
    for i, replay_file in enumerate(tqdm(replay_files)):
        try:
            replay_path = path + replay_file
            print('replay_path:', replay_path) if 1 else None

            do_write = False
            if i >= from_index:
                if end_index is None:
                    do_write = True
                elif end_index is not None and i < end_index:
                    do_write = True

            if not do_write:
                continue 

            features, labels = torch.load(replay_path)
            print('features.shape:', features.shape) if debug else None
            print('labels.shape::', labels.shape) if debug else None

            td_list.append(ReplayTensorDataset(features, labels, seq_len=SEQ_LEN))

        except Exception as e:
            traceback.print_exc()
            _bad.append((replay_file, type(e).__name__, str(e)[:100]))

    if _bad:
        _msg = "[SL-data] %d/%d 个文件读取失败：" % (len(_bad), len(replay_files))
        _detail = "; ".join("%s(%s:%s)" % (b[0], b[1], b[2]) for b in _bad[:5])
        if os.environ.get("MAS_SL_ALLOW_BAD_FILES", "0") == "1":
            print("[SL-data][warn] " + _msg + _detail + " —— 已按 MAS_SL_ALLOW_BAD_FILES=1 继续", flush=True)
        else:
            raise RuntimeError(_msg + _detail + "（确认可跳过请设 MAS_SL_ALLOW_BAD_FILES=1）")

    return td_list


class _WindowSubset(torch.utils.data.Dataset):
    """按 (文件下标, 起始帧) 清单取窗口，用于帧级随机切分（不复制张量）。"""

    def __init__(self, tensor_list, windows, seq_len):
        self.tensors = tensor_list
        self.windows = windows
        self.seq_len = seq_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, k):
        i, s = self.windows[k]
        f, l = self.tensors[i]
        f = f[s:s + self.seq_len]
        l = l[s:s + self.seq_len]
        # R3 (2026-11-14): fp16 存储的数据集（to_fp16.py 生成）在这里上采样到 fp32，
        # 模型侧完全无感；fp32 数据集走原路径，零开销。
        if f.dtype != torch.float32:
            f = f.float()
        if l.dtype != torch.float32:
            l = l.float()
        return f, l



def _balance_window_weights(train_set, seq_len):
    """按动作类型给每个训练窗口算采样权重（权重 ∝ 目标占比 / 实际占比）。"""
    import collections
    from pysc2.lib.actions import RAW_FUNCTIONS as _RF

    def _bucket(name):
        n = name.lower()
        if "attack" in n:
            return "attack"
        if n.startswith("train_zealot"):
            return "train_zealot"
        if n.startswith("train_probe"):
            return "train_probe"
        if "harvest" in n or "gather" in n:
            return "harvest"
        if n.startswith("build_pylon"):
            return "build_pylon"
        if n.startswith("build_gateway"):
            return "build_gateway"
        if "cancel" in n:
            return "cancel"
        return "other"

    _target = {
        "attack": 0.30, "train_zealot": 0.20, "train_probe": 0.20,
        "harvest": 0.12, "build_pylon": 0.08, "build_gateway": 0.05,
        "cancel": 0.02, "other": 0.03,
    }
    _spec = os.environ.get("MAS_SL_BALANCE_MIX", "").strip()
    if _spec:
        for _it in _spec.split(","):
            if "=" in _it:
                _k, _v = _it.split("=")
                _target[_k.strip()] = float(_v)
    _sum = sum(_target.values())
    _target = {k: v / _sum for k, v in _target.items()}

    _cnt = collections.Counter()
    _ids = []
    for (i, st) in train_set.windows:
        _lab = train_set.tensors[i][1]
        _j = min(st + seq_len - 1, _lab.shape[0] - 1)
        _aid = int(_lab[_j, :564].argmax())
        _nm = _RF[_aid].name if _aid < len(_RF) else "?"
        _b = _bucket(_nm)
        _ids.append(_b)
        _cnt[_b] += 1

    _n = max(len(_ids), 1)
    _maxw = float(os.environ.get("MAS_SL_BALANCE_MAXW", "30"))
    _w = []
    for _b in _ids:
        _emp = _cnt[_b] / _n
        _w.append(min(_maxw, _target.get(_b, 0.03) / max(_emp, 1e-6)))
    import torch as _t
    _w = _t.tensor(_w, dtype=_t.double)
    print("[SL-balance] windows=%d 实际=%s" % (_n, dict(_cnt)), flush=True)
    print("[SL-balance] 目标=%s  权重 min=%.2f med=%.2f max=%.2f"
          % ({k: round(v, 3) for k, v in _target.items()},
             float(_w.min()), float(_w.median()), float(_w.max())), flush=True)
    return _w


def main_worker(device):
    global _L2SP_ANCHOR
    print('==> Making model..')
    net = ArchModel()
    checkpoint = None

    # 第 85 轮（DS，#84）：证明"续训到底有没有加载到权重"。
    # 依据：#84 实测 FT_ft_w1.pth 与冠军权重 rel_l2=0.99（几乎不相关），
    # 但日志里没有任何加载痕迹 —— `strict=False` 让"加载成功"和"静默什么都没加载"
    # 在日志上完全不可区分（字典解包错、key 前缀不一致都会静默跳过）。
    # 判据：打印 param0 加载前后是否变化；`_sig_before == _sig_after` 即没加载。
    _sig_before = None
    try:
        _p0 = next(iter(net.parameters()))
        _sig_before = float(_p0.detach().float().abs().sum())
        _sig_after = None
    except Exception:
        pass

    if RESTORE:
        if LOAD_STATE_DICT:
            # use state dict to restore
            _sd = torch.load(RESTORE_PATH, map_location=device)
            if isinstance(_sd, dict) and 'model' in _sd:
                print('[SL-restore] ⚠️ %s 是 checkpoint 字典而不是 state_dict，取 ["model"]' % RESTORE_PATH)
                _sd = _sd['model']
            _missing, _unexpected = net.load_state_dict(_sd, strict=False)
            try:
                _sig_after = float(next(iter(net.parameters())).detach().float().abs().sum())
            except Exception:
                _sig_after = float('nan')
            print('[SL-restore] path=%s keys=%d missing=%d unexpected=%d changed=%s '
                  'param0_abs_sum %.6f -> %.6f'
                  % (RESTORE_PATH, len(_sd), len(_missing), len(_unexpected),
                     _sig_before != _sig_after, _sig_before, _sig_after))

        if LOAD_ALL_PKL:
            # use all to restore
            net = torch.load(RESTORE_PATH_TRAIN, map_location=device)

        if LOAD_CHECKPOINT:
            # use checkpoint to restore
            checkpoint = torch.load(RESTORE_PATH_TRAIN, map_location=device)
            net.load_state_dict(checkpoint['model'], strict=False)
    net = net.to(device)

    # 第 97 轮（DS）：只训“决策头”，冻结 backbone/encoder/LSTM 与 baseline。
    # 动机：#99/#107 已证否“全模型微调冠军”（Adam 累计位移打崩经济，36 局 0 胜）。
    # 这次把可训练参数限制在 action/units/target_unit/location 四个头，
    # 目标是修“攻击时选农民”，而不是重学经济。
    if os.environ.get("MAS_SL_FREEZE_BACKBONE", "0") == "1":
        _keep_env = os.environ.get("MAS_SL_TRAIN_HEADS", "").strip()
        if _keep_env:
            _keep = tuple(_x.strip() for _x in _keep_env.split(",") if _x.strip())
        else:
            _keep = ("action_type_head", "selected_units_head", "target_unit_head", "location_head")
        for _n, _p in net.named_parameters():
            _p.requires_grad = any(_k in _n for _k in _keep)
        _trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        _total = sum(p.numel() for p in net.parameters())
        print("[SL-freeze] backbone frozen, trainable=%d / total=%d (heads=%s)"
              % (_trainable, _total, ",".join(_keep)), flush=True)
        _L2SP_ANCHOR = {_n: _p.detach().clone() for _n, _p in net.named_parameters()
                        if _p.requires_grad}
        print("[SL-l2sp] anchor=%d params lambda=%g" % (len(_L2SP_ANCHOR), L2SP), flush=True)

    # 不冻结 backbone 时也要能锚定（全模型微调用）
    if L2SP > 0 and _L2SP_ANCHOR is None:
        _L2SP_ANCHOR = {_n: _p.detach().clone() for _n, _p in net.named_parameters()
                        if _p.requires_grad}
        print("[SL-l2sp-all] anchor=%d params lambda=%g"
              % (len(_L2SP_ANCHOR), L2SP), flush=True)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Making optimizer and scheduler..')

    optimizer, scheduler = None, None
    batch_iter, epoch = 0, 0

    if RESTORE and LOAD_CHECKPOINT:
        # use checkpoint to restore other
        optimizer = Adam(net.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler = StepLR(optimizer, step_size=STEP_SIZE)
        scheduler.load_state_dict(checkpoint['scheduler'])

        batch_iter = checkpoint['batch_iter']
        print('batch_iter is', batch_iter)
        epoch = checkpoint['epoch']
        print('epoch is', epoch)

        ckpt = torch.load(RESTORE_PATH_TRAIN)
        np.random.set_state(ckpt['numpy_random_state'])
        torch.random.set_rng_state(ckpt['torch_random_state'])
    else:
        optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        _sched = os.environ.get("MAS_SL_SCHED", "step")
        if _sched == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS,
                                          eta_min=float(os.environ.get("MAS_SL_ETA_MIN", "1e-6")))
            print("[SL-sched] cosine T_max=%d eta_min=%s" % (NUM_EPOCHS, os.environ.get("MAS_SL_ETA_MIN", "1e-6")))
        else:
            scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print('==> Preparing data..')

    # 第 81 轮（DS，#34）：**必须排序**。
    # 原实现用裸 os.listdir，返回顺序由 inode 决定 —— 于是
    #   (a) 每次训练/验证的划分（split_indices 取前 N / 末尾 5）其实是任意的，
    #   (b) 无法通过"把重要文件排到前面"来做数据筛选（#35）。
    # 排序后：文件顺序确定，划分可复现，且 `data/replay_data_v3_cbt/`（按攻击密度命名的软链接）
    # 的前缀排序能真正生效。
    replay_files = sorted([f for f in os.listdir(PATH) if f.endswith('.pt')])
    print('length of replay_files:', len(replay_files)) if debug else None
    replay_files.sort()

    if SIMPLE_TEST:
        # 冒烟模式只取前 2 个文件（1 训练 + 1 验证），不要按全量目录切分
        replay_files = replay_files[:2]

    # 第 78 轮（DS，#23）：可控文件数（按内存预算分批训练 / 做数据量消融）
    #   例：MAS_SL_MAX_FILES=20 -> 15 训练 + 5 验证（VAL_NUM=5 从末尾取）
    _max_files = os.environ.get("MAS_SL_MAX_FILES", "").strip()
    if _max_files and not SIMPLE_TEST:
        _n = max(2, int(_max_files))
        if _n < len(replay_files):
            replay_files = replay_files[:_n]
        print('MAS_SL_MAX_FILES=%d -> 实际使用 %d 个文件'
              % (_n, len(replay_files)))

    # 第 78 轮（DS，#23）：把"用了多少数据"提到启动日志的第一屏。
    # 上一次训练就是因为这条信息藏在第 12 行之后没人看，才让 1/60 的数据量跑了 2.5 小时。
    print('[SL-config] files=%d smoke=%s lr=%g epochs=%d batch=%d seq=%d workers=%d'
          % (len(replay_files), SIMPLE_TEST, LEARNING_RATE, NUM_EPOCHS,
             BATCH_SIZE, SEQ_LEN, NUM_WORKERS), flush=True)

    # ---- HY26: DAgger 严格区分（默认关闭，无 manifest 时完全不影响历史行为） ----
    import json as _json
    _mp = os.path.join(PATH, "manifest.json")
    _dg = 0
    if os.path.exists(_mp):
        with open(_mp, "r", encoding="utf-8") as _f:
            _man = _json.load(_f)
        _dg = 1 if bool(_man.get("dagger", False)) else 0
        print("[SL-manifest] %s" % _json.dumps(_man, ensure_ascii=False, sort_keys=True), flush=True)
        _want = int(os.environ.get("MAS_DAGGER", "0"))
        if _want != _dg:
            raise RuntimeError("[SL-manifest] 拒绝启动：数据集 dagger=%d 但 MAS_DAGGER=%d"
                               % (_dg, _want))
        print("[SL-manifest] 双向校验 PASS  dagger=%d MAS_DAGGER=%d" % (_dg, _want), flush=True)
    else:
        _want = int(os.environ.get("MAS_DAGGER", "0"))
        if _want == 1:
            raise RuntimeError("[SL-manifest] 声明 MAS_DAGGER=1 但 %s 不存在 manifest.json" % PATH)
        print("[SL-manifest] 无 manifest（历史数据集），dagger=0", flush=True)
    # ---- end HY26 ----

    (train_from, train_end), (val_from, val_end) = split_indices(len(replay_files))

    # --- P3b (DS #208, 2026-11-06): 复现官方的划分，用于分离「代码差异」与「165 帧数据差」---
    # 官方 VAL_FROM=0 -> val 取前 5 个文件，完全包含在 train 里（泄漏）；
    # 主仓修掉了泄漏，但训练集少 165 帧（#207）。置 1 时与官方训练集逐文件一致。
    if os.environ.get("MAS_SL_TRAIN_ALL", "0") == "1":
        (train_from, train_end), (val_from, val_end) = (0, len(replay_files)), (0, 5)
        print('[P3b] MAS_SL_TRAIN_ALL=1 -> train=[0:%d] val=[0:5] (复现官方划分)' % len(replay_files))
    print('data split: train_files[%d:%d] (%d)  val_files[%d:%d] (%d)'
          % (train_from, train_end, max(train_end - train_from, 0),
             val_from, val_end, max(val_end - val_from, 0)))

    train_list = getReplayData(PATH, replay_files, from_index=train_from, end_index=train_end)
    val_list = getReplayData(PATH, replay_files, from_index=val_from, end_index=val_end)

    print('len(train_list)', len(train_list)) if debug else None
    print('len(val_list)', len(val_list)) if debug else None

    if os.environ.get("MAS_SL_FRAME_SPLIT", "0") == "1":
        # --- 帧级随机切分（默认关闭）---
        _frac = float(os.environ.get("MAS_SL_VAL_FRAC", "0.2"))
        _sseed = int(os.environ.get("MAS_SL_SPLIT_SEED", "20261108"))
        _tensors = [(d.tensors[0], d.tensors[1]) for d in train_list + val_list]
        _wins = []
        for _i, (_f, _l) in enumerate(_tensors):
            _n = max(_f.size(0) - SEQ_LEN + 1, 0)
            _wins.extend((_i, _s) for _s in range(_n))
        _perm = np.random.RandomState(_sseed).permutation(len(_wins))
        _n_val = int(len(_wins) * _frac)
        _val_w = [_wins[k] for k in _perm[:_n_val]]
        _tr_w = [_wins[k] for k in _perm[_n_val:]]
        train_set = _WindowSubset(_tensors, _tr_w, SEQ_LEN)
        val_set = _WindowSubset(_tensors, _val_w, SEQ_LEN)
        print('[frame-split] files=%d windows=%d train=%d val=%d frac=%s seed=%s'
              % (len(_tensors), len(_wins), len(_tr_w), len(_val_w), _frac, _sseed), flush=True)
    else:
        train_set = ConcatDataset(train_list)
        val_set = ConcatDataset(val_list)

    print('len(train_set)', len(train_set)) if debug else None
    print('len(val_set)', len(val_set)) if debug else None

    # num_workers 原来默认 2，而这台机器有 64 个物理核；pin_memory=False 让每个 batch
    # 的 H2D 拷贝都走同步，单帧 1.4 MB、batch 8x8 就是 90 MB/步，差距可观。
    _dl_gen = torch.Generator()
    _dl_gen.manual_seed(_SEED)

    def _worker_init_fn(_wid):
        _s = _SEED + _wid
        try:
            import random as _r
            _r.seed(_s)
        except Exception:
            pass
        np.random.seed(_s)
        torch.manual_seed(_s)

    if os.environ.get("MAS_SL_BALANCE", "0") == "1" and hasattr(train_set, "windows"):
        from torch.utils.data import WeightedRandomSampler
        _bw = _balance_window_weights(train_set, SEQ_LEN)
        _sampler = WeightedRandomSampler(_bw, num_samples=len(_bw), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=_sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  worker_init_fn=_worker_init_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  generator=_dl_gen, worker_init_fn=_worker_init_fn)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)   

    print('len(train_loader)', len(train_loader)) if debug else None
    print('len(val_loader)', len(val_loader)) if debug else None

    train(net, optimizer, scheduler, train_set, train_loader, device, 
          val_set, batch_iter, epoch, val_loader)


def train(net, optimizer, scheduler, train_set, train_loader, device, 
          val_set, batch_iter, epoch, val_loader=None):

    now = datetime.datetime.now()
    summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "_" + str(os.getpid()) + "/"
    writer = SummaryWriter(summary_path)

    time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time_str)
    SAVE_PATH_TRAIN = os.path.join(MODEL_PATH_TRAIN, MODEL + "_" + time_str)

    epoch_start = time.time()

    # 第 78 轮（DS，#24）：跟踪最佳 val action acc，训练结束给出"该用哪个 epoch"的结论。
    # 判据用 val action acc（与训练 acc 不同源，不会自我印证）。
    best_val_acc, best_ep = -1.0, -1
    # 2026-11-19：同时按 val loss 记录一份最佳（acc 在塌缩后会虚高，见 LN 系列实验）
    best_val_loss = float('inf')

    # --- EMA（Hunyuan / HY24 T4 + HY25 O1 warmup）：默认关闭，行为不变 ---
    _EMA_DECAY = float(os.environ.get("MAS_SL_EMA_DECAY", "0.0"))
    _EMA_WARMUP = int(os.environ.get("MAS_SL_EMA_WARMUP", "0"))
    _ema_step = 0
    _ema_state = None
    if 0.0 < _EMA_DECAY < 1.0:
        _ema_state = {k: v.detach().clone().float()
                      for k, v in net.state_dict().items() if v.is_floating_point()}
        print("[SL-ema] enabled decay=%s warmup=%s tensors=%d"
              % (_EMA_DECAY, _EMA_WARMUP, len(_ema_state)), flush=True)
    # --- end EMA ---

    for ep in range(NUM_EPOCHS):
        # 第 93 轮（DS，#102）：MAS_SL_VAL_ONLY=1 —— 只跑一次验证就返回，不训练、不存 checkpoint。
        # 用途：同一个 ckpt 在 strict / 非 strict 两种分母口径下各跑一遍 val，量化"分母漂移"。
        # 默认不设 → 行为与历史完全一致。
        if os.environ.get("MAS_SL_VAL_ONLY") == "1":
            print('[SL-valonly] 跳过训练，只跑验证（strict=%s）'
                  % os.environ.get("MAS_SL_SEL_STRICT", "1"), flush=True)
            _vl, _va = eval(net, val_set, val_loader, device)
            print('[SL-valonly] done loss=%.6f action=%.4f sel_acc=%.4f sel_type=%.4f sel_num=%.4f'
                  % (_vl, _va[0], _va[6], _va[7], _va[8]), flush=True)
            return

        loss_sum = 0
        epoch += 1 

        # put model in train mode
        net.train()
        # --- HY25 / O2: 冻结 BN running stats（默认关闭，行为不变） ---
        if os.environ.get("MAS_SL_BN_FREEZE") == "1":
            _bn_n = 0
            for _m in net.modules():
                if isinstance(_m, torch.nn.BatchNorm2d):
                    _m.eval()
                    _bn_n += 1
            print("[SL-bnfreeze] frozen %d BatchNorm2d" % _bn_n, flush=True)
        # --- end BN freeze ---
        # ---- HY26: LR 线性 warmup（默认 0，行为不变） ----
        _WARMUP = int(os.environ.get("MAS_SL_WARMUP", "0"))
        for batch_idx, (features, labels) in enumerate(train_loader):
            if _WARMUP > 0:
                _gs = ep * len(train_loader) + batch_idx
                if _gs < _WARMUP:
                    _lr = LEARNING_RATE * float(_gs + 1) / float(_WARMUP)
                    for _pg in optimizer.param_groups:
                        _pg["lr"] = _lr
                    if batch_idx == 0 or _gs == _WARMUP - 1:
                        print("[SL-warmup] step=%d/%d lr=%.3e"
                              % (_gs + 1, _WARMUP, _lr), flush=True)
            start = time.time()

            feature_tensor = features.to(device).float()
            labels_tensor = labels.to(device).float()
            del features, labels

            loss, loss_list, \
                acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                           labels_tensor, net, 
                                                           decrease_smart_opertaion=True,
                                                           return_important=True,
                                                           only_consider_small=False,
                                                           train=True)
            del feature_tensor, labels_tensor

            # 第 97 轮（DS）：L2-SP 锚定。只解冻部分 backbone 时，防止它漂离冠军太远
            # （#107 全模型微调就是被 Adam 累计位移打崩经济的）。
            if L2SP > 0 and _L2SP_ANCHOR:
                _reg = None
                for _n, _p in net.named_parameters():
                    if _p.requires_grad and _n in _L2SP_ANCHOR:
                        _d = (_p - _L2SP_ANCHOR[_n]).pow(2).sum()
                        _reg = _d if _reg is None else _reg + _d
                if _reg is not None:
                    loss = loss + L2SP * _reg

            optimizer.zero_grad()
            loss.backward() 

            # 梯度裁剪必须在 optimizer.step() 之前，否则只改 .grad 而本步更新已经做完，
            # 等于裁剪完全没生效（原代码把这段放在 step() 之后）。
            if CLIP:
                parameters = [p for p in net.parameters() if p is not None and p.requires_grad]
                torch.nn.utils.clip_grad_norm_(parameters, CLIP_VALUE)

            optimizer.step()

            if _ema_state is not None:
                _ema_step += 1
                _eff = _EMA_DECAY
                if _EMA_WARMUP > 0:
                    # 标准 EMA warmup：初期 decay 小、跟随当前权重更快，随后升到目标值
                    _eff = min(_EMA_DECAY, (1.0 + _ema_step) / (_EMA_WARMUP + _ema_step))
                with torch.no_grad():
                    _ns = net.state_dict()
                    for _k, _v in _ema_state.items():
                        _v.mul_(_eff).add_(_ns[_k].detach().float(),
                                           alpha=1.0 - _eff)

            loss_value = float(loss.item())

            loss_sum += loss_value
            del loss

            action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)
            move_camera_accuracy = acc_num_list[2] / (acc_num_list[3] + 1e-9)
            non_camera_accuracy = acc_num_list[4] / (acc_num_list[5] + 1e-9)
            short_important_accuracy = acc_num_list[6] / (acc_num_list[7] + 1e-9)

            location_accuracy = acc_num_list[8] / (acc_num_list[9] + 1e-9)
            location_distance = acc_num_list[11] / (acc_num_list[9] + 1e-9)

            selected_units_accuracy = acc_num_list[12] / (acc_num_list[13] + 1e-9)
            selected_units_type_right = acc_num_list[14] / (acc_num_list[15] + 1e-9)
            selected_units_num_right = acc_num_list[16] / (acc_num_list[17] + 1e-9)

            target_unit_accuracy = acc_num_list[18] / (acc_num_list[19] + 1e-9)

            batch_time = time.time() - start

            batch_iter += 1
            print('batch_iter', batch_iter)

            write(writer, loss_value, loss_list, action_accuracy, 
                  move_camera_accuracy, non_camera_accuracy, short_important_accuracy,
                  location_accuracy, location_distance, selected_units_accuracy,
                  selected_units_type_right, selected_units_num_right, target_unit_accuracy,
                  batch_iter)

            gc.collect()

            print('Batch/Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_iter, epoch, loss_value, action_accuracy, batch_time))

        # 第 78 轮（DS，#24）：每个 epoch 存**独立文件**。
        # 原实现 SAVE_PATH 用启动时间戳命名，10 个 epoch 全写同一个 .pth/.pkl，
        # 只保留最后一轮 —— 若末轮过拟合，最好的中间模型就丢了（本轮 ep9=0.940、ep10=0.937）。
        _ep_tag = "_ep%02d" % (ep + 1)
        if SAVE_STATE_DICT:
            save_path = SAVE_PATH + _ep_tag + ".pth"
            print('Save model state_dict to', save_path)
            if _ema_state is not None:
                _ns = net.state_dict()
                _cur = {k: _ns[k].detach().clone() for k in _ema_state}
                with torch.no_grad():
                    for _k, _v in _ema_state.items():
                        _ns[_k].copy_(_v)
                torch.save(net.state_dict(), save_path)
                with torch.no_grad():
                    for _k, _v in _cur.items():
                        _ns[_k].copy_(_v)
                print("[SL-ema] saved EMA weights to %s" % save_path, flush=True)
            else:
                torch.save(net.state_dict(), save_path)

        if SAVE_ALL_PKL:
            save_path = SAVE_PATH_TRAIN + _ep_tag + ".pkl"
            print('Save model all to', save_path)
            torch.save(net, save_path)

        if SAVE_CHECKPOINT:
            save_path = SAVE_PATH_TRAIN + _ep_tag + ".pkl"
            save_dict = {'batch_iter': batch_iter,
                         'epoch': epoch,
                         'model': net.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'loss': loss_value,
                         'numpy_random_state': np.random.get_state(),
                         'torch_random_state': torch.random.get_rng_state(),
                         }

            print('Save model checkpoint to', save_path)
            torch.save(save_dict, save_path)

        if True:
            print('eval begin')
            val_loss, val_acc = eval(net, val_set, val_loader, device)
            print('eval end')

            print("Val loss: {:.6f}.".format(val_loss))
            writer.add_scalar('Val/Loss', val_loss, batch_iter)

            # for accuracy of actions in val
            print("Val action acc: {:.6f}.".format(val_acc[0]))
            writer.add_scalar('Val/action Acc', val_acc[0], batch_iter)
            print("Val move_camera acc: {:.6f}.".format(val_acc[1]))
            writer.add_scalar('Val/move_camera Acc', val_acc[1], batch_iter)
            print("Val non_camera acc: {:.6f}.".format(val_acc[2]))
            writer.add_scalar('Val/non_camera Acc', val_acc[2], batch_iter)
            print("Val short_important acc: {:.6f}.".format(val_acc[3]))
            writer.add_scalar('Val/short_important Acc', val_acc[3], batch_iter)
            print("Val location_accuracy acc: {:.6f}.".format(val_acc[4]))
            writer.add_scalar('Val/location_accuracy Acc', val_acc[4], batch_iter)
            print("Val location_distance acc: {:.6f}.".format(val_acc[5]))
            writer.add_scalar('Val/location_distance Acc', val_acc[5], batch_iter)
            print("Val selected_units_accuracy acc: {:.6f}.".format(val_acc[6]))
            writer.add_scalar('Val/selected_units_accuracy Acc', val_acc[6], batch_iter)
            print("Val selected_units_type_right acc: {:.6f}.".format(val_acc[7]))
            writer.add_scalar('Val/selected_units_type_right Acc', val_acc[7], batch_iter)
            print("Val selected_units_num_right acc: {:.6f}.".format(val_acc[8]))
            writer.add_scalar('Val/selected_units_num_right Acc', val_acc[8], batch_iter)
            print("Val target_unit_accuracy acc: {:.6f}.".format(val_acc[9]))
            writer.add_scalar('Val/target_unit_accuracy Acc', val_acc[9], batch_iter)            

            # 第 78 轮（DS，#24）：记录最佳 epoch 并单独存一份 _best.pth
            if val_acc[0] > best_val_acc:
                best_val_acc, best_ep = float(val_acc[0]), ep + 1
                _best_path = SAVE_PATH + "_best.pth"
                if SAVE_STATE_DICT:
                    torch.save(net.state_dict(), _best_path)
                print('[SL-best] new best val_action_acc=%.6f at epoch %d -> %s'
                      % (best_val_acc, best_ep, _best_path), flush=True)

            # 2026-11-19：按 val loss 另存一份 _bestloss.pth。
            # 动机：action acc 在策略塌缩到多数类时会"跳升"，按 acc 选模型等于选退化权重；
            # val loss 对塌缩更敏感（今日 b0s1：ep2 acc 0.284/loss 更低，ep4 acc 0.646）。
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                _bl_path = SAVE_PATH + "_bestloss.pth"
                if SAVE_STATE_DICT:
                    torch.save(net.state_dict(), _bl_path)
                print('[SL-bestloss] val_loss=%.6f at epoch %d -> %s'
                      % (best_val_loss, ep + 1, _bl_path), flush=True)

            del val_loss, val_acc

        gc.collect()

        scheduler.step()

    print('[SL-best] 最终：best val_action_acc=%.6f at epoch %d；'
          '逐 epoch 权重在 %s_epNN.pth，最佳副本在 %s_best.pth'
          % (best_val_acc, best_ep, SAVE_PATH, SAVE_PATH), flush=True)

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


def eval(model, val_set, val_loader, device):
    model.eval()

    # should we use it ？
    # model.core.lstm.train()

    loss_sum = 0.0
    i = 0

    action_acc_num = 0.
    action_all_num = 0.

    move_camera_action_acc_num = 0.
    move_camera_action_all_num = 0.

    non_camera_action_acc_num = 0.
    non_camera_action_all_num = 0.

    short_important_action_acc_num = 0.
    short_important_action_all_num = 0.

    location_acc_num = 0.
    location_effect_num = 0.
    location_dist = 0.

    selected_units_correct_num, selected_units_gt_num = 0, 0
    selected_units_type_correct_num, selected_units_pred_num = 0, 0
    selected_units_nums_equal, selected_units_batch_size = 0, 0

    target_unit_correct_num, target_unit_all_num = 0, 0

    # #112：分层准确率（no_op 主导，说明见 sl_loss_multi_gpu.py 尾部注释）
    noop_ok_num, noop_all_num = 0, 0
    nonnoop_ok_num, nonnoop_all_num = 0, 0
    # #112c：target 头在"要点单位"帧上的准确率
    tgt_ok_num, tgt_all_num = 0, 0
    # #122b：8 个关键动作的 (判对数, 真值数)，顺序见 sl_loss_multi_gpu.py 里的 _KEY_ACTS
    # （ID 由 #122c 的 `_audit_action_hist.py` 量出来，不是猜的）
    KEY_ACTS = (12, 2, 49, 35, 64, 37, 36, 47)
    key_ok = [0] * len(KEY_ACTS)
    key_all = [0] * len(KEY_ACTS)

    for i, (features, labels) in enumerate(val_loader):

        feature_tensor = features.to(device).float()
        labels_tensor = labels.to(device).float()
        del features, labels

        with torch.no_grad():
            loss, _, acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                                labels_tensor, model, 
                                                                decrease_smart_opertaion=True,
                                                                return_important=True,
                                                                only_consider_small=False,
                                                                train=False)
            del feature_tensor, labels_tensor

        print('eval i', i, 'loss', loss) if 1 else None

        loss_sum += loss.item()

        action_acc_num += acc_num_list[0]
        action_all_num += acc_num_list[1]

        move_camera_action_acc_num += acc_num_list[2]
        move_camera_action_all_num += acc_num_list[3]

        non_camera_action_acc_num += acc_num_list[4]
        non_camera_action_all_num += acc_num_list[5]

        short_important_action_acc_num += acc_num_list[6]
        short_important_action_all_num += acc_num_list[7]

        location_acc_num += acc_num_list[8]
        location_effect_num += acc_num_list[9]
        location_dist += acc_num_list[11]

        selected_units_correct_num += acc_num_list[12]
        selected_units_gt_num += acc_num_list[13]

        selected_units_type_correct_num += acc_num_list[14]
        selected_units_pred_num += acc_num_list[15]

        selected_units_nums_equal += acc_num_list[16]
        selected_units_batch_size += acc_num_list[17]

        target_unit_correct_num += acc_num_list[18]
        target_unit_all_num += acc_num_list[19]

        # #112：增量式指标，旧 ckpt 产物不会出现，只在本进程内累加
        if len(acc_num_list) >= 24:
            noop_ok_num += acc_num_list[20]
            noop_all_num += acc_num_list[21]
            nonnoop_ok_num += acc_num_list[22]
            nonnoop_all_num += acc_num_list[23]
        # #112c：target 头的真实口径（只在"要点单位"的帧上算）
        if len(acc_num_list) >= 26:
            tgt_ok_num += acc_num_list[24]
            tgt_all_num += acc_num_list[25]
        # #122b：逐动作召回（下标 26..41，共 8 个动作 × 2）
        if len(acc_num_list) >= 42:
            for _j in range(len(KEY_ACTS)):
                key_ok[_j] += acc_num_list[26 + 2 * _j]
                key_all[_j] += acc_num_list[27 + 2 * _j]

        del loss, acc_num_list

        gc.collect()

    # 原来的写法是 loss_sum / (i + 1e-9)：i 是 enumerate 的最后一个下标（从 0 开始），
    # 只有 1 个 batch 时 i=0，相当于除以 1e-9，val loss 会被放大到 1e9 量级。
    # 这里改成除以 batch 数量（i 从 0 起，所以个数是 i + 1）。
    val_loss = loss_sum / max(i + 1, 1)

    action_accuracy = action_acc_num / (action_all_num + 1e-9)
    move_camera_accuracy = move_camera_action_acc_num / (move_camera_action_all_num + 1e-9)
    non_camera_accuracy = non_camera_action_acc_num / (non_camera_action_all_num + 1e-9)
    short_important_accuracy = short_important_action_acc_num / (short_important_action_all_num + 1e-9)

    location_accuracy = location_acc_num / (location_effect_num + 1e-9)
    location_distance = location_dist / (location_effect_num + 1e-9)

    selected_units_accuracy = selected_units_correct_num / (selected_units_gt_num + 1e-9)
    selected_units_type_right = selected_units_type_correct_num / (selected_units_pred_num + 1e-9)
    selected_units_num_right = selected_units_nums_equal / (selected_units_batch_size + 1e-9)

    target_unit_accuracy = target_unit_correct_num / (target_unit_all_num + 1e-9)

    # #112：分层准确率。`action` 的总体准确率在 v3 上被 92.49% 的 no_op 主导，
    # 平凡基线（永远输出 no_op）就有 92.49%，所以 `action` 只适合当"没有崩"的体检指标，
    # 不能当"模型会不会动"的判据。下面三个数才是能区分模型好坏的：
    #   noop_r     = no_op 帧的召回（越高说明没有乱动的倾向）
    #   nonnoop_r  = 非 no_op 帧的召回（**这才是"会动"的指标**）
    #   bal        = 两者均值（类别平衡准确率，平凡基线 = 0.5，随机 ≈ 0.5）
    noop_recall = noop_ok_num / (noop_all_num + 1e-9)
    nonnoop_recall = nonnoop_ok_num / (nonnoop_all_num + 1e-9)
    balanced_accuracy = 0.5 * (noop_recall + nonnoop_recall)
    tgt_on_targeted = tgt_ok_num / (tgt_all_num + 1e-9)

    # 第 85 轮（DS，#86）：一行汇总，便于模型间直接比（MAS_SL_VAL_VERBOSE=1）。
    # 以前这些指标只进 tensorboard，终端只能看到 Val loss —— 于是"选中单位准确率恒为 0"
    # 这件事一直没人发现（见 sl_loss_multi_gpu.py 的 #86 说明）。
    if os.environ.get("MAS_SL_VAL_VERBOSE", "0") == "1":
        print("[VAL-METRICS] action=%.4f  bal=%.4f  noop_r=%.4f  nonnoop_r=%.4f  "
              "sel_acc=%.4f  sel_type=%.4f  sel_num=%.4f  "
              "loc=%.4f  target=%.4f  tgt_on_tgt=%.4f  n_sel_gt=%d  n_sel_pred=%d  "
              "n_noop=%d  n_nonnoop=%d  n_tgt=%d"
              % (action_accuracy, balanced_accuracy, noop_recall, nonnoop_recall,
                 selected_units_accuracy, selected_units_type_right,
                 selected_units_num_right, location_accuracy, target_unit_accuracy,
                 tgt_on_targeted, selected_units_gt_num, selected_units_pred_num,
                 noop_all_num, nonnoop_all_num, tgt_all_num))
        # #122b：逐动作召回。格式 `act<ID>=<ok>/<n>`，n=0 时打 `-`（表示 val 里没有这类帧，
        # 不要读成"召回 0"）。这行的用途是区分"不会动"和"会挖矿但不会造兵"。
        _kparts = []
        for _j, _a in enumerate(KEY_ACTS):
            if key_all[_j] > 0:
                _kparts.append("act%d=%d/%d(%.2f)" % (_a, key_ok[_j], key_all[_j],
                                                      key_ok[_j] / float(key_all[_j])))
            else:
                _kparts.append("act%d=-" % _a)
        print("[VAL-ACTIONS] " + "  ".join(_kparts))

    return val_loss, [action_accuracy, move_camera_accuracy, non_camera_accuracy, 
                      short_important_accuracy, location_accuracy, location_distance,
                      selected_units_accuracy, selected_units_type_right, selected_units_num_right, target_unit_accuracy]


def write(writer, loss, loss_list, action_accuracy, move_camera_accuracy, 
          non_camera_accuracy, short_important_accuracy, 
          location_accuracy, location_distance, 
          selected_units_accuracy, selected_units_type_right, selected_units_num_right, target_unit_accuracy,
          batch_iter):

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

        print("One batch short_important_accuracy: {:.6f}.".format(short_important_accuracy))
        writer.add_scalar('OneBatch/short_important_accuracy', short_important_accuracy, batch_iter)

        print("One batch location_accuracy: {:.6f}.".format(location_accuracy))
        writer.add_scalar('OneBatch/location_accuracy', location_accuracy, batch_iter)

        print("One batch location_distance: {:.6f}.".format(location_distance))
        writer.add_scalar('OneBatch/location_distance', location_distance, batch_iter)

        print("One batch selected_units_accuracy: {:.6f}.".format(selected_units_accuracy))
        writer.add_scalar('OneBatch/selected_units_accuracy', selected_units_accuracy, batch_iter)

        print("One batch selected_units_type_right: {:.6f}.".format(selected_units_type_right))
        writer.add_scalar('OneBatch/selected_units_type_right', selected_units_type_right, batch_iter)

        print("One batch selected_units_num_right: {:.6f}.".format(selected_units_num_right))
        writer.add_scalar('OneBatch/selected_units_num_right', selected_units_num_right, batch_iter)

        print("One batch target_unit_accuracy: {:.6f}.".format(target_unit_accuracy))
        writer.add_scalar('OneBatch/target_unit_accuracy', target_unit_accuracy, batch_iter)


def test(on_server):
    # gpu setting
    ON_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")

    if ON_GPU:
        if torch.backends.cudnn.is_available():
            print('cudnn available')
            print('cudnn version', torch.backends.cudnn.version())
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = (os.environ.get("MAS_SL_DETERMINISTIC", "0") != "1")

    main_worker(DEVICE)
