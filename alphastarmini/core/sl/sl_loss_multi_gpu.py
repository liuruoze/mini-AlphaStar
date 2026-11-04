#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Library for SL losses for mutli-gpu."

import os
import traceback

import numpy as np

import torch
import torch.nn as nn

from pysc2.lib.actions import RAW_FUNCTIONS as F

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

from alphastarmini.lib import utils as L

__author__ = "Ruo-Ze Liu"

debug = False


# criterion = nn.CrossEntropyLoss()
# due to CrossEntropyLoss only accepts loss with lables.shape = [N]
# we define a loss accept soft_target, which label.shape = [N, C]
# for some rows, it should not be added to loss, so we also need a mask one
def cross_entropy(soft_targets, pred, mask=None, 
                  debug=False, outlier_remove=True, 
                  entity_nums=None, select_size=None,
                  select_units_num=None, use_select_size=True,
                  avg_type='None', sample_weight=None):
    # 第 79 轮（DS，#30）：支持**逐样本权重**。
    # 动机：实测数据里 no_op 占 81%，模型在实战中 99.9% 输出 no_op（DS12 79.12/79.14）。
    # 对 no_op 样本降权（MAS_SL_NOOP_WEIGHT<1）能直接削弱它的主导地位，
    # 让模型更"敢动"。默认 None = 不改任何行为。
    # class is always in the last dim
    logsoftmax = nn.LogSoftmax(dim=-1)
    x_1 = - soft_targets * logsoftmax(pred)

    if debug:
        for i, (t, p) in enumerate(zip(soft_targets, logsoftmax(pred))):
            # print('t', t)
            # print('t.shape', t.shape)
            # print('p', p)
            # print('p.shape', p.shape)
            value = - t * logsoftmax(p)
            # print('value', value)
            # print('value.shape', value.shape)
            m = mask[i].item()
            if value.sum() > 1e6 and m != 0:
                print('i', i)
                if entity_nums is not None:
                    if use_select_size and select_size is not None and select_units_num is not None:
                        i_1 = int(i / select_size)
                        print('i_1', i_1)
                        i_2 = i - i_1 * select_size
                        print('i_2', i_2)
                        print('entity_nums[i_1]', entity_nums[i_1])
                        print('select_units_num[i_1]', select_units_num[i_1])
                    else:
                        print('entity_nums[i]', entity_nums[i])

                print('find value large than 1e6')
                print('t', t)
                z = torch.nonzero(t, as_tuple=True)[-1]
                print('z', z)
                idx = z.item()
                print('p', p)
                q = p[idx]
                print('q', q)
                print('value', value)
                print('m', m)
                stop()

    print('x_1:', x_1) if debug else None
    print('x_1.shape:', x_1.shape) if debug else None

    x_2 = torch.sum(x_1, -1)
    print('x_2:', x_2) if debug else None
    print('x_2.shape:', x_2.shape) if debug else None

    # This mask is for each item's mask
    if mask is not None:
        x_2 = x_2 * mask

    # 第 79 轮（DS，#30）：逐样本权重（如 no_op 降权）
    if sample_weight is not None:
        x_2 = x_2 * sample_weight.reshape(x_2.shape).to(x_2.device).float()

        if outlier_remove:
            outlier_mask = (x_2 >= 1e6)
            x_2 = x_2 * ~outlier_mask
        else:
            outlier_mask = (x_2 >= 1e6)
            if outlier_mask.any() > 0:
                stop()

    if use_select_size and select_size is not None and select_units_num is not None:
        x_2 = x_2.reshape(-1, select_size)
        x_2 = torch.sum(x_2, dim=-1, keepdim=True)

        avg_type = 'None'

        if avg_type == 'None':
            x_2 = x_2  # increase the multi unit weight
        elif avg_type == 'PrefSingle':
            x_2 = x_2 / select_units_num.unsqueeze(dim=1)  # increase the single unit weight
        elif avg_type == 'Log':
            x_2 = x_2 / (torch.log(select_units_num.unsqueeze(dim=1).float()) + 1e-9)
        elif avg_type == 'Sqrt':
            x_2 = x_2 / torch.sqrt(select_units_num.unsqueeze(dim=1).float())
        else:
            x_2 = x_2

    x_4 = torch.mean(x_2)
    print('x_4:', x_4) if debug else None
    print('x_4.shape:', x_4.shape) if debug else None

    return x_4


def get_sl_loss(traj_batch, model, use_mask=True, use_eval=False):
    criterion = cross_entropy

    loss = 0
    feature_size = Feature.getSize()
    label_size = Label.getSize()

    print('traj_batch.shape:', traj_batch.shape) if debug else None
    batch_size = traj_batch.shape[0]
    seq_len = traj_batch.shape[1]

    feature = traj_batch[:, :, :feature_size].reshape(batch_size * seq_len, feature_size)
    label = traj_batch[:, :, feature_size:feature_size + label_size].reshape(batch_size * seq_len, label_size)
    is_final = traj_batch[:, :, -1:]

    state = Feature.feature2state(feature)
    print('state:', state) if debug else None

    action_gt = Label.label2action(label)
    print('action_gt:', action_gt) if debug else None

    device = next(model.parameters()).device
    print("model.device:", device) if debug else None

    # state.to(device)
    # print('state:', state) if debug else None

    # action_gt.to(device)
    # print('action_gt:', action_gt) if debug else None

    loss = torch.tensor([0.])
    loss_list = [0., 0., 0., 0., 0., 0.]
    acc_num_list = [0., 0., 0., 0., 0., 0.]    

    # we can't make them all into as a list or into ArgsActionLogits
    # if we do that, the pytorch DDP will cause a runtime error, just like the loss don't include all parameters
    # This error is strange, so we choose to use a specific loss writing schema for multi-gpu calculation.
    action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
        delay_logits, queue_logits, \
        units_logits, target_unit_logits, \
        target_location_logits, select_units_num = model.forward(state, batch_size=batch_size, 
                                                                 sequence_length=seq_len, multi_gpu_supvised_learning=True)

    print('action_pred.shape', action_pred.shape) if debug else None   

    loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, action_type_logits,
                                                             delay_logits, queue_logits, units_logits,
                                                             target_unit_logits, target_location_logits, 
                                                             select_units_num, criterion, device)

    acc_num_list = SU.get_accuracy(action_gt.action_type, action_pred, device)

    print('loss', loss) if debug else None

    return loss, loss_list, acc_num_list


# ---- 第 82 轮（DS，#40）：动作配重（把被采样搞歪的动作比例掰回目标分布） ----
# 白话：不叫 "prior" 了，就叫「动作配重表」——
#   数据里 no_op 占 81%、补农民只占 1.1%（因为转换按固定帧步长采样，采到的多是"没下命令"的帧），
#   这张表给"数据里少的动作"加权、给"多的"减权，让训练时看到的比例接近目标分布。
# 实测（DS12 82.5）：
#   训练数据的动作分布  no_op 81.0% / Build_Pylon 2.7% / Train_Probe 1.1% / Build_Gateway 0.7%
#   官方模型的实战分布  no_op 0%    / Build_Pylon 29.8% / Train_Probe 21.1% / Build_Gateway 12.9%
# 差 11-19 倍。原因是数据转换按"固定帧步长"采样，而 agent 每 8 帧才下达一条命令，
# 于是绝大多数采样帧的标签是 no_op —— 模型因此学会"发呆 + 刷叉子"。
# 这里按 importance weighting 把每类的有效频率拉回官方分布：
#   w[a] = clip( (target[a] / data[a]) ** exp , lo, hi )
# 只作用于 action_type 头（与 #30/#31 的 sample_weight 复用同一条通路）。
_ACTION_DATA_SHARE = {
    0: 0.8103, 12: 0.0760, 2: 0.0305, 35: 0.0272, 49: 0.0197,
    64: 0.0111, 36: 0.0105, 37: 0.0069, 50: 0.0056, 47: 0.0010,
}
_ACTION_TARGET_SHARE = {
    # 第 82 轮（DS，#41）修正：**不要把 Smart_unit(12) 降权**。
    # 官方模型不用 Smart（它用 Harvest_Gather 102，占 7.5%），但**我们的数据里 102 是 0%**，
    # Smart 是这套数据唯一的"采集/移动"指令。第一版把 12 的目标占比设为 0.05（权重 0.8），
    # 结果 collected 从 8,660 掉到 4,552 —— 等于把采矿功能关掉了。
    # 教训：重加权必须**以自己数据的动作空间为界**，不能照抄别人分布里的"没有的动作"。
    0: 0.010, 12: 0.076, 2: 0.110, 35: 0.280, 49: 0.150,
    64: 0.211, 36: 0.020, 37: 0.129, 50: 0.014, 47: 0.010,
}
# 第 83 轮（DS，#43）：允许用环境变量覆盖少数几个动作的目标占比，
# 这样扫参不用改代码。实测依据（DS12 82.x）：
#   我们实战 Zealot 占 35%（官方 15%）、Stalker 只有 5 次（官方 25-48 次），
#   军队更大（22 vs 14）但击杀只有官方的 81% —— 兵种构成偏"纯叉子"是最具体的嫌疑。
_TARGET_ENV_OVERRIDES = {
    "MAS_SL_TGT_NOOP": 0,
    "MAS_SL_TGT_ATTACK": 2,
    "MAS_SL_TGT_PYLON": 35,
    "MAS_SL_TGT_ZEALOT": 49,
    "MAS_SL_TGT_STALKER": 50,
    "MAS_SL_TGT_PROBE": 64,
    "MAS_SL_TGT_GATEWAY": 37,
}
for _env_k, _act_id in _TARGET_ENV_OVERRIDES.items():
    _v = os.environ.get(_env_k)
    if _v:
        _ACTION_TARGET_SHARE[_act_id] = float(_v)

_DEFAULT_TARGET = 0.001


def build_action_prior_weights(exp=0.5, lo=0.2, hi=20.0, num_actions=None):
    """返回长度 num_actions 的权重列表；未在表里的动作按默认目标占比计算。"""
    import math as _m
    n = int(num_actions)
    w = [1.0] * n
    for a in range(n):
        d = _ACTION_DATA_SHARE.get(a, 0.0005)
        t = _ACTION_TARGET_SHARE.get(a, _DEFAULT_TARGET)
        r = (t / max(d, 1e-6)) ** exp
        w[a] = max(lo, min(hi, r))
    return w


def get_sl_loss_for_tensor(features, labels, model, decrease_smart_opertaion=False,
                           return_important=False, only_consider_small=False,
                           train=True, use_masked_loss=True):

    criterion = cross_entropy

    batch_size = features.shape[0]
    seq_len = features.shape[1]

    assert batch_size == labels.shape[0]
    assert seq_len == labels.shape[1]

    features = features.reshape(batch_size * seq_len, -1)
    labels = labels.reshape(batch_size * seq_len, -1)

    state = Feature.feature2state(features)
    print('state:', state) if debug else None

    action_gt = Label.label2action(labels)
    print('action_gt:', action_gt) if debug else None

    device = next(model.parameters()).device
    print("model.device:", device) if debug else None

    # state.to(device)
    # print('state:', state) if debug else None

    # action_gt.to(device)
    # print('action_gt:', action_gt) if debug else None

    loss = torch.tensor([0.])
    loss_list = [0., 0., 0., 0., 0., 0.]
    acc_num_list = [0., 0., 0., 0., 0., 0.]    

    # we can't make them all into as a list or into ArgsActionLogits
    # if we do that, the pytorch DDP will cause a runtime error, just like the loss don't include all parameters
    # This error is strange, so we choose to use a specific loss writing schema for multi-gpu calculation.
    if True:
        # ---- 第 79 轮（DS，#26）：Simple64 的 Smart_unit -> Harvest_Gather_unit 替换 ----
        # 时间线证据（决定了它该不该开）：
        #   · 官方 sl_21-12-29 模型训练于 2021-12-29；
        #   · 这段替换由原作者在 **2022-05-02** 的提交 54aa123 引入
        #     （commit: "* a new SL settings for more chance to train a good SL model;"）；
        #   · 也就是说：**官方发布的那个模型训练时，这段替换还不存在**。
        #
        # 它做了什么：在 Simple64 图上，把 GT 的 Smart_unit（12，Protoss 的"智能动作"，
        # 会根据目标自动变成攻击/移动/采集）**整类改写成 Harvest_Gather_unit（102，纯采集）**。
        # 后果：专家的所有进攻操作在标签里都变成"采集"，模型于是只学会采集、永远不攻击。
        # 实测症状完全吻合：我们的模型 collected 6,074（会采集）但 killed=0，
        # 而官方模型 collected 9,539、killed 1,196。
        #
        # 所以复现官方效果时必须**关掉**它（默认关）。保留开关仅为做消融/兼容老日志。
        _smart_replace = os.environ.get("MAS_SL_SMART_REPLACE", "0") == "1"
        if SCHP.map_name == 'Simple64' and _smart_replace:
            smart_action_gt = torch.zeros(1, LS.action_type_encoding, device=device).float()
            smart_action_gt[0, F.Smart_unit.id.value] = 1

            gather_action_gt = torch.zeros(1, LS.action_type_encoding, device=device).float()
            gather_action_gt[0, F.Harvest_Gather_unit.id.value] = 1

            action_gt.action_type[(action_gt.action_type == smart_action_gt).all(dim=-1).bool()] = gather_action_gt

        gt_units = action_gt.units
        units_size = gt_units.shape[-1]

        bias_action_gt = torch.zeros(1, 1, units_size, device=device).float()
        bias_action_gt[0, 0, -1] = 1

        gt_select_units_num = (~(action_gt.units == bias_action_gt).all(dim=-1)).sum(dim=-1)

        print('gt_select_units_num', gt_select_units_num) if debug else None
        print('gt_select_units_num.shape', gt_select_units_num.shape) if debug else None

        action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
            delay_logits, queue_logits, \
            units_logits, target_unit_logits, \
            target_location_logits, select_units_num, \
            hidden_state, unit_types_one = model.mimic_forward(state, 
                                                               action_gt, 
                                                               gt_select_units_num,
                                                               batch_size=batch_size, 
                                                               sequence_length=seq_len, 
                                                               multi_gpu_supvised_learning=True)

        if use_masked_loss:
            # masked loss
            loss, loss_list = get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, action_type_logits,
                                                                     delay_logits, queue_logits, units_logits,
                                                                     target_unit_logits, target_location_logits, 
                                                                     select_units_num, criterion, device, 
                                                                     decrease_smart_opertaion=decrease_smart_opertaion,
                                                                     only_consider_small=only_consider_small)    
        else:
            # noraml loss
            loss, loss_list = get_classify_loss_for_multi_gpu(action_gt, action_type_logits,
                                                              delay_logits, queue_logits, units_logits,
                                                              target_unit_logits, target_location_logits, 
                                                              criterion)

    # 下面这一次 forward 只是为了算 accuracy，不参与反传。
    # 原来它是带梯度跑的，等于每个 step 建两张计算图（mimic_forward 一张 + 这里一张），
    # 白扔掉约 1/3 的前向开销。包进 no_grad 后语义不变，单步时间能降 25~30%。
    with torch.no_grad():
        action_pred, entity_nums, units, target_unit, target_location, action_type_logits, \
            delay_logits, queue_logits, \
            units_logits, target_unit_logits, \
            target_location_logits, select_units_num = model.forward(state, 
                                                                     batch_size=batch_size, 
                                                                     sequence_length=seq_len, 
                                                                     multi_gpu_supvised_learning=True)

        acc_num_list, action_equal_mask = SU.get_accuracy(action_gt.action_type, action_pred, 
                                                          device, return_important=return_important)

        location_acc = SU.get_location_accuracy(action_gt.target_location, target_location, action_equal_mask, device, 
                                                strict_comparsion=True)
        selected_acc = SU.get_selected_units_accuracy(action_gt.units, units, action_gt.action_type, action_pred,
                                                      select_units_num, action_equal_mask,
                                                      device, unit_types_one, entity_nums, strict_comparsion=True)
        targeted_acc = SU.get_target_unit_accuracy(action_gt.target_unit, target_unit, action_equal_mask,
                                                   device, strict_comparsion=True)

        acc_num_list.extend(location_acc)
        acc_num_list.extend(selected_acc)
        acc_num_list.extend(targeted_acc)

    return loss, loss_list, acc_num_list


def get_masked_classify_loss_for_multi_gpu(action_gt, action_pred, entity_nums, action_type_logits, 
                                           delay_logits, queue_logits, units_logits,
                                           target_unit_logits, target_location_logits, select_units_num,
                                           criterion, device, 
                                           decrease_smart_opertaion=False,
                                           only_consider_small=False,
                                           strict_comparsion=True, remove_none=True):
    loss = 0.

    # consider using move camera weight
    move_camera_weight = SU.get_move_camera_weight_in_SL(action_gt.action_type, 
                                                         action_pred, 
                                                         device, 
                                                         decrease_smart_opertaion=decrease_smart_opertaion,
                                                         only_consider_small=only_consider_small).reshape(-1)
    #move_camera_weight = None
    # 第 79 轮（DS，#30/#31）：按动作类别对**样本**加权
    #   MAS_SL_NOOP_WEIGHT   : no_op 样本权重（默认 1.0）。0.1 能让模型"敢动"（DS12 79.22）
    #   MAS_SL_ATTACK_WEIGHT : Attack_pt 样本权重（默认 1.0）。用于提升"把优势转成击杀"的能力
    #                           （实测负局率 33% vs 官方 18%，DS12 79.30）
    _noop_w = float(os.environ.get("MAS_SL_NOOP_WEIGHT", "1.0"))
    _attack_w = float(os.environ.get("MAS_SL_ATTACK_WEIGHT", "1.0"))
    # 白话：这里的 "prior" 就是"动作配重" —— 给每个动作一个权重，
    # 把数据里被采样搞歪的比例掰回目标分布（详见 build_action_prior_weights）。
    # 环境变量主名用 MAS_SL_ACTION_REWEIGHT；旧名 MAS_SL_ACTION_PRIOR 保留兼容。
    _use_prior = (os.environ.get("MAS_SL_ACTION_REWEIGHT",
                                 os.environ.get("MAS_SL_ACTION_PRIOR", "0")) == "1")
    _sample_w = None

    if _use_prior:
        # 第 82 轮（DS，#40）：动作先验重加权
        global _ACTION_PRIOR_CACHE
        try:
            _ACTION_PRIOR_CACHE
        except NameError:
            _ACTION_PRIOR_CACHE = None
        if _ACTION_PRIOR_CACHE is None or _ACTION_PRIOR_CACHE.numel() != LS.action_type_encoding:
            _pw = build_action_prior_weights(
                exp=float(os.environ.get("MAS_SL_PRIOR_EXP", "0.5")),
                lo=float(os.environ.get("MAS_SL_PRIOR_LO", "0.2")),
                hi=float(os.environ.get("MAS_SL_PRIOR_HI", "20.0")),
                num_actions=LS.action_type_encoding)
            _ACTION_PRIOR_CACHE = torch.tensor(_pw, dtype=torch.float32)
            if os.environ.get("MAS_SL_PRIOR_VERBOSE"):
                print("[prior] 前 60 类权重:", [round(x, 2) for x in _pw[:60:1]][:60], flush=True)
        _ids = action_gt.action_type.argmax(dim=-1)
        _tbl = _ACTION_PRIOR_CACHE.to(_ids.device)
        _sample_w = _tbl[_ids].float()

    if _noop_w != 1.0 or _attack_w != 1.0:
        _at = action_gt.action_type
        _base = _sample_w if _sample_w is not None else torch.ones_like(_at[..., 0])
        _sample_w = _base
        if _noop_w != 1.0:
            # 第 0 列是 no_op 的 one-hot
            _sample_w = torch.where(_at[..., 0] > 0.5,
                                    torch.full_like(_sample_w, _noop_w), _sample_w)
        if _attack_w != 1.0:
            # 第 2 列是 Attack_pt 的 one-hot（已实测核对 id=2）
            _sample_w = torch.where(_at[..., 2] > 0.5,
                                    torch.full_like(_sample_w, _attack_w), _sample_w)
        _numel_ratio = action_type_logits.shape[0] / max(_sample_w.numel(), 1)
        if _numel_ratio > 1.0001:      # logits 带 seq 维时对齐
            _sample_w = _sample_w.repeat(int(round(_numel_ratio)), 1).transpose(0, 1).reshape(-1)

    action_type_loss = criterion(action_gt.action_type, action_type_logits,
                                 mask=move_camera_weight, sample_weight=_sample_w)
    loss += action_type_loss

    #mask_tensor = get_one_way_mask_in_SL(action_gt.action_type, device)
    mask_tensor = SU.get_two_way_mask_in_SL(action_gt.action_type, action_pred, device, strict_comparsion=False)  # False

    # we now onsider delay loss
    delay_weight = 0.0
    if AHP.use_predict_step_mul:
        delay_weight = 1.0
    delay_loss = delay_weight * criterion(action_gt.delay, delay_logits)
    loss += delay_loss

    queue_loss = criterion(action_gt.queue, queue_logits, mask=mask_tensor[:, 2].reshape(-1))
    loss += queue_loss

    batch_size = action_gt.units.shape[0]
    select_size = action_gt.units.shape[1]
    units_size = action_gt.units.shape[-1]

    entity_nums = entity_nums
    print('entity_nums', entity_nums) if debug else None
    print('entity_nums.shape', entity_nums.shape) if debug else None

    # use extended select_size in SL for including EOF
    extended_select_size = select_size + 1

    units_mask = mask_tensor[:, 3]  # selected units is in the fourth position of units_mask
    units_mask = units_mask.unsqueeze(1).repeat(1, extended_select_size)
    units_mask = units_mask.reshape(-1)
    print('units_mask', units_mask) if debug else None
    print('units_mask.shape', units_mask.shape) if debug else None

    selected_mask = torch.arange(extended_select_size, device=device).float()
    selected_mask = selected_mask.repeat(batch_size, 1)

    # note, the select_units_num is actually gt_select_units_num here
    # we extend select_units_num by 1 to include the EFO
    selected_mask = selected_mask < (select_units_num + 1).unsqueeze(dim=1)
    selected_mask = selected_mask.reshape(-1)
    print('selected_mask', selected_mask) if debug else None
    print('selected_mask.shape', selected_mask.shape) if debug else None

    gt_units = action_gt.units.long()
    padding = torch.zeros(batch_size, 1, units_size, dtype=gt_units.dtype, device=gt_units.device)
    token = torch.tensor(AHP.max_entities - 1, dtype=padding.dtype, device=padding.device)
    padding[:, 0] = L.tensor_one_hot(token, units_size).reshape(-1)
    gt_units = torch.cat([gt_units, padding], dim=1)
    print('gt_units', gt_units) if debug else None
    print('gt_units.shape', gt_units.shape) if debug else None

    gt_units[torch.arange(batch_size), select_units_num] = L.tensor_one_hot(entity_nums, units_size).long()
    print('gt_units', gt_units) if debug else None
    print('gt_units.shape', gt_units.shape) if debug else None

    gt_units = gt_units.float()

    all_units_mask = units_mask * selected_mask  # * gt_units_mask

    # TODO: change to a proporate calculation of selected units
    # selected_units_weight = 10.
    # target_unit_weight = 1.
    # location_weight = 5.

    # 第 81 轮（DS，#36）：三个"交战头"的 loss 权重做成可配。
    # 背景：作者 2022-10-14（commit 55c37eb1）曾把这组权重写成注释掉的 10 / 1 / 5，
    # 然后改成全 1。官方 sl_21-12-29 训练于 2021-12-29，两版都在它之后，无法从 git 判断。
    # 但定向诊断（DS12 81.19）显示我们的瓶颈正是"交战效率"：
    #   army_count 16.4（官方 14.0，比官方还多）、killed 1,205（官方 1,811）、
    #   每点资源换来的击杀 0.298 vs 0.374。
    # 而这三个头（选兵 / 选目标 / 选位置）正是决定"打得准不准"的。
    # 默认全 1（不改历史行为），用 MAS_SL_UNITS_W / MAS_SL_TARGET_W / MAS_SL_LOC_W 做对照。
    selected_units_weight = float(os.environ.get("MAS_SL_UNITS_W", "1."))
    target_unit_weight = float(os.environ.get("MAS_SL_TARGET_W", "1."))
    location_weight = float(os.environ.get("MAS_SL_LOC_W", "1."))

    units_loss = selected_units_weight * criterion(gt_units.reshape(-1, units_size), units_logits.reshape(-1, units_size), 
                                                   mask=all_units_mask, debug=False, outlier_remove=True, entity_nums=entity_nums,
                                                   select_size=extended_select_size,
                                                   select_units_num=select_units_num + 1)
    loss += units_loss

    target_unit_loss = target_unit_weight * criterion(action_gt.target_unit.squeeze(-2), target_unit_logits.squeeze(-2), 
                                                      mask=mask_tensor[:, 4].reshape(-1), debug=False, outlier_remove=True, entity_nums=entity_nums)
    loss += target_unit_loss

    batch_size = action_gt.target_location.shape[0]

    target_location_loss = location_weight * criterion(action_gt.target_location.reshape(batch_size, -1),
                                                       target_location_logits.reshape(batch_size, -1), mask=mask_tensor[:, 5].reshape(-1))
    loss += target_location_loss

    return loss, [action_type_loss.item(), delay_loss.item(), queue_loss.item(), units_loss.item(), target_unit_loss.item(), target_location_loss.item()]


def get_classify_loss_for_multi_gpu(action_gt, action_type, delay, queue, units, target_unit, target_location, criterion):
    loss = 0.

    action_type_loss = criterion(action_gt.action_type, action_type)
    loss += action_type_loss

    delay_loss = criterion(action_gt.delay, delay)
    loss += delay_loss

    queue_loss = criterion(action_gt.queue, queue)
    loss += queue_loss

    units_size = action_gt.units.shape[-1]
    units_loss = criterion(action_gt.units.reshape(-1, units_size), units.reshape(-1, units_size))
    loss += units_loss

    target_unit_loss = criterion(action_gt.target_unit.squeeze(-2), target_unit.squeeze(-2))
    loss += target_unit_loss

    batch_size = action_gt.target_location.shape[0]
    target_location_loss = criterion(action_gt.target_location.reshape(batch_size, -1), target_location.reshape(batch_size, -1))
    loss += target_location_loss

    return loss, [action_type_loss.item(), delay_loss.item(), queue_loss.item(), units_loss.item(), target_unit_loss.item(), target_location_loss.item()]
