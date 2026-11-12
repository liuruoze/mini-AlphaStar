#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实体槽重排：is_selected 优先 -> 我方 -> 其余原序（稳定分区）。

背景
----
`entity_encoder.py:112` 按 `raw_units` 原序硬截前 128 槽。实测 bvb5（AI 局）：
33.56% 的帧有我方单位被丢（6.552 个/帧），10,000-20,000 步段 61.79%、
>20,000 步 100%。本模块在截断前重排，保证「当前选中」和「我方」优先入槽。

约束
----
- 只使用观测可得字段（`is_selected`、`alliance`），训练与评测必须共用。
- 不能用标签 `selected_units` 排序：那是模型要预测的答案，会构成标签泄漏。
- 动作最多选 `max_selected=12` 个单位，所以 selected 层最多保 12 个；
  更多框选仍会被截，这是硬上限。
"""
from __future__ import annotations


def _is_selected(u):
    try:
        return bool(u.is_selected)
    except Exception:
        return False


def _is_my(u):
    try:
        return int(u.alliance) == 1
    except Exception:
        return False


def prune_keep_my(my_flags, max_entities):
    """**不重排**，只在超限时优先删掉「非我方」，返回升序原下标。

    与 reorder_indices 的本质差别：本函数**保持原序**，因此模型学到的
    「我方在某段固定槽位」这一位置先验完全不变。

    - n <= max_entities：返回 range(n)，输出与不启用时**逐位相同**
      （人类 90 局 len(raw_units) 实测 max 108 < 128 ⇒ 全部走这里）
    - n >  max_entities：保留全部我方 + 原序靠前的非我方，仍按原序排列

    背景（J136 实测）：reorder_indices 把我方从槽 32-79 搬到槽 0-31，
    破坏了位置先验——模型输出槽 47 时在 myall 顺序下多数帧越过我方上限，
    命中中立单位，30k 步全灭。prune 只删不搬，避免这个问题。
    """
    n = len(my_flags)
    cap = int(max_entities)
    if n <= cap:
        return list(range(n))
    keep = [i for i in range(n) if my_flags[i]]
    remain = cap - len(keep)
    if remain > 0:
        for i in range(n):
            if not my_flags[i]:
                keep.append(i)
                remain -= 1
                if remain == 0:
                    break
    keep.sort()
    return keep[:cap]


def reorder_indices(sel_flags, my_flags, max_entities):
    """稳定分区，返回原下标序列：selected -> 我方 -> 其余，截到 max_entities。

    入参是两个等长布尔序列（或可迭代），不依赖 numpy/protobuf 类型；
    调用方用 `raw_units[order]` 保持原数组类型（pysc2 的 raw_units 是
    二维结构化数组，reverse_raw_action 依赖 `[:, FeatureUnit.tag]`）。
    """
    sel, my, rest = [], [], []
    for i, (s, m) in enumerate(zip(sel_flags, my_flags)):
        if s:
            sel.append(i)
        elif m:
            my.append(i)
        else:
            rest.append(i)
    return (sel + my + rest)[:int(max_entities)]


def reorder_entities(raw_units, max_entities):
    """稳定分区：selected -> 我方 -> 其余，截到 max_entities（list 场景）。"""
    sel, my, rest = [], [], []
    for u in raw_units:
        if _is_selected(u):
            sel.append(u)
        elif _is_my(u):
            my.append(u)
        else:
            rest.append(u)
    return (sel + my + rest)[:int(max_entities)]


def reorder_order(raw_units, max_entities):
    """返回 (new_units, old_index_of_new)。

    `old_index_of_new[k]` = 新槽 k 对应原列表的下标；标签重映射用：
    `new_pos = {old: new for new, old in enumerate(old_index_of_new)}`。
    """
    sel, my, rest = [], [], []
    for i, u in enumerate(raw_units):
        if _is_selected(u):
            sel.append(i)
        elif _is_my(u):
            my.append(i)
        else:
            rest.append(i)
    order = (sel + my + rest)[:int(max_entities)]
    return [raw_units[i] for i in order], order
