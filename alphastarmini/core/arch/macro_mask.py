# -*- coding: utf-8 -*-
"""宏观动作遮罩 / 软偏置的共享状态（2026-11-20 hybrid）。

为什么单独一个模块：干预必须在 `ArchModel.forward` 里、在 action_type 被采样
**之前**生效；而设置它的是评测循环（通过 agent.py）。放独立模块避免互相 import。

两种干预方式（互斥，遮罩优先）：
  · set_macro_mask(ids)       硬遮罩：不在集合内的 logits 置 -1e9
  · set_macro_bias(ids, w)    软偏置：给集合内 logits 加 w，不排除任何动作

默认都为 None（不干预），只有评测脚本在 MAS_HYBRID=1 时才会设置。
"""

_MACRO_MASK = None
_MACRO_BIAS = None


def set_macro_mask(mask):
    """mask: set[int] 允许的 action_type id；None = 不限制。返回旧值。"""
    global _MACRO_MASK
    old = _MACRO_MASK
    _MACRO_MASK = None if mask is None else set(int(x) for x in mask)
    return old


def get_macro_mask():
    return _MACRO_MASK


def set_macro_bias(ids, weight):
    """软偏置：给这些 action_type 的 logits 加 weight，不排除其它动作。"""
    global _MACRO_BIAS
    old = _MACRO_BIAS
    _MACRO_BIAS = None if not ids else (set(int(x) for x in ids), float(weight))
    return old


def get_macro_bias():
    return _MACRO_BIAS


def _reshape_like(new_a, action_type):
    if action_type is not None and tuple(new_a.shape) != tuple(action_type.shape):
        return new_a.reshape(action_type.shape)
    return new_a


def apply(action_type_logits, action_type):
    """按当前设置干预 logits 并重算 action_type。返回 (logits, action)。"""
    if action_type_logits is None:
        return action_type_logits, action_type
    try:
        import torch
        at = action_type_logits

        if _MACRO_BIAS is not None and _MACRO_MASK is None:
            ids, w = _MACRO_BIAS
            idx = torch.tensor(sorted(ids), dtype=torch.long, device=at.device)
            idx = idx[(idx >= 0) & (idx < at.shape[-1])]
            if idx.numel() == 0:
                return at, action_type
            delta = torch.zeros(at.shape[-1], dtype=at.dtype, device=at.device)
            delta[idx] = w
            biased = at + delta
            return biased, _reshape_like(biased.argmax(dim=-1), action_type)

        if _MACRO_MASK is None:
            return at, action_type

        keep = torch.zeros(at.shape[-1], dtype=torch.bool, device=at.device)
        idx = torch.tensor(sorted(_MACRO_MASK), dtype=torch.long, device=at.device)
        idx = idx[(idx >= 0) & (idx < at.shape[-1])]
        if idx.numel() == 0:
            return at, action_type
        keep[idx] = True
        masked = torch.where(keep, at, at.new_full((), -1e9))
        return masked, _reshape_like(masked.argmax(dim=-1), action_type)
    except Exception:
        return action_type_logits, action_type
