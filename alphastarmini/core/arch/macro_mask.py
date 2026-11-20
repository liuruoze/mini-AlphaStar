# -*- coding: utf-8 -*-
"""宏观动作遮罩的共享状态（2026-11-20 hybrid）。

为什么单独一个模块：遮罩必须在 `ArchModel.forward` 里、在 action_type 被采样/取 argmax
**之前**生效；而设置它的是评测循环（通过 agent.py）。放独立模块避免 arch_model 与 agent 互相 import。

约定：默认 None（不干预），只有评测脚本在 MAS_HYBRID=1 时才会设置。
"""

_MACRO_MASK = None


def set_macro_mask(mask):
    """mask: set[int] 允许的 action_type id；None = 不限制。返回旧值。"""
    global _MACRO_MASK
    old = _MACRO_MASK
    _MACRO_MASK = None if mask is None else set(int(x) for x in mask)
    return old


def get_macro_mask():
    return _MACRO_MASK


def apply(action_type_logits, action_type):
    """把不允许的 logits 置 -1e9，并据此重算 action_type（argmax）。返回 (logits, action)。"""
    if _MACRO_MASK is None or action_type_logits is None:
        return action_type_logits, action_type
    try:
        import torch
        at = action_type_logits
        keep = torch.zeros(at.shape[-1], dtype=torch.bool, device=at.device)
        idx = torch.tensor(sorted(_MACRO_MASK), dtype=torch.long, device=at.device)
        idx = idx[(idx >= 0) & (idx < at.shape[-1])]
        if idx.numel() == 0:
            return at, action_type
        keep[idx] = True
        masked = torch.where(keep, at, at.new_full((), -1e9))
        new_a = masked.argmax(dim=-1)
        # 保持与原来 action_type 完全相同的形状（原实现可能是 [B,T] 或 [B,T,1]）
        if action_type is not None and tuple(new_a.shape) != tuple(action_type.shape):
            new_a = new_a.reshape(action_type.shape)
        return masked, new_a
    except Exception:
        return action_type_logits, action_type
