#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LayerNorm LSTM 的三种可选实现，供 Core 通过 MAS_LSTM_LN 切换。

设计要点（2026-11-19）：
- off  : 完全等价于原来的 nn.LSTM，任何情况下都不得改变历史行为。
- out  : 只对 LSTM 的输出做 LayerNorm（今天上午试过的那种，会塌缩）。
- pre  : 只对 LSTM 的输入做 LayerNorm（不动输出尺度，最温和）。
- gate : 门级 LayerNorm（DI-Star 的做法）：i/h 两路预激活各自归一化，cell 再归一化。
         注意：这是纯 Python 逐时间步循环，比 nn.LSTM 慢，且不做 jit。

所有这些实现都必须满足：输出形状 [B*T, H]、隐状态 (h, c) 形状 [L, B, H]，
与 Core 现有接口一致。
"""

import torch
import torch.nn as nn


class OutLN(nn.Module):
    """输出归一化。gamma 可缩放初始值，用于验证"尺度错配"假设。"""

    def __init__(self, hidden_size, gamma_init=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        if gamma_init != 1.0:
            with torch.no_grad():
                self.norm.weight.fill_(gamma_init)

    def forward(self, x):
        return self.norm(x)


class PreLN(nn.Module):
    """输入归一化：只改输入分布，不动输出尺度。"""

    def __init__(self, input_size):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x):
        return self.norm(x)


class LNCell(nn.Module):
    """单个 LayerNorm LSTM cell（DI-Star 风格，Apache-2.0 思路重写）。"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.ln_i = nn.LayerNorm(4 * hidden_size)
        self.ln_h = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        # 与 nn.LSTM 一致：uniform(-1/sqrt(H), 1/sqrt(H))
        k = 1.0 / (hidden_size ** 0.5)
        with torch.no_grad():
            self.weight_ih.uniform_(-k, k)
            self.weight_hh.uniform_(-k, k)

    def forward(self, x, state):
        hx, cx = state
        gates = self.ln_i(torch.mm(x, self.weight_ih.t())) + self.ln_h(torch.mm(hx, self.weight_hh.t()))
        i, f, g, o = gates.chunk(4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        cy = self.ln_c(f * cx + i * g)
        hy = o * torch.tanh(cy)
        return hy, (hy, cy)


class GatedLSTM(nn.Module):
    """多层门级 LN LSTM，输入 [B, T, D]，隐状态 (h, c) 各 [L, B, H]。"""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        cells = []
        for layer in range(num_layers):
            cells.append(LNCell(input_size if layer == 0 else hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, hidden=None):
        batch, steps, _ = x.shape
        if hidden is None:
            zeros = x.new_zeros(self.num_layers, batch, self.hidden_size)
            hidden = (zeros, zeros.clone())
        h, c = hidden
        layer_input = x
        new_h, new_c = [], []
        for layer, cell in enumerate(self.cells):
            hl, cl = h[layer], c[layer]
            outputs = []
            for t in range(steps):
                hl, (hl, cl) = cell(layer_input[:, t, :], (hl, cl))
                outputs.append(hl)
            layer_input = torch.stack(outputs, dim=1)
            new_h.append(hl)
            new_c.append(cl)
        return layer_input, (torch.stack(new_h), torch.stack(new_c))
