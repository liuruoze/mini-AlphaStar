#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LayerNorm LSTM 的验证套件（不需要 SC2，几秒跑完）。

检查项：
  T1 off 模式与"直接调用 nn.LSTM"逐元素一致（回归：不得改变历史行为）
  T2 out 模式等于手算 LayerNorm(nn.LSTM(x))
  T3 gate 模式形状/隐状态正确
  T4 四种模式下，LSTM 与 LN 参数都能拿到非零梯度
  T5 常数输入不产生 NaN（eps 保护）
  T6 退化形状（B=1,T=1）不崩

用法：python experiments/test_ln_lstm.py
"""

import os
import os
import sys

# 允许从任意工作目录运行：以脚本所在目录的上一级作为仓库根
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from alphastarmini.core.arch.core import Core


def build(mode, gamma="1.0"):
    os.environ["MAS_LSTM_LN"] = mode
    os.environ["MAS_LN_GAMMA"] = gamma
    return Core(embedding_dim=32, hidden_dim=24, batch_size=2, sequence_length=5, n_layers=2)


def run(core, x, hidden=None):
    return core.forward_lstm(x, hidden)


def main():
    torch.manual_seed(0)
    B, T, D, H = 2, 5, 32, 24
    x = torch.randn(B, T, D)
    zeros = torch.zeros(2, B, H)
    hidden = (zeros, zeros.clone())

    results = []

    # T1: off 与裸 nn.LSTM 一致
    core = build("off")
    ref = nn.LSTM(input_size=D, hidden_size=H, num_layers=2, batch_first=True)
    ref.load_state_dict({k.replace("lstm.", ""): v for k, v in core.state_dict().items() if k.startswith("lstm.")})
    with torch.no_grad():
        out_core, (h_core, c_core) = run(core, x, hidden)
        out_ref, (h_ref, c_ref) = ref(x, hidden)
    assert out_core.shape == out_ref.shape, (out_core.shape, out_ref.shape)
    d = (out_core - out_ref).abs().max().item()
    results.append(("T1 off==nn.LSTM", d < 1e-6, "max_diff=%.2e" % d))

    # T2: out 等于手算 LN(LSTM(x))
    core = build("out")
    ref = nn.LSTM(input_size=D, hidden_size=H, num_layers=2, batch_first=True)
    ref.load_state_dict({k.replace("lstm.", ""): v for k, v in core.state_dict().items() if k.startswith("lstm.")})
    with torch.no_grad():
        out_core, _ = run(core, x, hidden)
        out_ref, _ = ref(x, hidden)
        manual = nn.functional.layer_norm(out_ref, (H,), core.out_norm.norm.weight,
                                          core.out_norm.norm.bias, 1e-5)
    d = (out_core - manual).abs().max().item()
    results.append(("T2 out==manual LN", d < 1e-5, "max_diff=%.2e" % d))

    # T3: gate 形状
    core = build("gate")
    out, (h, c) = run(core, x, hidden)
    ok = out.shape == (B, T, H) and h.shape == (2, B, H) and c.shape == (2, B, H)
    results.append(("T3 gate shape", ok, str(tuple(out.shape))))

    # T4: 梯度
    for mode in ("off", "out", "pre", "gate"):
        core = build(mode)
        out, _ = run(core, x, hidden)
        loss = out.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in core.parameters() if p.grad is not None]
        nonzero = all(g.abs().sum().item() > 0 for g in grads) if grads else False
        ln_has_grad = all(
            (p.grad is not None and p.grad.abs().sum().item() > 0)
            for n, p in core.named_parameters() if ".norm." in n or "ln_" in n
        )
        results.append(("T4 grad %s" % mode, nonzero and ln_has_grad,
                        "params_with_grad=%d ln_grad=%s" % (len(grads), ln_has_grad)))

    # T5: 常数输入不 NaN
    for mode in ("out", "gate"):
        core = build(mode)
        const = torch.ones(B, T, D)
        out, _ = run(core, const, hidden)
        ok = torch.isfinite(out).all().item()
        results.append(("T5 const-input %s" % mode, ok, "nan=%s" % (not ok)))

    # T6: 退化形状
    for mode in ("off", "out", "pre", "gate"):
        core = build(mode)
        x1 = torch.randn(1, 1, D)
        z1 = torch.zeros(2, 1, H)
        out, _ = run(core, x1, (z1, z1.clone()))
        ok = out.shape == (1, 1, H) and torch.isfinite(out).all().item()
        results.append(("T6 shape B1T1 %s" % mode, ok, str(tuple(out.shape))))

    print("%-24s %-6s %s" % ("check", "pass", "detail"))
    bad = 0
    for name, ok, detail in results:
        print("%-24s %-6s %s" % (name, "OK" if ok else "FAIL", detail))
        bad += 0 if ok else 1
    print("\n%d/%d passed" % (len(results) - bad, len(results)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
