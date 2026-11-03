#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 67 轮：`Trajectory.memory` 的 hidden 契约（Kimi-07 的可执行子集）。

## Kimi-07 要的契约

- `Trajectory.memory[t]` 是第 t 步 forward 的**输入** hidden；
- teacher 第 t 步也消费同一输入 hidden；
- server 对行为策略 hidden **只推进一次**；
- local 继续推进 actor hidden，batch 的变量重绑定**不得再次驱动** server；
- 正常 episode 结束、崩溃重启、槽位复用时均清零。

## 本脚本能验什么、不能验什么

| 项 | 本脚本 |
|---|---|
| `memory[t]` 非零且**每步不同** | ✅ 直接算 |
| `memory[t]` 的**顺序**没错位 | ✅ 间接但很强：把 `memory` 当初始状态喂给 `rl_unroll`，它逐点复现 behavior logits（Δ≤2e-4，见 §66.2）；若错位一位，LSTM 状态就全错，Δ 会是 1e-2 量级 |
| 两个 actor **同时驻留同一批** | ❌ 需要在线插桩 |
| reset / 崩溃重启 / 槽位复用后清零 | ❌ 需要在线插桩 |

**不把没验的写成验过**。上面两项留作下一轮。

用法：
    python _test_r67_hidden.py log/traj_r66.pt
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch


def vec_norm(x):
    return float(torch.as_tensor(x).float().norm())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = torch.load(sys.argv[1], map_location="cpu")
    trajs = d.get("trajs") or [d["traj"]]
    print("载入 %s：%d 条轨迹，T=%d，pv=%s"
          % (sys.argv[1], len(trajs), len(trajs[0].state), d.get("policy_version", "n/a")), flush=True)

    ok_nonzero = True
    ok_diff = True
    for ti, tr in enumerate(trajs):
        mem = tr.memory                      # T 个 (h, c) 元组
        T = len(mem)
        nh = [vec_norm(m[0]) for m in mem]
        nc = [vec_norm(m[1]) for m in mem]
        # 相邻步的差
        diffs = []
        for t in range(1, T):
            dh = torch.as_tensor(mem[t][0]).float() - torch.as_tensor(mem[t - 1][0]).float()
            diffs.append(float(dh.abs().max()))
        # ⚠️ 第一版要求**所有** t 的 |h_t|>0，把 traj[0] 判成"异常"。
        #    其实 t=0 的 h 为 0 是**正确**的：轨迹从 episode 开头开始，
        #    LSTM 初始状态就是全零。要检的是"之后每一步都非零"。
        nz = all(n > 1e-8 for n in nh[1:]) if T > 1 else True
        df = all(x > 1e-9 for x in diffs)
        ok_nonzero = ok_nonzero and nz
        ok_diff = ok_diff and df
        if ti < 3:
            print("  traj[%d] T=%d  |h_t| ∈ [%.3f, %.3f]  |c_t| ∈ [%.3f, %.3f]  "
                  "相邻步 max|Δh| ∈ [%.3e, %.3e]  %s"
                  % (ti, T, min(nh), max(nh), min(nc), max(nc),
                     min(diffs), max(diffs),
                     "OK" if (nz and df) else "**异常**"), flush=True)

    # 首步是否为 0（若轨迹从 episode 开头开始，h_0 应为全零）
    h0 = vec_norm(trajs[0].memory[0][0])
    print("\n  第一条轨迹的 |h_0| = %.3e  -> %s"
          % (h0, "episode 开头（应为 0）" if h0 < 1e-8 else "**中途开始**（不是 bug，但要注意）"),
          flush=True)

    # 跨轨迹连续性：若 8 条来自同一 episode，|h| 应连续、|c| 应单调增长
    hh = [vec_norm(t.memory[-1][0]) for t in trajs]
    cc = [vec_norm(t.memory[-1][1]) for t in trajs]
    print("\n  跨轨迹连续性：各轨迹末步 |h| ∈ [%.3f, %.3f]，|c| 从 %.1f 到 %.1f（%s）"
          % (min(hh), max(hh), cc[0], cc[-1],
             "单调递增 -> 同一 episode 连续推进" if all(cc[i] <= cc[i+1] + 1e-6 for i in range(len(cc)-1))
             else "非单调"), flush=True)

    print("\n=== 结论 ===", flush=True)
    print(" 所有轨迹 hidden 非零：%s" % ok_nonzero, flush=True)
    print(" 所有轨迹 hidden 每步不同（没有卡住不动）：%s" % ok_diff, flush=True)
    print(" 顺序正确性：由 §66.2 的 Δ≤2e-4 间接支持（错位一位会是 1e-2 量级）", flush=True)
    print(" 同时驻留/崩溃重置/槽位复用：**未验**（需要在线插桩）", flush=True)
    return 0 if (ok_nonzero and ok_diff) else 2


if __name__ == "__main__":
    sys.exit(main())
