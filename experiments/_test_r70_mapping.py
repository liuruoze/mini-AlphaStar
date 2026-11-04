# -*- coding: utf-8 -*-
"""第 70 轮（Kimi-07 P0）：批内 actor 映射 + hidden 生命周期契约。

**为什么必须做这个测试**（不是走形式）：

`_serve_batch` 里所有的 hidden/action 读写都用 `self.h[:, idx_t, :]`。
只要有人把它"优化"成 `self.h[:, :k, :]`（看起来等价，因为 k == len(idxs)），
当批里的 actor id **不连续**时（实测平均 batch 8.14、上限 30，几乎每批都不连续），
A 的 LSTM 状态就会被喂给 B —— 不崩、不报错、loss 也正常，
只是策略在"别人的记忆"下做动作。这类缺陷只能靠注入式测试抓。

**做法**：不需要 SC2、不需要权重、不需要 GPU。用假 agent 顶替
`batch_action_logits_by_state`，把它收到的 hidden 原样录下来，
并把输出构造成「输入 + 1」、动作构造成「输入 hidden 的 10 倍」，
于是每一行数据都能反查它来自哪个 actor。

跑法：python experiments/_test_r70_mapping.py
"""
from __future__ import print_function

import collections
import os
import sys

# 脚本放在 experiments/ 下，直接 `python experiments/xxx.py` 时 sys.path[0] 是 experiments/，
# 找不到 alphastarmini 包。把仓库根目录插进来，避免依赖 PYTHONPATH 环境变量。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes

Acts = collections.namedtuple(
    "Acts", "action_type delay queue units target_unit target_location")

MAX_UNITS = 8      # 每帧实体数上限（真实是 128，这里缩小只为让断言可读）
MAX_SELECTED = 4   # 真实是 48
H = 128            # server 里 hidden 维度写死 128


class FakeAgent(object):
    """顶替真 agent：记录收到的 hidden，返回可反查来源的确定性输出。"""

    def __init__(self):
        self.calls = []          # [(h_in(k,H), c_in(k,H), obs 条数)]

    def batch_action_logits_by_state(self, st, hidden_state=None, obs_list=None):
        h, c = hidden_state
        assert h.shape[0] == 1, "server 传进来的 hidden 应为 (1,k,H)，实际 %s" % (h.shape,)
        k = h.shape[1]
        self.calls.append((h.detach().clone(), c.detach().clone(), len(obs_list)))

        marker = h[:, :, 0].reshape(-1).long()      # 每个 actor 的 h 首元素 = 它的编号
        acts = Acts(
            action_type=marker * 10,
            delay=marker * 10 + 1,
            queue=marker * 10 + 2,
            units=(marker * 10).reshape(k, 1).repeat(1, MAX_UNITS),
            target_unit=marker * 10 + 3,
            target_location=(marker * 10).reshape(k, 1).repeat(1, 2),
        )
        nh = (h + 1.0, c + 1.0)
        sun = marker + 1
        en = marker + 2
        return None, acts, nh, sun, en


def build(N=6):
    shapes = StateShapes(entity_shape=(2, 3), map_shape=(2, 2, 2),
                         scalar_shapes=[(3,), (2,)])
    fa = FakeAgent()
    srv = AsyncBatchInferenceServer(fa, N, torch.device("cpu"), shapes,
                                    max_units=MAX_UNITS, max_selected=MAX_SELECTED,
                                    return_logits=False)
    # 给每个 actor 一个互不相同的 hidden：h[0, i, :] = i+1
    for i in range(N):
        srv.h[:, i, :] = float(i + 1)
        srv.c[:, i, :] = float(i + 1)
    return srv, fa


def main():
    # 用可变容器而不是裸变量：嵌套函数里对裸变量赋值要写 nonlocal，
    # 写成 global 会指向模块级（不存在）的 ok，直接 NameError。
    res = [True]

    def chk(name, cond, detail):
        res[0] = res[0] and bool(cond)
        print("  %-52s %s   %s" % (name, "PASS" if cond else "FAIL", detail), flush=True)

    print("=== [A] 非连续、乱序的 idx：hidden 必须按 idx 取，不能按位置取 ===", flush=True)
    srv, fa = build()
    order = [3, 0, 5]
    srv._serve_batch(order)
    h_in = fa.calls[0][0][0]                       # (k, H)
    got = [float(h_in[j, 0]) for j in range(len(order))]
    want = [float(i + 1) for i in order]
    chk("喂进网络的 hidden 行 == 对应 actor 的 h", got == want,
        "idx %s -> 收到 %s（期望 %s）" % (order, got, want))

    print("\n=== [B] h_prev 存的是「本步输入」，不是「本步输出」 ===", flush=True)
    prev3 = float(srv.h_prev[:, 3, :].abs().max())
    now3 = float(srv.h[:, 3, :].abs().max())
    chk("h_prev[3] == 输入值 4.0", prev3 == 4.0, "h_prev=%.1f" % prev3)
    chk("h[3] == 输出值 5.0（已推进）", now3 == 5.0, "h=%.1f" % now3)

    print("\n=== [C] 只写回被服务的行，其它 actor 不被污染 ===", flush=True)
    untouched = [float(srv.h[:, i, :].abs().max()) for i in (1, 2, 4)]
    chk("未参与的 actor 1/2/4 保持 2.0/3.0/5.0", untouched == [2.0, 3.0, 5.0],
        str(untouched))

    print("\n=== [D] 6 个动作字段落到正确的行 ===", flush=True)
    chk("a_type[3]=40 / [0]=10 / [5]=60",
        [int(srv.a_type[i]) for i in order] == [40, 10, 60],
        str([int(srv.a_type[i]) for i in order]))
    chk("a_units[3] 前 4 位全 40", set(srv.a_units[3, :MAX_SELECTED].tolist()) == {40},
        str(srv.a_units[3, :MAX_SELECTED].tolist()))
    chk("a_tloc[5] == [60, 60]", srv.a_tloc[5].tolist() == [60, 60],
        str(srv.a_tloc[5].tolist()))
    chk("a_sun/a_en 落位（sun[0]=2, en[0]=3）",
        (int(srv.a_sun[0]), int(srv.a_en[0])) == (2, 3),
        "sun=%d en=%d" % (int(srv.a_sun[0]), int(srv.a_en[0])))

    print("\n=== [E] 同一 actor 连续两轮：hidden 要连续推进 ===", flush=True)
    # ⚠️ 不能写死成 5.0：第 1 轮服务 [3,0,5] 之后 h[0] 是 1.0+1=2.0，5.0 是 h[3]。
    #    第一版就是这里把断言写死了，把正确实现判成 FAIL（探针错，第 2 类错误）。
    before = float(srv.h[:, 0, :].abs().max())
    srv._serve_batch([0])
    h2 = float(fa.calls[-1][0][0, 0, 0])
    chk("第 2 轮输入 == 第 1 轮输出（%.1f）" % before, h2 == before, "%.1f" % h2)
    chk("第 2 轮输出 == 输入 + 1",
        float(srv.h[:, 0, :].abs().max()) == before + 1,
        "%.1f" % float(srv.h[:, 0, :].abs().max()))

    print("\n=== [F] reset_hidden 之后首步 hidden 必须全 0（在线计数器）===", flush=True)
    before_ok, before_bad = srv.h0_ok, srv.h0_bad
    srv.reset_hidden(0)
    srv._serve_batch([0])
    chk("reset 后首步 h_prev == 0", float(srv.h_prev[:, 0, :].abs().max()) == 0.0,
        "%.3f" % float(srv.h_prev[:, 0, :].abs().max()))
    chk("计数器 ok +1 / bad +0",
        (srv.h0_ok - before_ok, srv.h0_bad - before_bad) == (1, 0),
        "ok+%d bad+%d" % (srv.h0_ok - before_ok, srv.h0_bad - before_bad))

    print("\n=== [G] 反例自检：不清 hidden 直接换 episode，计数器必须能抓到 ===", flush=True)
    # 人为把 reset_count 加 1 但不真清零 —— 模拟"上层漏调 reset_hidden、又标记成新局"
    srv.h[:, 2, :] = 7.0
    srv.reset_count[2] += 1
    before_bad = srv.h0_bad
    srv._serve_batch([2])
    chk("非零 hidden 被计入 bad（证明探针不是恒真）",
        srv.h0_bad == before_bad + 1,
        "bad %d -> %d" % (before_bad, srv.h0_bad))

    print("\n=== [H] mark_exited 后 hidden 归零（槽位复用契约）===", flush=True)
    srv.h[:, 4, :] = 9.0
    srv.c[:, 4, :] = 9.0
    srv.h_prev[:, 4, :] = 9.0
    srv.mark_exited(4)
    vals = [float(srv.h[:, 4, :].abs().max()), float(srv.c[:, 4, :].abs().max()),
            float(srv.h_prev[:, 4, :].abs().max())]
    chk("h/c/h_prev 全 0", vals == [0.0, 0.0, 0.0], str(vals))
    chk("req[4] == EXITED", srv.req[4] == srv.EXITED, str(int(srv.req[4])))

    print("\n=== 结论 ===", flush=True)
    if res[0]:
        print(" 全部通过：批内映射按 idx、h_prev 是输入、reset/exit 后 hidden 归零，"
              "且反例能被计数器抓到。", flush=True)
        return 0
    print(" 存在 FAIL —— 不要用'看起来正常'覆盖它。", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
