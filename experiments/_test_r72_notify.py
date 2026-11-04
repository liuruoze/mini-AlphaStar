# -*- coding: utf-8 -*-
"""第 72 轮：信号量通知机制的握手测试。

**为什么需要它**：第 72 轮把"通知 actor"从 `mp.Event` 换成可选 `mp.Semaphore`。
这个改动看着只是换了个原语，但它有一个**独有的失效模式**：

    信号量是**计数型**的。Event 是电平型（set 一次，之后 wait 都立即返回），
    而 Semaphore 会把 release 累加。如果某次 release 没有对应的 acquire
    （actor 死了、或 mark_exited 时残留），下一次 acquire **会立刻返回**，
    actor 于是读到"还没算好的结果" —— 不报错、不崩，只是一步旧数据。

所以测试必须证明两件事：
  [A] 正常握手：submit -> server 服务 -> acquire 返回，且拿到的是**本次**结果
  [B] 残留令牌：退出/重启后必须被排空，不能让下一次 acquire 空转

跑法：python experiments/_test_r72_notify.py
"""
from __future__ import print_function

import collections
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes

Acts = collections.namedtuple(
    "Acts", "action_type delay queue units target_unit target_location")

N = 4
MAX_UNITS = 4
MAX_SELECTED = 2


class FakeAgent(object):
    def __init__(self):
        self.calls = 0

    def batch_action_logits_by_state(self, st, hidden_state=None, obs_list=None):
        h, c = hidden_state
        k = h.shape[1]
        self.calls += 1
        marker = torch.full((k,), float(self.calls))      # 每次前向换一个标记
        acts = Acts(
            action_type=marker.long(),
            delay=marker.long(),
            queue=marker.long(),
            units=marker.long().reshape(k, 1).repeat(1, MAX_UNITS),
            target_unit=marker.long(),
            target_location=marker.long().reshape(k, 1).repeat(1, 2),
        )
        return None, acts, (h + 1.0, c + 1.0), marker.long(), marker.long()


def build(mode):
    os.environ["MAS_NOTIFY"] = mode
    shapes = StateShapes(entity_shape=(2, 3), map_shape=(2, 2, 2),
                         scalar_shapes=[(3,), (2,)])
    fa = FakeAgent()
    srv = AsyncBatchInferenceServer(fa, N, torch.device("cpu"), shapes,
                                    max_units=MAX_UNITS, max_selected=MAX_SELECTED,
                                    return_logits=False)
    return srv, fa


def main():
    res = [True]

    def chk(name, cond, detail):
        res[0] = res[0] and bool(cond)
        print("  %-46s %s   %s" % (name, "PASS" if cond else "FAIL", detail), flush=True)

    print("=== [A] 正常握手（MAS_NOTIFY=sem）===", flush=True)
    srv, fa = build("sem")
    chk("已启用信号量", srv.sem_done is not None, str(srv.notify_mode))

    got = {}

    def worker():
        # read_action 返回 7 元组：(a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n)
        act = srv.submit_and_wait(2)
        got["a"] = int(act[0])

    th = threading.Thread(target=worker)
    th.start()
    # 等它真的阻塞在 acquire 上（否则测的是"没等"）
    for _ in range(200):
        if srv.req[2] == srv.SUBMITTED:
            break
        time.sleep(0.005)
    chk("actor 已提交且阻塞", srv.req[2] == srv.SUBMITTED, "req=%d" % int(srv.req[2]))
    srv._serve_batch([2])
    th.join(timeout=5)
    chk("线程已返回（未被阻塞）", not th.is_alive(), "alive=%s" % th.is_alive())
    chk("拿到的是本次结果（marker=1）", got.get("a") == 1, "action_type=%r" % got.get("a"))
    chk("无残留令牌", srv.sem_done[2].acquire(False) is False, "acquire(False) 应为 False")

    print("\n=== [B] 残留令牌必须被 mark_exited 排空 ===", flush=True)
    srv.sem_done[1].release()          # 人为留一个没人消费的令牌
    leaked = srv.sem_done[1].acquire(False)
    chk("先确认令牌确实存在", leaked is True, "acquire(False)=%s" % leaked)
    srv.sem_done[1].release()          # 再留一个
    srv.mark_exited(1)
    chk("mark_exited 后令牌被排空", srv.sem_done[1].acquire(False) is False,
        "残留会导致下一次 acquire 空转")

    print("\n=== [C] 默认仍是 event（历史行为不变）===", flush=True)
    srv2, _ = build("event")
    chk("默认/显式 event 时 sem_done 为 None", srv2.sem_done is None, str(srv2.notify_mode))
    chk("req 状态机未变", srv2.req[0] == 0, "IDLE=0")

    print("\n=== 结论 ===", flush=True)
    if res[0]:
        print(" 信号量握手正常、无残留令牌、默认机制未变。", flush=True)
        return 0
    print(" 存在 FAIL —— 计数型信号量的残留会让 actor 读到未完成的结果，不可上线。", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
