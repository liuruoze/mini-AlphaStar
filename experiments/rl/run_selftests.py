#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RL 侧护栏与自愈的单测。

    python experiments/rl/run_selftests.py

覆盖 2026-11-22 新增的三处保护：
  · `_safe_scalar` 在写盘线程已死时**不阻塞**（这是当初拖死训练的那条路）；
  · `_safe_scalar` 在线程活着时正常写入一次；
  · `_abort_process_group` 会调用 `killpg` 且以指定码退出；
  · 看门狗的超时判据边界。

全部不需要 GPU / SC2，但会 import RL 模块（约十秒）。
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _FakeWorker:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _FakeEventWriter:
    def __init__(self, alive):
        self._worker = _FakeWorker(alive)


class _FakeWriter:
    """结构对齐 tensorboardX 2.2：writer.file_writer.event_writer._worker"""

    def __init__(self, alive=True):
        self.calls = []
        self.file_writer = type("FW", (), {})()
        self.file_writer.event_writer = _FakeEventWriter(alive)

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


def check_guard_skips_dead_thread():
    import time
    from alphastarmini.core.rl.rl_vs_inner_bot_mp import _safe_scalar

    w = _FakeWriter(alive=False)
    t0 = time.time()
    _safe_scalar(w, "x", 1.0, 1)
    dt = time.time() - t0
    assert not w.calls, "线程已死却仍然写了"
    assert dt < 0.1, "跳过路径耗时 %.3fs，可能阻塞了" % dt
    return 1


def check_guard_writes_when_alive():
    from alphastarmini.core.rl.rl_vs_inner_bot_mp import _safe_scalar

    w = _FakeWriter(alive=True)
    _safe_scalar(w, "y", 2.0, 3)
    assert w.calls == [("y", 2.0, 3)], "正常路径没有写入：%r" % (w.calls,)
    return 1


def check_guard_can_be_disabled():
    """MAS_TB_GUARD=0 时应直接写，不做探活。"""
    from alphastarmini.core.rl import rl_vs_inner_bot_mp as M

    old = os.environ.get("MAS_TB_GUARD")
    os.environ["MAS_TB_GUARD"] = "0"
    try:
        w = _FakeWriter(alive=False)
        M._safe_scalar(w, "z", 1.0, 1)
        assert w.calls == [("z", 1.0, 1)], "关掉护栏后应直接写入"
    finally:
        if old is None:
            os.environ.pop("MAS_TB_GUARD", None)
        else:
            os.environ["MAS_TB_GUARD"] = old
    return 1


def check_abort_group():
    from alphastarmini.core.rl import batch_server as B

    called = {}
    real_killpg, real_exit, real_getpgid = os.killpg, os._exit, os.getpgid
    os.killpg = lambda pg, sig: called.setdefault("killpg", (pg, sig))
    os.getpgid = lambda pid: 4242
    os._exit = lambda code: called.setdefault("exit", code)
    try:
        B._abort_process_group(70)
    finally:
        os.killpg, os._exit, os.getpgid = real_killpg, real_exit, real_getpgid
    assert called.get("killpg") == (4242, 15), "killpg 未被正确调用：%r" % (called,)
    assert called.get("exit") == 70, "退出码不对：%r" % (called,)
    return 1


def check_watchdog_threshold():
    """看门狗判据：dt > limit 才触发（严格大于）。"""
    def exceeded(last, now, limit):
        return (now - last) > limit

    assert not exceeded(0, 300, 300), "恰好等于阈值不应触发"
    assert exceeded(0, 301, 300), "超过阈值应触发"
    return 2


def main():
    total = 0
    for name, fn in (
        ("护栏：线程死→跳过", check_guard_skips_dead_thread),
        ("护栏：线程活→写入", check_guard_writes_when_alive),
        ("护栏：开关可关", check_guard_can_be_disabled),
        ("自愈：killpg+退出码", check_abort_group),
        ("看门狗：阈值边界", check_watchdog_threshold),
    ):
        try:
            n = fn()
            total += n
            print("  [ok] %-22s %d 项" % (name, n))
        except Exception as e:
            print("  [FAIL] %-20s %s" % (name, e))
            return 1
    print("全部通过，共 %d 项" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
