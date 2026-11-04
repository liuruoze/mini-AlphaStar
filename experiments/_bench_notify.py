# -*- coding: utf-8 -*-
"""第 72 轮：把"通知 actor"的三种机制量清楚（K=8 与 K=30 两档）。

背景（第 71 轮实测，不是推测）：
    30 min 长跑 write 段 12.21 ms，其中 h/c 0.24 + 动作 0.41，**通知占 11.06**。
    每次 `Event.set()` 折合 1.4~1.7 ms。

为什么 Event 慢：CPython 的 `multiprocessing.Event` 内部是 `Condition`，
`set()` 要 `with self._cond` 加锁、`notify_all()` 释放内部信号量；
而 actor 侧的 `wait()`/`clear()` 也要同一把锁 —— 父进程每轮要和 actor **交接这把锁 k 次**。

候选方案：
    EVENTS : k 个 mp.Event，各自 set()                  ← 当前实现
    COND   : 1 个 mp.Condition + 共享 req，notify_all()  ← 一次广播（惊群）
    SEM    : k 个 mp.Semaphore(0)，各自 release()        ← 一次 sem_post，无 Condition 锁

判据：父进程侧 ms/轮（server 正是这个角色，它 set 完就走，不等 actor）。
同时打印子进程实际被唤醒次数，防止出现"没人等所以很快"的假象。

跑法：python experiments/_bench_notify.py
"""
from __future__ import print_function

import multiprocessing as mp
import time

ROUNDS = 400


# ---------------- 子进程 ----------------

def child_events(evs, i, stop, ack):
    while not stop.is_set():
        if evs[i].wait(0.05):
            evs[i].clear()
            with ack.get_lock():
                ack.value += 1


def child_cond(cond, req, i, stop, ack):
    while not stop.is_set():
        with cond:
            if not cond.wait(0.05):
                continue
            done = (req[i] == 9)
        if done:
            req[i] = 0
            with ack.get_lock():
                ack.value += 1


def child_sem(sem, i, stop, ack):
    while not stop.is_set():
        if sem[i].acquire(timeout=0.05):
            with ack.get_lock():
                ack.value += 1


# ---------------- 父进程侧基准 ----------------

def bench(mode, K):
    stop = mp.Event()
    ack = mp.Value("i", 0)
    procs = []

    if mode == "EVENTS":
        evs = [mp.Event() for _ in range(K)]
        for i in range(K):
            p = mp.Process(target=child_events, args=(evs, i, stop, ack), daemon=True)
            p.start()
            procs.append(p)
        time.sleep(0.5)
        t0 = time.time()
        for _ in range(ROUNDS):
            for i in range(K):
                evs[i].set()
        dt = time.time() - t0

    elif mode == "COND":
        cond = mp.Condition()
        req = mp.Array("i", K)
        for i in range(K):
            p = mp.Process(target=child_cond, args=(cond, req, i, stop, ack), daemon=True)
            p.start()
            procs.append(p)
        time.sleep(0.5)
        t0 = time.time()
        for _ in range(ROUNDS):
            with cond:
                for i in range(K):
                    req[i] = 9
                cond.notify_all()
        dt = time.time() - t0

    else:  # SEM
        sem = [mp.Semaphore(0) for _ in range(K)]
        for i in range(K):
            p = mp.Process(target=child_sem, args=(sem, i, stop, ack), daemon=True)
            p.start()
            procs.append(p)
        time.sleep(0.5)
        t0 = time.time()
        for _ in range(ROUNDS):
            for i in range(K):
                sem[i].release()
        dt = time.time() - t0

    # ⚠️ 先让子进程把积压的唤醒消化完，再取 ack。
    #    第一版直接 stop.set() 后读 ack，结果 EVENTS（慢）能收到满额 3200，
    #    而 COND/SEM（快）只收到 42~161 —— 不是没送到，是父进程跑到终点时
    #    子进程还没消化，stop 一置它们就退出了。这会让"快"看起来像"没送到"。
    time.sleep(0.5)
    acked = int(ack.value)
    stop.set()
    for p in procs:
        p.join(timeout=3)
    return dt / ROUNDS * 1000.0, acked


def main():
    print("父进程每轮通知 K 个 actor，重复 %d 轮；子进程真实阻塞等待。\n" % ROUNDS)
    for K in (8, 30):
        print("=== K=%d ===" % K)
        print("%-8s %12s %14s %14s" % ("方式", "ms/轮", "每次通知 us", "子进程收到"))
        base = None
        for mode in ("EVENTS", "COND", "SEM"):
            ms, acked = bench(mode, K)
            if base is None:
                base = ms
            print("%-8s %9.3f ms %11.1f us %12d   %s"
                  % (mode, ms, ms * 1000.0 / K, acked,
                     "" if mode == "EVENTS" else "%.1fx" % (base / ms)))
        print()
    print("读法：EVENTS 与 SEM 都是 K 次单播，COND 是 1 次广播（惊群）。")
    print("      COND 的 K=30 若明显劣于 K=8，说明惊群代价随 waiter 数上升。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
