# -*- coding: utf-8 -*-
"""第 71 轮：`write` 段那 11.6 ms 到底花在哪 —— 共享数组的**锁竞争**微基准。

背景（本轮实测）：
    write 段总计 12.21 ms，但拆开只有 h/c 0.24 + 动作 0.41 = 0.65 ms。
    也就是说 95% 不在 `.cpu()` 搬运上。剩下的只可能是这一段：

        for i in idxs:            # k≈8
            self.req[i] = self.DONE
            self.ev_done[i].set()

怀疑对象：`self.req = mp.Array('i', N)` —— `mp.Array` 返回的是
**SynchronizedArray，整块数组共用一把 RLock**。每个 actor 在 `submit_and_get` 里写
`self.req[idx] = SUBMITTED`，server 在 `_collect()` 里读全部 N 个槽、在 write 里写 k 个槽，
**全都是同一把锁**。30 个 actor 高频提交时，这把锁是全局串行点。

这个基准确认/否证该假设：同样 8 次写 + 30 次读，对比
  A) mp.Array      （带锁）
  B) mp.RawArray   （无锁，裸内存）
在 8 个"打锁"进程持续骚扰下的单次操作耗时。

跑法：python experiments/_bench_req_lock.py
"""
from __future__ import print_function

import multiprocessing as mp
import time

N = 30
K = 8          # 每轮写的槽位数（≈ 平均 batch）
ROUNDS = 300   # 重复轮数
HOGS = 8       # 骚扰进程数（真实是 30 个 actor，这里取 8 已足以看出量级）


def hammer(arr, stop):
    """持续写同一个槽位，模拟 actor 的 submit。"""
    while not stop.is_set():
        for _ in range(50):
            arr[0] = 1


def bench(kind):
    stop = mp.Event()
    arr = mp.Array("i", N) if kind == "Array" else mp.RawArray("i", N)
    procs = [mp.Process(target=hammer, args=(arr, stop)) for _ in range(HOGS)]
    for p in procs:
        p.start()
    time.sleep(0.3)          # 让骚扰进程进入稳态

    # ① 写 k 个槽（server 的 write 段）
    t0 = time.time()
    for _ in range(ROUNDS):
        for i in range(K):
            arr[i] = 2
    w_us = (time.time() - t0) / (ROUNDS * K) * 1e6

    # ② 读 N 个槽（server 的 _collect）
    t0 = time.time()
    for _ in range(ROUNDS):
        got = [i for i in range(N) if arr[i] == 1]
    r_us = (time.time() - t0) / (ROUNDS * N) * 1e6

    stop.set()
    for p in procs:
        p.join(timeout=5)

    # ③ 无争用基线
    stop2 = mp.Event()
    arr2 = mp.Array("i", N) if kind == "Array" else mp.RawArray("i", N)
    t0 = time.time()
    for _ in range(ROUNDS):
        for i in range(K):
            arr2[i] = 2
    w0_us = (time.time() - t0) / (ROUNDS * K) * 1e6
    return w_us, r_us, w0_us


def main():
    print("配置：N=%d 槽，每轮写 %d，重复 %d 轮，%d 个骚扰进程\n" % (N, K, ROUNDS, HOGS))
    print("%-10s %14s %14s %14s" % ("实现", "写/次(有争用)", "读/次(有争用)", "写/次(无争用)"))
    res = {}
    for kind in ("Array", "RawArray"):
        w, r, w0 = bench(kind)
        res[kind] = (w, r, w0)
        print("%-10s %11.2f us %11.2f us %11.2f us" % (kind, w, r, w0))

    a, r_, _ = res["Array"]
    b, r2, _ = res["RawArray"]
    print("\n带锁 / 无锁 倍数：写 %.1fx， 读 %.1fx" % (a / max(b, 1e-9), r_ / max(r2, 1e-9)))
    print("按本轮真实规模折算（write 段 8 次写）：Array 约 %.1f ms，RawArray 约 %.1f ms"
          % (a * K / 1000.0, b * K / 1000.0))
    print("（对照：本轮实测 write 段 12.21 ms，其中搬运 0.65 ms）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
