# -*- coding: utf-8 -*-
"""第 73 轮（Kimi-05）：`SharedSnapshot` 的正确性与抗撕裂测试。

**要证明什么**

1. 发布出去的字节，读回来逐元素相同（含 int64 / bool 等非 float 缓冲区）；
2. `hash` 与**内容**严格对应 —— 读者拿到 (ver, hash, sd) 后自己重算 md5 必须相等；
   这条是 Kimi-05 的"快照哈希与发布版本一一对应"的可执行版本；
3. **不会读到半新半旧**：一边高频发布、一边高频读，读者每次校验都必须通过；
4. 内容不变时重复发布：hash 不变、版本 +1（版本是"发布次数"，不是"内容变化次数"）。

为什么第 3 条最重要：如果读和写没有互斥，读者可能拿到"前 300 个张量是 v1、
后 400 个是 v2"的混合体 —— 它形状合法、能 load、能 forward，只是策略是虚构的。

跑法：python experiments/_test_r73_snapshot.py
"""
from __future__ import print_function

import hashlib
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphastarmini.core.rl.batch_server import SharedSnapshot

SPEC = [
    ("w", (8, 4), torch.float32),
    ("b", (8,), torch.float32),
    ("num_batches_tracked", (), torch.int64),
    ("mask", (3, 5), torch.bool),
]


def make_sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "w": torch.randn(8, 4, generator=g),
        "b": torch.randn(8, generator=g),
        "num_batches_tracked": torch.tensor(seed, dtype=torch.int64),
        "mask": torch.randint(0, 2, (3, 5), generator=g).bool(),
    }


def digest(snap, sd):
    """按 meta 顺序重算内容哈希 —— 与 publish 的算法一致，但**独立实现**。"""
    md5 = hashlib.md5()
    for k, shape, npdt, off, nb, n in snap.meta:
        md5.update(sd[k].contiguous().numpy().reshape(-1).view("uint8").tobytes())
    return md5.hexdigest()[:32]


def reader(snap, out_q, stop):
    """持续读，直到父进程置 stop。

    ⚠️ 第一版写的是"成功读满 rounds 次才退出"，于是子进程一直在等
    "比自己已见版本更新"的版本；父进程发布完就停了，它永远等不到 -> 看起来像死锁。
    真正的原因是我的终止条件依赖"还会继续发布"，而发布是有限次的。
    """
    seen, bad, last, idle = 0, 0, -1, 0
    while not stop.is_set():
        got = snap.read_if_newer(last)
        if got is None:
            idle += 1
            if idle > 200000:
                break
            time.sleep(0.001)
            continue
        idle = 0
        ver, h, sd = got
        if ver <= last:
            bad += 1
            out_q.put(("mono", ver, last))
        if digest(snap, sd) != h:
            bad += 1
            out_q.put(("tear", ver, h))
        last = ver
        seen += 1
    out_q.put(("done", seen, bad))


def main():
    res = [True]

    def chk(name, cond, detail):
        res[0] = res[0] and bool(cond)
        print("  %-44s %s   %s" % (name, "PASS" if cond else "FAIL", detail), flush=True)

    print("=== [A] 基本读写（float32 / int64 / bool）===", flush=True)
    snap = SharedSnapshot(make_sd(0))
    sd1 = make_sd(1)
    ver1, h1 = snap.publish(sd1)
    got = snap.read()
    chk("版本从 1 开始", ver1 == 1 and got[0] == 1, "ver=%s" % (got[0],))
    ok = all(torch.equal(got[2][k], sd1[k]) for k in sd1)
    chk("四个张量逐元素相同", ok,
        "dtype=%s" % {k: str(sd1[k].dtype) for k in sd1})

    print("\n=== [B] hash 与内容一一对应（独立重算）===", flush=True)
    chk("读者重算的 md5 == 发布的 hash", digest(snap, got[2]) == h1,
        "h=%s 重算=%s" % (h1, digest(snap, got[2])))

    print("\n=== [C] 内容不变时重复发布：hash 不变 / ver+1 ===", flush=True)
    ver2, h2 = snap.publish(sd1)
    chk("hash 不变", h2 == h1, "%s vs %s" % (h1, h2))
    chk("版本 +1", ver2 == ver1 + 1, "%d -> %d" % (ver1, ver2))

    print("\n=== [D] 边写边读：不得出现撕裂 ===", flush=True)
    # 高频发布 200 个不同版本，子进程持续读并校验
    q = mp.Queue()
    stop = mp.Event()
    p = mp.Process(target=reader, args=(snap, q, stop), daemon=True)
    p.start()
    for i in range(200):
        snap.publish(make_sd(1000 + i))
        if i % 20 == 0:
            time.sleep(0.002)
    stop.set()
    p.join(timeout=10)
    if p.is_alive():
        p.terminate()
    results = []
    while not q.empty():
        results.append(q.get())
    tears = [r for r in results if r[0] == "tear"]
    monos = [r for r in results if r[0] == "mono"]
    seen = [r for r in results if r[0] == "done"]
    chk("子进程确实在读（非空转）", bool(seen) and seen[0][1] > 0,
        "读到 %s 个版本" % (seen[0][1] if seen else 0))
    chk("无撕裂（md5 校验全过）", not tears, "撕裂 %d 次" % len(tears))
    chk("版本单调", not monos, "回退 %d 次" % len(monos))
    if tears:
        print("     首个撕裂: %s" % (tears[0],), flush=True)

    print("\n=== 结论 ===", flush=True)
    if res[0]:
        print(" 快照读写一致、hash 可独立复算、并发下无撕裂。", flush=True)
        return 0
    print(" 存在 FAIL —— 权重快照不可信，不能用它做 rho/复现。", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
