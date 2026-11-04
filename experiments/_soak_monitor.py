#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长跑采样器（第 50 轮，DeepSeek 署名）。

独立于训练进程运行，**只用标准库**（不 import torch/numpy），
所以它不会给被测进程带来任何 GIL/线程/显存干扰。

采什么、为什么：

| 列 | 为什么需要 |
|---|---|
| `*_mem_mb` | §11.3 的「2 份 vs N+1 份」是**审查结论、没有实测**。模型只有 2.66 M 参数（≈10 MB fp32），
如果实测发现显存差只有几十 MB，那说明「省显存」这条论据基本是**伪的**，
真正的成本在每进程的 CUDA context（几百 MB）。这条必须用数字说话。 |
| `*_util` | forward 是 CPU-bound（第 3 轮实测 GPU 净耗时 0.0 ms）。若长跑中 GPU util 一直很低，
说明 4 卡的算力完全没用上，扩容方向应该是 CPU 侧。 |
| `py_rss_mb` / `py_threads` | 显存泄漏之外最可能的长跑问题是 **CPU 内存/线程泄漏**（每轮 new tensor、线程只增不减）。 |
| `sc2_n` | SC2 实例泄漏是这套代码的历史问题（`pkill` 之前经常残留 20+ 个）。 |
| `load1` | 128 核上跑 30~50 个进程，load 若持续 > 核数说明超订。 |

用法：
    python _soak_monitor.py --out monitor.csv --interval 10 --duration 2400
"""

import argparse
import csv
import os
import subprocess
import time


GPU_QUERY = ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]


def sample_gpu():
    """返回 {物理序号: (mem_used_MiB, util_%)}；失败返回 {}。"""
    try:
        out = subprocess.check_output(GPU_QUERY, stderr=subprocess.DEVNULL,
                                      timeout=10).decode("utf-8", "ignore")
    except Exception:
        return {}
    res = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            res[int(parts[0])] = (int(float(parts[1])), int(float(parts[2])))
        except ValueError:
            continue
    return res


def _read_proc(pid):
    """返回 (name, rss_kb, threads, cmdline)；进程消失则返回 None。"""
    base = "/proc/" + str(pid)
    try:
        with open(base + "/status", "r") as f:
            name, rss_kb, thr = "", 0, 0
            for line in f:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("Threads:"):
                    thr = int(line.split()[1])
        with open(base + "/cmdline", "rb") as f:
            cmd = f.read().decode("utf-8", "ignore")
    except (IOError, OSError, ValueError, IndexError):
        return None
    return name, rss_kb, thr, cmd


def _children(pid):
    """`/proc/<pid>/task/<pid>/children` 是内核直接给的子进程列表（Linux >= 3.5）。"""
    try:
        with open("/proc/%d/task/%d/children" % (pid, pid), "r") as f:
            return [int(x) for x in f.read().split()]
    except (IOError, OSError, ValueError):
        return []


def _tree(roots):
    seen, stack = set(), list(roots)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(_children(p))
    return seen


def sample_procs():
    """只统计**本项目的进程树**（Kimi-12）。

    ⚠️ 第 71 轮之前的实现是"扫全机器所有 python 进程"，
    于是 `py_rss` / `py_threads` 混进了同机别人的进程 ——
    这也正是 §56.2 需要额外用 `ps -o nlwp` 抽查来"补救"的原因。
    现在改成：先找 `run_rl.py` 的根进程，再沿 `/proc/<pid>/task/<pid>/children`
    递归收集子进程，只对这些进程求和。同时输出 `py_other_n`（树外的 python 进程数），
    用来量化"以前到底混进了多少"。

    返回 (py_n, py_rss_mb, py_thr, sc2_n, sc2_rss_mb, run_rl_n, py_other_n)
    """
    roots = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        r = _read_proc(pid)
        if r is None:
            continue
        if "run_rl.py" in r[3]:
            roots.append(int(pid))

    tree = _tree(roots)

    py_n = py_rss = py_thr = 0
    sc2_n = sc2_rss = 0
    run_rl_n = len(roots)
    py_other_n = 0

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        ipid = int(pid)
        r = _read_proc(ipid)
        if r is None:
            continue
        name, rss_kb, thr, cmd = r

        # ⚠️ SC2 必须**全局**数，不能只看进程树：
        #    pysc2 启动 SC2 时是双重 fork（中间进程退出），SC2 被挂到 init 下，
        #    `/proc/<pid>/task/<pid>/children` 里根本看不到它们。
        #    第一版按树过滤，`sc2_n` 直接变成 0 —— 与"SC2 泄漏"完全无法区分。
        #    `SC2_x64` 在本机是唯一可执行文件名，全局匹配是安全的。
        if "SC2_x64" in cmd:
            sc2_n += 1
            sc2_rss += rss_kb
        elif ipid in tree:
            if name.startswith("python"):
                py_n += 1
                py_rss += rss_kb
                py_thr += thr
        elif name.startswith("python"):
            # 树外的 python 进程：以前会被算进 py_rss/py_threads
            py_other_n += 1

    return py_n, py_rss // 1024, py_thr, sc2_n, sc2_rss // 1024, run_rl_n, py_other_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=2400.0)
    args = ap.parse_args()

    ngpu = 4
    header = ["ts", "elapsed"]
    for i in range(ngpu):
        header += ["gpu%d_mem_mb" % i, "gpu%d_util" % i]
    header += ["py_n", "py_rss_mb", "py_threads", "sc2_n", "sc2_rss_mb",
               "run_rl_n", "py_other_n", "load1"]

    t0 = time.time()
    # 追加写：允许「先起监控、后起训练」，也允许断点续采
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(header)
        f.flush()

    while time.time() - t0 < args.duration:
        g = sample_gpu()
        row = [time.strftime("%H:%M:%S"), "%.0f" % (time.time() - t0)]
        for i in range(ngpu):
            mem, util = g.get(i, (-1, -1))
            row += [mem, util]
        py_n, py_rss, py_thr, sc2_n, sc2_rss, run_rl_n, py_other_n = sample_procs()
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            load1 = -1
        row += [py_n, py_rss, py_thr, sc2_n, sc2_rss, run_rl_n, py_other_n,
                "%.2f" % load1]
        w.writerow(row)
        f.flush()
        time.sleep(args.interval)

    f.close()


if __name__ == "__main__":
    main()
