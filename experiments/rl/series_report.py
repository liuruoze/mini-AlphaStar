#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总连续 soak 的结果。

不依赖 supervisor 写的 summary.csv（那版有列错位的 bug），
直接读每轮的 `log/soak_<tag>/summary.txt` 与 `heartbeat.txt` 重算。

用法：python experiments/rl/series_report.py [--root /tmp/series] [--log ~/mini-AlphaStar/log]
"""

import glob
import os
import re
import sys


def parse(summary_path, hb_path, run_log):
    wall = rc = None
    if os.path.exists(summary_path):
        txt = open(summary_path, errors="ignore").read()
        m = re.search(r"rc=(-?\d+)", txt)
        w = re.search(r"wall=(\d+)", txt)
        rc = int(m.group(1)) if m else None
        wall = int(w.group(1)) if w else None

    vsteps = stalls = None
    if os.path.exists(hb_path):
        lines = [l for l in open(hb_path, errors="ignore") if "v_steps" in l]
        if lines:
            mm = re.search(r"v_steps=(\d+)", lines[-1])
            vsteps = int(mm.group(1)) if mm else None

    tput = None
    if os.path.exists(run_log):
        txt = open(run_log, errors="ignore").read()
        stalls = txt.count("[stall]")
        # 日志里每 worker 会打印自己的吞吐；取中位数，别用 v_steps 除（那是多 actor 累计）
        vals = [float(x) for x in re.findall(r"吞吐 ([0-9.]+) step/s", txt)]
        if vals:
            vals.sort()
            tput = vals[len(vals) // 2]
    return wall, rc, vsteps, stalls, tput


def main():
    root = "/tmp/series"
    logroot = os.path.expanduser("~/mini-AlphaStar/log")
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--root" and i + 1 < len(args):
            root = args[i + 1]
        if a == "--log" and i + 1 < len(args):
            logroot = os.path.expanduser(args[i + 1])

    tags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "r*"))
                  if os.path.isdir(p))
    if not tags:
        print("没有找到轮次目录：%s" % root)
        return 1

    # 从 series.log 建 tag -> (par, act) 的映射，比从目录名猜可靠
    cfg = {}
    logp = os.path.join(root, "series.log")
    if os.path.exists(logp):
        for line in open(logp, errors="ignore"):
            m = re.search(r"开始 (r\d+) \(par=(\d+) act=(\d+)\)", line)
            if m:
                cfg[m.group(1)] = (m.group(2), m.group(3))

    print("%-4s %-4s %-5s %8s %5s %7s %10s %s" % (
        "轮", "并发", "actor", "墙钟(s)", "rc", "卡死", "虚拟步", "吞吐(step/s)"))
    total_s = total_stall = 0
    done = 0
    for tag in tags:
        d = os.path.join(logroot, "soak_" + tag)
        wall, rc, vsteps, stalls, tput_v = parse(
            os.path.join(d, "summary.txt"),
            os.path.join(d, "heartbeat.txt"),
            os.path.join(d, "run.log"))
        par, act = cfg.get(tag, ("?", "?"))

        if wall:
            total_s += wall
            done += 1
        total_stall += (stalls or 0)
        tput = "%.2f" % tput_v if tput_v else "-"
        print("%-4s %-4s %-5s %8s %5s %7s %10s %s" % (
            tag, par, act, wall if wall else "—", rc if rc is not None else "—",
            stalls if stalls is not None else "—", vsteps if vsteps else "—", tput))

    print()
    print("完成 %d 轮，累计 %.1f 分钟，卡死事件 %d 次" % (done, total_s / 60.0, total_stall))
    if done and total_stall == 0:
        # 零事件时，真实故障率的上限（95% 置信，rule of three）
        print("零故障时故障率的 95%% 置信上限：%.4f 次/分钟（rule of three: 3 / %.0f 分钟）"
              % (3.0 / (total_s / 60.0), total_s / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
