#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仲裁器 6 条单测（不需要 SC2，秒级）。

T1 同类型第二条指令排队，不并行
T2 build 与 train 可以同时生效
T3 超时作废并计 dropped
T4 队列满时拒绝
T5 build 生效时 mask 含建造类、且不含训练类
T6 attack 生效时 mask 含攻击类

用法：python experiments/hybrid/test_arbiter.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from arbiter import Arbiter, ACTIONS
from commander import MacroCommand


def main():
    res = []

    ar = Arbiter()
    a = MacroCommand("build", "Pylon", deadline_s=30)
    b = MacroCommand("build", "Gateway", deadline_s=30)
    r1, r2 = ar.submit(a, 0), ar.submit(b, 0)
    res.append(("T1 同类排队", r1 == "accepted" and r2 == "queued" and len(ar.active) == 1))

    ar = Arbiter()
    r1 = ar.submit(MacroCommand("build", "Pylon"), 0)
    r2 = ar.submit(MacroCommand("train", "Zealot"), 0)
    res.append(("T2 build+train 并行", r1 == "accepted" and r2 == "accepted" and len(ar.active) == 2))

    ar = Arbiter()
    ar.submit(MacroCommand("build", "Pylon", deadline_s=10), 0)
    ar.tick(15)
    res.append(("T3 超时作废", ar.stats["dropped"] == 1 and not ar.active))

    ar = Arbiter(max_queue=1)
    ar.submit(MacroCommand("build", "Pylon", deadline_s=60), 0)
    ar.submit(MacroCommand("build", "Gateway", deadline_s=60), 0)
    r = ar.submit(MacroCommand("build", "Nexus", deadline_s=60), 0)
    res.append(("T4 队列满拒绝", r == "rejected" and ar.stats["rejected"] == 1))

    if ACTIONS:
        ar = Arbiter()
        ar.submit(MacroCommand("build", "Pylon", deadline_s=60), 0)
        m = ar.mask()
        ok = (m is not None and (m & ACTIONS["build"]) and not (m & (ACTIONS["train"] - ACTIONS["build"])))
        res.append(("T5 build mask", bool(ok)))
        ar = Arbiter()
        ar.submit(MacroCommand("attack", "enemy_main", deadline_s=60), 0)
        m2 = ar.mask()
        res.append(("T6 attack mask", m2 is not None and bool(m2 & ACTIONS["attack"])))
    else:
        res.append(("T5 build mask", False))
        res.append(("T6 attack mask", False))

    bad = 0
    print("%-20s %-5s" % ("check", "pass"))
    for n, ok in res:
        print("%-20s %-5s" % (n, "OK" if ok else "FAIL"))
        bad += 0 if ok else 1
    print("\n%d/%d passed" % (len(res) - bad, len(res)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
