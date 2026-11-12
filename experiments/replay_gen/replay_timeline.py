#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 replay 里取「首个 Pylon / Gateway / Zealot 的时间线」以及 Zealot 存活时长。

sc2reader 1.8 对本地自采 replay 会在 details["cache_handles"] 为空时崩
（resources.py: self.region = details["cache_handles"][0].server.lower()），
这里 patch 一个假 handle 绕过，不影响 events 解析。
"""
import glob
import os
import sys

import sc2reader
from sc2reader.resources import Replay

_orig_load_details = Replay.load_details


def _safe_load_details(self):
    """本地自采 replay 的 details["cache_handles"] 常为空（非战网下载），
    sc2reader 会在这里抛 IndexError。details 只是元信息，跳过不影响 events。"""
    try:
        _orig_load_details(self)
    except Exception as e:
        print("[patch] load_details skipped: %s" % e)


Replay.load_details = _safe_load_details

GAME_LOOPS_PER_SEC = 22.4


def main(pat):
    paths = sorted(glob.glob(pat))
    if not paths:
        print("no replay for", pat)
        return
    p = paths[-1]
    print("replay:", os.path.basename(p))
    r = sc2reader.load_replay(p, load_level=4)
    print("duration: %.1f s (%d loops)" % (r.game_length.seconds, r.frames))
    init, done, born, died = {}, {}, {}, {}
    for e in r.events:
        u = getattr(e, "unit", None)
        if u is None or u.owner is None or u.owner.pid != 1:
            continue
        n = u.name
        if n not in ("Pylon", "Gateway", "Zealot", "Probe", "Nexus"):
            continue
        if e.name == "UnitInitEvent":
            init.setdefault(n, []).append(e.frame)
        elif e.name == "UnitDoneEvent":
            done.setdefault(n, []).append(e.frame)
        elif e.name == "UnitBornEvent":
            born.setdefault(n, []).append(e.frame)
        elif e.name == "UnitDiedEvent":
            died.setdefault(n, []).append(e.frame)

    def show(tag, d, limit=3):
        arr = sorted(d.get(tag, []))
        if not arr:
            print("  %-8s none" % tag)
            return
        head = ", ".join("%.1fs" % (f / GAME_LOOPS_PER_SEC) for f in arr[:limit])
        print("  %-8s first=%s  total=%d" % (tag, head, len(arr)))

    print("== 开始建造 (UnitInit) ==")
    show("Pylon", init)
    show("Gateway", init)
    print("== 建成 (UnitDone) ==")
    show("Pylon", done)
    show("Gateway", done)
    print("== 出生 (UnitBorn) ==")
    show("Probe", born, 5)
    show("Zealot", born, 8)
    print("== 死亡 (UnitDied) ==")
    show("Probe", died, 10)
    show("Zealot", died, 10)
    show("Pylon", died, 10)
    show("Gateway", died, 10)

    zb, zd = sorted(born.get("Zealot", [])), sorted(died.get("Zealot", []))
    if zb:
        print("== Zealot 逐个 ==")
        for i, f in enumerate(zb[:10]):
            d = zd[i] if i < len(zd) else None
            print("  #%d born=%.1fs died=%s life=%s"
                  % (i + 1, f / GAME_LOOPS_PER_SEC,
                     ("%.1fs" % (d / GAME_LOOPS_PER_SEC)) if d else "-",
                     ("%.1fs" % ((d - f) / GAME_LOOPS_PER_SEC)) if d else "alive"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/home/liuruoze/mini-AlphaStar-off/revised_simple64_replays/r5m2*_*.SC2Replay")
