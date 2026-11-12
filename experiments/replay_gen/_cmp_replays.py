#!/usr/bin/env python3
"""对比两个 replay 的对局内容差异（时长 / 胜负 / 动作序列）。

用法: python experiments/replay_gen/_cmp_replays.py a.SC2Replay b.SC2Replay
"""
import collections
import os
import sys

import numpy as np
from absl import flags
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

flags.FLAGS(["x"])
MAP = os.path.expanduser("~/StarCraftII/Maps/Melee/Simple64.SC2Map")


def scan(rc, md, p):
    with open(p, "rb") as f:
        data = f.read()
    acts = []
    loops = 0
    with rc.start() as c:
        c.start_replay(sc_pb.RequestStartReplay(
            replay_data=data, map_data=md, observed_player_id=1,
            options=sc_pb.InterfaceOptions(raw=True), disable_fog=False, realtime=False))
        while True:
            try:
                o = c.observe()
            except Exception:
                break
            gl = o.observation.game_loop
            if gl >= 2 ** 32 - 1:
                break
            loops = gl
            for a in o.actions:
                if a.HasField("action_raw") and a.action_raw.HasField("unit_command"):
                    acts.append(int(a.action_raw.unit_command.ability_id))
            try:
                c.step(1)
            except Exception:
                break
        try:
            c.leave()
        except Exception:
            pass
    return loops, acts


def main():
    md = open(MAP, "rb").read()
    rc = run_configs.get()
    out = []
    for p in sys.argv[1:3]:
        loops, acts = scan(rc, md, p)
        out.append((p, loops, acts))
        print("%-40s loops=%6d (%.1f min)  actions=%d" %
              (os.path.basename(p)[:40], loops, loops / 22.4 / 60.0, len(acts)))

    a, b = out[0][2], out[1][2]
    n = min(len(a), len(b))
    same_prefix = 0
    for i in range(n):
        if a[i] == b[i]:
            same_prefix += 1
        else:
            break
    print()
    print("第一个不同的动作位置 = %d / %d" % (same_prefix, n))
    print("动作序列逐位相同比例 = %.1f%%" % (100.0 * sum(1 for i in range(n) if a[i] == b[i]) / max(n, 1)))

    ca = collections.Counter(a)
    cb = collections.Counter(b)
    keys = set(ca) | set(cb)
    va = np.array([ca.get(k, 0) for k in keys], dtype=float)
    vb = np.array([cb.get(k, 0) for k in keys], dtype=float)
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    print("ability 直方图余弦 = %.3f" % cos)
    print("top abilities A:", ca.most_common(5))
    print("top abilities B:", cb.most_common(5))


if __name__ == "__main__":
    main()
