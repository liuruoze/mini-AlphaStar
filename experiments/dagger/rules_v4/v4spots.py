#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线枚举 Simple64 基地的固定建筑点（Pylon / Gateway）。

思路（参考 python-sc2 的做法：相对 Nexus 的固定 world 偏移 + placement_grid 校验）：
1. 以我方 Nexus（minimap (19,23)，world (29.25,60.75)）为原点，枚举 ±12 minimap 格；
2. Pylon 需 2x2 全可建（half=1.0 world）；
3. 对每个 Pylon 候选，统计它能量场（6.5 world 半径）内还能放几个 Gateway（3x3，half=1.5）；
4. 输出「Pylon 点 + 可放 Gateway 数」，按两者综合排序，人工取前几个当固定点表。

说明：minimap 坐标 ↔ world 换算（实测）
  world_x = mmx*1.5 + 0.75
  world_y = 96 - (mmy*1.5 + 0.75)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PG = np.load(os.path.join(HERE, "simple64_placement.npy"))   # (96, 88) 1=可建
MAP_H = 96.0
ANCHOR = (19, 23)          # 左上出生点（seed 3038）
POWER_R = 6.5              # SC2 Pylon 能量半径（world）


def to_world(mmx, mmy):
    return (mmx * 1.5 + 0.75, MAP_H - (mmy * 1.5 + 0.75))


def buildable(mmx, mmy, r_world):
    """以 minimap 点为中心、半径 r_world 的方块是否全部可建。"""
    h, w = PG.shape
    wx, wy = to_world(mmx, mmy)
    x0, x1 = int(round(wx - r_world)), int(round(wx + r_world))
    y0, y1 = int(round(wy - r_world)), int(round(wy + r_world))
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return False
    return bool(PG[y0:y1 + 1, x0:x1 + 1].min() > 0)


def main():
    ax, ay = ANCHOR
    pylons = []
    for dy in range(-12, 13):
        for dx in range(-12, 13):
            d2 = dx * dx + dy * dy
            if not (9 <= d2 <= 100):       # 距 Nexus 3~10 格
                continue
            mx, my = ax + dx, ay + dy
            if not (1 <= mx <= 62 and 1 <= my <= 62):
                continue
            if not buildable(mx, my, 1.0):  # Pylon 2x2
                continue
            # 该 Pylon 的能量场内能放几个 Gateway（3x3）
            gws = []
            for gdy in range(-4, 5):
                for gdx in range(-4, 5):
                    if not (4 <= gdx * gdx + gdy * gdy <= 11):   # 圆心距 2~3.3
                        continue
                    gx, gy = mx + gdx, my + gdy
                    if abs(gx - ax) < 3 and abs(gy - ay) < 3:     # 不压 Nexus(2.7)
                        continue
                    if buildable(gx, gy, 1.5):
                        gws.append((gx, gy))
            if gws:
                pylons.append((dx, dy, mx, my, len(gws), gws[:3]))

    pylons.sort(key=lambda t: (-t[4], t[0] * t[0] + t[1] * t[1]))
    print("== Pylon 候选（相对 anchor 偏移；gw=能量场内可放 Gateway 点数）==")
    print("dx dy  abs        gw  sample_gw")
    for dx, dy, mx, my, n, sample in pylons[:15]:
        print("%+3d %+3d (%2d,%2d)  %2d  %s" % (dx, dy, mx, my, n, sample))
    print("\n合计可用 Pylon 点: %d" % len(pylons))


if __name__ == "__main__":
    main()
