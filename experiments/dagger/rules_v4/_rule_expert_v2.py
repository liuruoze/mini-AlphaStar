#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAgger 规则专家 v2 —— 阶段 1 交付：**动态位置接口**（只做接口 + 自测）。

版本串：rule_v2_20261109
与 v1 的边界（严格区分，不覆盖 v1）：
  - v1 (`_rule_expert_v1.py`, rule_v1_20261108) 只有 decide/summarize，位置由
    `_dagger_build.py` 的**固定模板**提供 ⇒ 去重率 0.086 / 0.018（HY27 #2）。
  - v2 本文件只新增 `LocationPicker`：按当前帧的已占实体坐标动态选合法建造点，
    并用 tick 轮换保证去重率。**不接入 dump 管线、不生成 v3 数据集、不训练**
    （计划阶段 1 范围）。
  - decide 逻辑仍沿用 v1 的 `decide`；停止条件 / 动作多样性属后续阶段，不在此改。

设计依据（实测，HY27 #3）：
  - `feature_screen/minimap["buildable"]` 在本机 SC2 版本下全 0 ⇒ 地形合法率**不可测**；
  - 因此 v2 不依赖该层，改用「以基地锚点为中心、避开当前已占实体邻域」的可测策略；
  - 目标门禁：位置去重率 ≥0.3（v1 0.086 / v2数据 0.018 均未达标）。

用法：python experiments/dagger/rules_v4/_rule_expert_v2.py   # 单元自测
"""
from __future__ import annotations

VERSION = "rule_v2_20261109"
GRID = 64
# Simple64 左上出生角我方 Nexus 开局实测坐标（HY27 #3, seed 3038）
DEFAULT_ANCHOR = (19, 23)
# Protoss 建筑占地：Pylon 2×2、Gateway/Nexus 更大；留出 2 格缓冲
EXCLUDE_RADIUS = 2
R_MIN, R_MAX = 4, 12


class LocationPicker:
    """按当前帧已占实体坐标，动态给一个「尽量合法且每帧不同」的建造点。

    occupied: 可迭代的 (x, y)；来自当前帧 observation 的 raw_units（含建筑/中立矿）。
    tick: 单调递增的帧序号，用来轮换候选点，避免所有规则帧落在同一格。
    """

    def __init__(self, anchor=DEFAULT_ANCHOR, grid=GRID,
                 r_min=R_MIN, r_max=R_MAX, exclude=EXCLUDE_RADIUS):
        self.anchor = tuple(anchor)
        self.grid = grid
        self.r_min, self.r_max, self.exclude = r_min, r_max, exclude
        self._ring_cache = None

    def _ring(self):
        if self._ring_cache is None:
            pts = []
            ax, ay = self.anchor
            for r in range(self.r_min, self.r_max + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r:
                            continue
                        x, y = ax + dx, ay + dy
                        if 0 <= x < self.grid and 0 <= y < self.grid:
                            pts.append((x, y))
            self._ring_cache = pts
        return self._ring_cache

    def candidates(self, occupied):
        """过滤掉与已占实体（含半径 exclude）冲突的点。"""
        occ = {(int(x), int(y)) for x, y in occupied}
        out = []
        for x, y in self._ring():
            if any(abs(x - ox) <= self.exclude and abs(y - oy) <= self.exclude
                   for ox, oy in occ):
                continue
            out.append((x, y))
        return out

    def pick(self, occupied, tick):
        c = self.candidates(occupied)
        if not c:
            return None
        # 轮换：tick 线性 + 质数步长打散，保证相邻帧不落同格
        return c[(int(tick) * 7 + 3) % len(c)]

    @staticmethod
    def to_label_index(x, y, orient="A"):
        """4096 段展平索引。坐标方向尚未从生成代码确认 ⇒ 两种都支持，默认 A。"""
        return y * GRID + x if orient == "A" else x * GRID + y


def self_test():
    p = LocationPicker()
    # 开局实测：Nexus (19,23) + 12 Probe
    units = [(19, 23), (17, 21), (17, 23), (18, 21), (17, 24), (19, 21),
             (20, 21), (17, 25), (19, 21), (17, 23), (17, 21), (17, 22)]
    cand = p.candidates(units)
    print("候选点数 =", len(cand), "（环形 r=4..12，避开已占 ±2 格）")
    picks = [p.pick(units, t) for t in range(20)]
    uniq = len(set(picks))
    print("20 次连续取点 不同坐标 =", uniq, " 去重率 = %.2f" % (uniq / 20))
    print("前 8 个 =", picks[:8])
    assert len(cand) > 0, "候选为空"
    assert uniq / 20 >= 0.3, "去重率未达门禁 0.3"
    for x, y in picks:
        assert 0 <= x < GRID and 0 <= y < GRID
        assert not any(abs(x - ox) <= EXCLUDE_RADIUS and abs(y - oy) <= EXCLUDE_RADIUS
                       for ox, oy in units), "与已占实体冲突"
    assert all(p.to_label_index(*q) < 4096 for q in picks)
    print("SELF-TEST PASS：动态位置接口可用，去重率达标、避开占位、索引在界内")
    print("注意：地形 buildable 不可测，本接口只保证「不撞当前实体」，不保证地形合法")


if __name__ == "__main__":
    self_test()
