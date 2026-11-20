#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规则版指挥官：只依赖观测里的绝对量，输出一条 MacroCommand 或 None。

为什么先做规则版：
  - 逻辑可读、可测、无延迟，适合先验证"遮罩这条管线通不通"；
  - LLM 版接同一个接口（Commander.propose），换实现不换管线。

规则（按优先级，命中即返回）：
  R1 人口 ≥ 90% 上限            → build Pylon（防卡人口）
  R2 存款 ≥ 400 且军队未增长     → train Zealot（把钱变成兵）
  R3 空闲农民 ≥ 8               → train Probe（把人拉回生产）
  R4 8 分钟后军队 < 8           → train Zealot（补兵）
 否则返回 None（交还网络）
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from commander import MacroCommand  # noqa: E402


def stats_from_obs(obs):
    pl = getattr(obs, "player", None)
    return {
        "t_s": float(getattr(obs, "game_loop", [0])[0]) / 22.4,
        "used": int(getattr(pl, "food_used", 0) or 0),
        "cap": int(getattr(pl, "food_cap", 0) or 0),
        "minerals": int(getattr(pl, "minerals", 0) or 0),
        "vespene": int(getattr(pl, "vespene", 0) or 0),
        "idle": int(getattr(pl, "idle_worker_count", 0) or 0),
        "army": int(getattr(pl, "army_count", 0) or 0),
    }


def decide_from_stats(st, army_prev=None):
    if st["cap"] > 0 and st["used"] >= 0.9 * st["cap"]:
        return MacroCommand("build", "Pylon", count=1, priority=4), "R1_supply"
    stalled = army_prev is not None and st["army"] - army_prev < 1
    if st["minerals"] >= 400 and (stalled or st["t_s"] > 480):
        return MacroCommand("train", "Zealot", count=4, priority=4), "R2_bank"
    if st["idle"] >= 8:
        return MacroCommand("train", "Probe", count=2, priority=3), "R3_idle"
    if st["t_s"] > 480 and st["army"] < 8:
        return MacroCommand("train", "Zealot", count=4, priority=4), "R4_army"
    return None, None


class RuleCommander:
    """与 Commander.propose(text) 同签名，方便直接替换。"""

    def __init__(self, stall_window_s=120.0):
        self.last_army = None
        self.last_t = None
        self.stall_window_s = stall_window_s
        self.last_key = None

    def propose(self, state_text_or_obs):
        # 注意：pysc2 的 Observation 也支持 obj["cap"] 这种取键，所以不能只靠 isinstance(dict)
        if isinstance(state_text_or_obs, dict) and "cap" in state_text_or_obs:
            st = state_text_or_obs
        else:
            st = stats_from_obs(state_text_or_obs)
        # 只在"距上次采样 ≥ 窗口长度"时比较军队增量，避免间隔太短误判停滞
        prev_army = None
        if self.last_t is not None and st["t_s"] - self.last_t >= self.stall_window_s:
            prev_army = self.last_army
        if prev_army is None or st["t_s"] - (self.last_t or 0) >= self.stall_window_s:
            self.last_army = st["army"]
            self.last_t = st["t_s"]
        cmd, key = decide_from_stats(st, prev_army)
        self.last_key = key
        return cmd, key, ""


if __name__ == "__main__":
    for st in ({"t_s": 300, "used": 30, "cap": 31, "minerals": 500, "vespene": 0, "idle": 2, "army": 5},
               {"t_s": 500, "used": 60, "cap": 80, "minerals": 100, "vespene": 0, "idle": 10, "army": 20},
               {"t_s": 600, "used": 80, "cap": 90, "minerals": 100, "vespene": 0, "idle": 2, "army": 4}):
        print(st, "->", decide_from_stats(st)[1])
