#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指令仲裁器：把 MacroCommand 变成"允许的动作集合"（action mask），并处理冲突/超时。

规则（写死，避免与网络打架）：
  1. 同一时刻最多 1 条 build + 1 条 train 生效，其余排队（队列上限 4）；
  2. 指令超时未完成 → 作废，计 command_dropped，动作集交还网络（全集）；
  3. attack/defend 只放宽到"移动/攻击"类，不限制具体目标；
  4. mask 只做减法：只屏蔽与当前指令无关的动作大类，不在允许集内再排序。

用法（库）：
    ar = Arbiter()
    ar.submit(cmd, t=12.0)
    allowed = ar.mask()          # set[int] or None（None = 不限制）
    ar.tick(t=45.0)              # 到期作废
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from commander import MacroCommand  # noqa: E402

MAX_QUEUE = 4


def _action_sets():
    """按 pysc2 RAW_FUNCTIONS 的名字切分动作集合。拿不到 pysc2 时返回空表（mask=None）。"""
    try:
        from pysc2.lib.actions import RAW_FUNCTIONS
    except Exception:
        return {}
    sets = {"build": set(), "train": set(), "move": set(), "attack": set(),
            "tech": set(), "expand": set(), "base": set()}
    for i, fn in enumerate(RAW_FUNCTIONS):
        n = fn.name
        if n.startswith("Build_"):
            sets["build"].add(i)
            if n.startswith("Build_Nexus"):
                sets["expand"].add(i)
        if n.startswith("Train_"):
            sets["train"].add(i)
        # 基础动作永远放行：否则农民无法交矿、回合无法空转，会把经济打死
        if n in ("no_op", "Harvest_Gather_screen", "Harvest_Gather_unit",
                 "Rally_Units_minimap", "Rally_Units_screen", "HoldPosition_pt", "Stop_pt"):
            sets["base"].add(i)
        if n.startswith(("Move_", "Attack_", "Smart")):
            sets["move"].add(i)
            sets["attack"].add(i)
    return sets


ACTIONS = _action_sets()


class Arbiter:
    def __init__(self, max_queue=MAX_QUEUE):
        self.active = {}          # kind -> (cmd, submit_t, deadline_t)
        self.queue = []
        self.max_queue = max_queue
        self.stats = {"submitted": 0, "accepted": 0, "queued": 0, "rejected": 0, "dropped": 0}

    # ---- 提交 ----
    def submit(self, cmd, t):
        assert cmd.valid(), "非法 MacroCommand"
        self.stats["submitted"] += 1
        if cmd.kind in self.active:
            if len(self.queue) >= self.max_queue:
                self.stats["rejected"] += 1
                return "rejected"
            self.queue.append((cmd, t))
            self.stats["queued"] += 1
            return "queued"
        self.active[cmd.kind] = (cmd, t, t + cmd.deadline_s)
        self.stats["accepted"] += 1
        return "accepted"

    # ---- 时间推进 ----
    def tick(self, t):
        """作废超时指令；腾出的槽位从队列里补。"""
        for kind in list(self.active):
            cmd, _, deadline = self.active[kind]
            if t >= deadline:
                del self.active[kind]
                self.stats["dropped"] += 1
        while self.queue:
            cmd, sub_t = self.queue[0]
            if cmd.kind in self.active:
                break
            self.queue.pop(0)
            self.active[cmd.kind] = (cmd, sub_t, t + cmd.deadline_s)
            self.stats["accepted"] += 1

    # ---- 动作遮罩 ----
    def mask(self):
        """返回允许的动作 id 集合；None 表示不限制（交还网络）。"""
        if not self.active or not ACTIONS:
            return None
        kinds = set(self.active)
        allow = set()
        # 任何时刻都保留"移动/攻击"与 no_op 之外的基础操作，避免把局面锁死
        allow |= ACTIONS.get("base", set())
        allow |= ACTIONS.get("move", set())
        if "build" in kinds or "tech" in kinds or "expand" in kinds:
            allow |= ACTIONS.get("build", set())
        if "train" in kinds:
            allow |= ACTIONS.get("train", set())
        if "attack" in kinds:
            allow |= ACTIONS.get("attack", set())
        if "defend" in kinds:
            allow |= ACTIONS.get("base", set())
        allow |= ACTIONS.get("move", set())
        return allow or None

    def snapshot(self):
        return {"active": {k: v[0].kind + ":" + v[0].target for k, v in self.active.items()},
                "queued": len(self.queue), "stats": dict(self.stats)}


if __name__ == "__main__":
    ar = Arbiter()
    print("动作集大小：", {k: len(v) for k, v in ACTIONS.items()})
