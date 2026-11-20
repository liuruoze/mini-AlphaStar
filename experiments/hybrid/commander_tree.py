#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""决策树版指挥官：把"8 选 1"换成 3 个二元问题。

依据（2026-11-20 实测）：同一个 Qwen3.5-9B，
  · 8 选 1：无论局面（甚至 198/200 人口、军队 90）都答 S1，完全无视状态；
  · 二元提问："军队是否少于 10？" 三个极端局面全答对（否/是/否）。
所以不是模型不会读局面，而是"8 选 1"这个形式让它退化。这里把决策拆成逐层二元判断。

三个问题（每层只问一次，答案来自状态里的数字）：
  Q1 人口是否已到上限的 90%？   → 是：补水晶（S2）
  Q2 军队是否少于 10？          → 是：量产狂热者（S3）
  Q3 空闲农民是否 ≥8？          → 是：补农民（S1）
  否则：进攻（S6）

与 Commander 同接口：propose(text) -> (MacroCommand|None, key, raw)
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from commander import STRATEGIES  # noqa: E402
from obs_to_text import summarize_obs  # noqa: E402

SYS = "你是数值比较器。只答一个字：是 或 否。"

# few-shot：实测 7B 在没有示例时会把"军队 90 是否少于 10"答成"是"，
# 给 3 个示例后 5/5 全对（含边界 10 与 9）。
FEWSHOT = {
    "supply": ("示例：人口 198/200，问\"是否已达上限的 90%?\"→是。\n"
               "示例：人口 60/63，问\"是否已达上限的 90%?\"→是。\n"
               "示例：人口 20/47，问\"是否已达上限的 90%?\"→否。\n"),
    "army": ("示例：军队=90，问\"少于10?\"→否。\n"
             "示例：军队=1，问\"少于10?\"→是。\n"
             "示例：军队=20，问\"少于10?\"→否。\n"),
    "idle": ("示例：空闲农民=2，问\"是否≥8?\"→否。\n"
             "示例：空闲农民=10，问\"是否≥8?\"→是。\n"
             "示例：空闲农民=8，问\"是否≥8?\"→是。\n"),
}


def yes(text):
    t = (text or "").strip()
    if not t:
        return None
    if "是" in t[:6] and "否" not in t[:6]:
        return True
    if "否" in t[:6]:
        return False
    if re.search(r"\byes\b", t[:6], re.I):
        return True
    if re.search(r"\bno\b", t[:6], re.I):
        return False
    return None


class TreeCommander:
    def __init__(self, llm, max_new_tokens=900):
        self.llm = llm
        self.max_new_tokens = max_new_tokens     # 思考模型需要足够 token 才能吐出答案
        self.stats = {"calls": 0, "unparsed": 0}

    def _ask(self, state_text, question, kind=""):
        self.stats["calls"] += 1
        fs = FEWSHOT.get(kind, "")
        out = self.llm.chat([{"role": "system", "content": SYS},
                             {"role": "user", "content": fs + state_text + "\n" + question}],
                            max_new_tokens=self.max_new_tokens, temperature=0.0)
        v = yes(out)
        if v is None:
            self.stats["unparsed"] += 1
        return v

    def propose(self, state_text):
        raw = []
        q1 = self._ask(state_text, "人口是否已达到人口上限的 90%？", "supply")
        raw.append("Q1=%s" % q1)
        if q1:
            return STRATEGIES["S2"][1], "S2", "|".join(raw)

        q2 = self._ask(state_text, "军队数量是否少于 10？", "army")
        raw.append("Q2=%s" % q2)
        if q2:
            return STRATEGIES["S3"][1], "S3", "|".join(raw)

        q3 = self._ask(state_text, "空闲农民数量是否大于等于 8？", "idle")
        raw.append("Q3=%s" % q3)
        if q3:
            return STRATEGIES["S1"][1], "S1", "|".join(raw)

        return STRATEGIES["S6"][1], "S6", "|".join(raw)


# ---- 派生量版：把"需要做减法"的量预先算好（实测把准确率从 30% 拉到 75%）----
DT_FEWSHOT = {
    "gap": ("示例：人口余量=3，问\"≤6?\"→是。\n"
            "示例：人口余量=30，问\"≤6?\"→否。\n"),
    "workers": ("示例：农民=14，问\"<18?\"→是。\n"
                "示例：农民=26，问\"<18?\"→否。\n"),
    "army": ("示例：军队=4，问\"<10?\"→是。\n"
             "示例：军队=14，问\"<10?\"→否。\n"),
}


class TreeDCommander(TreeCommander):
    """接收 obs（不是文本）；阈值取自 V5 专家：余量≤6 / 农民<18 / 军队<10。"""

    def propose(self, obs):
        if isinstance(obs, str):
            base = obs
            gap = None
        else:
            base = summarize_obs(obs)
            pl = getattr(obs, "player", None)
            gap = int(getattr(pl, "food_cap", 0) or 0) - int(getattr(pl, "food_used", 0) or 0)
        text = base if gap is None else base + "\n人口余量 %d（上限-已用）" % gap

        raw = []
        q1 = self._ask_d(text, "人口余量是否小于等于 6？", DT_FEWSHOT["gap"])
        raw.append("gap=%s" % q1)
        if q1:
            return STRATEGIES["S2"][1], "S2", "|".join(raw)
        q2 = self._ask_d(text, "农民数量是否少于 18？", DT_FEWSHOT["workers"])
        raw.append("wk=%s" % q2)
        if q2:
            return STRATEGIES["S1"][1], "S1", "|".join(raw)
        q3 = self._ask_d(text, "军队数量是否少于 10？", DT_FEWSHOT["army"])
        raw.append("army=%s" % q3)
        if q3:
            return STRATEGIES["S3"][1], "S3", "|".join(raw)
        return STRATEGIES["S6"][1], "S6", "|".join(raw)

    def _ask_d(self, text, q, shots):
        self.stats["calls"] += 1
        out = self.llm.chat([{"role": "system", "content": "只答一个字：是 或 否。"},
                             {"role": "user", "content": shots + text + "\n" + q}],
                            max_new_tokens=8, temperature=0.0)
        v = yes(out)
        if v is None:
            self.stats["unparsed"] += 1
        return v
