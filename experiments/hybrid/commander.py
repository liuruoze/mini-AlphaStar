#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指挥官：把"当前局势"映射成一条 MacroCommand。

关键设计：**不让 LLM 自由生成**，只让它从固定策略表里选一个。
理由（今天实测）：Qwen2.5-1.5B 对 SC2 的知识很弱，自由生成会产出"神族造攻城机枪塔"这种非法指令；
约束成选择题后，即使模型很笨，输出空间也是合法的。

用法:
  python experiments/hybrid/commander.py --trace <jsonl> [--backend local]
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from experiments.hybrid.obs_to_text import summarize_record
    from experiments.hybrid import llm_client
except Exception:  # 直接以脚本运行时
    sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))
    from obs_to_text import summarize_record
    import llm_client


@dataclass
class MacroCommand:
    kind: str          # build | train | expand | attack | defend | tech
    target: str        # "Pylon" / "Zealot" / "natural" / "enemy_main"
    count: int = 1
    deadline_s: float = 30.0
    priority: int = 1

    def valid(self):
        return (self.kind in ("build", "train", "expand", "attack", "defend", "tech")
                and isinstance(self.target, str) and self.target
                and 1 <= self.count <= 20
                and 5.0 <= self.deadline_s <= 120.0
                and 1 <= self.priority <= 5)


# 固定策略表：id -> (中文说明, MacroCommand 模板)
STRATEGIES = {
    "S1": ("补农民到饱和，然后补水晶", MacroCommand("train", "Probe", count=4, priority=2)),
    "S2": ("补兵营，压人口上限", MacroCommand("build", "Pylon", count=1, priority=2)),
    "S3": ("量产狂热者", MacroCommand("train", "Zealot", count=4, priority=3)),
    "S4": ("开二矿", MacroCommand("expand", "natural", count=1, deadline_s=60.0, priority=2)),
    "S5": ("攀科技：控制核心", MacroCommand("build", "CyberneticsCore", count=1, deadline_s=60.0, priority=2)),
    "S6": ("全军进攻敌方主基地", MacroCommand("attack", "enemy_main", count=1, deadline_s=45.0, priority=5)),
    "S7": ("回防，集结在二矿", MacroCommand("defend", "natural", count=1, deadline_s=20.0, priority=5)),
    "S8": ("补气矿", MacroCommand("build", "Assimilator", count=2, priority=2)),
}
DEFAULT_KEY = "S3"

SYSTEM = (
    "你是星际争霸2神族的战术指挥官。你只能从给定编号里选一个，"
    "只输出编号本身（例如 S3），不要解释，不要输出别的字。"
)


def build_prompt(state_text):
    opts = "\n".join("%s: %s" % (k, v[0]) for k, v in STRATEGIES.items())
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "当前局势：\n%s\n\n可选策略：\n%s\n\n输出编号：" % (state_text, opts)},
    ]


def parse_choice(raw):
    """从模型输出里抽 S1..S8；抽不到返回 None。"""
    m = re.search(r"\bS([1-8])\b", raw.upper())
    if m:
        return "S" + m.group(1)
    return None


class Commander:
    def __init__(self, llm=None, max_new_tokens=8, temperature=0.0):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.stats = {"calls": 0, "invalid": 0}

    def propose(self, state_text):
        """返回 (MacroCommand, key, raw)。无 LLM 或解析失败时用默认策略。"""
        if self.llm is None:
            return STRATEGIES[DEFAULT_KEY][1], DEFAULT_KEY, ""
        self.stats["calls"] += 1
        raw = self.llm.chat(build_prompt(state_text),
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature)
        key = parse_choice(raw)
        if key is None:
            self.stats["invalid"] += 1
            return STRATEGIES[DEFAULT_KEY][1], DEFAULT_KEY, raw
        return STRATEGIES[key][1], key, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.trace, errors="ignore") if l.strip()]
    text = summarize_record(rows[-1])
    print("== 状态摘要 ==\n%s\n" % text)

    llm = llm_client.build(args.backend, model_path=args.model_path)
    cmd, key, raw = Commander(llm).propose(text)
    print("== LLM 原始输出 == %r" % raw)
    print("== 选定策略 == %s %s" % (key, STRATEGIES[key][0]))
    print("== MacroCommand == %s" % json.dumps(asdict(cmd), ensure_ascii=False))
    print("== 合法? == %s" % cmd.valid())


if __name__ == "__main__":
    main()
