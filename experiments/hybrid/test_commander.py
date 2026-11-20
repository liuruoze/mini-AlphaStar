#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指挥官的 4 条单测：不依赖模型与 SC2，秒级。

T1 合法输出（"S5"）→ 选中 S5
T2 输出里带噪声（"我选 S7。"）→ 仍能解析出 S7
T3 非法输出（"建造机枪塔"）→ 回退到默认 + invalid 计数 +1
T4 所有策略模板生成的 MacroCommand 都通过 valid()

用法：python experiments/hybrid/test_commander.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from commander import Commander, STRATEGIES, DEFAULT_KEY, parse_choice, MacroCommand


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, max_new_tokens=None, temperature=None):
        return self.reply


def main():
    results = []

    c = Commander(FakeLLM("S5"))
    cmd, key, _ = c.propose("人口 30/31")
    results.append(("T1 直接输出 S5", key == "S5" and cmd.kind == "build"))

    c = Commander(FakeLLM("我选 S7。"))
    cmd, key, _ = c.propose("人口 30/31")
    results.append(("T2 带噪声仍解析", key == "S7" and cmd.kind == "defend"))

    c = Commander(FakeLLM("建造一个攻城机枪塔"))
    cmd, key, raw = c.propose("人口 30/31")
    results.append(("T3 非法回退", key == DEFAULT_KEY and c.stats["invalid"] == 1 and raw != ""))

    ok = all(v[1].valid() for v in STRATEGIES.values())
    results.append(("T4 模板全合法", ok and MacroCommand("attack", "enemy_main", deadline_s=45.0, priority=5).valid()))

    bad = 0
    print("%-20s %-5s" % ("check", "pass"))
    for name, passed in results:
        print("%-20s %-5s" % (name, "OK" if passed else "FAIL"))
        bad += 0 if passed else 1
    print("\n%d/%d passed" % (len(results) - bad, len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
