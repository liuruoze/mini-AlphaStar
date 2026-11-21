#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""教条-规则指挥官：完全按 doctrine.decide_bits 算五条规则，不经 LLM。

存在的目的只有一个：把"教条本身好不好"和"LLM 执行得准不准"分开。

  · 本模块（doctrine 臂）：教条 → 命令，机器直接算，永不判错；
  · commander_v3（llm_v3 臂）：同样的教条 → 交给 32B 逐条判断，可能判错。

三臂对照：
    sl       纯策略（无指挥官）
    doctrine 教条-规则（本模块）
    llm_v3   教条-LLM

若 doctrine ≈ llm_v3，说明 LLM 没拖后腿；若 doctrine 明显更好，差的就是 LLM 的判断误差；
若两者都不如 sl，说明问题在教条本身，而不是执行。

接口与其它指挥官一致：
    propose(state_or_obs) -> (MacroCommand | None, bits, reason)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from experiments.hybrid import doctrine
    from experiments.hybrid.commander_v3 import state_from_obs, pick_command
except Exception:  # 直接以脚本运行时
    sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))
    import doctrine
    from commander_v3 import state_from_obs, pick_command


class DoctrineCommander:
    """零不确定性的下界：教条本身能拿到什么成绩。"""

    def __init__(self):
        self.stats = {"decisions": 0}
        self.last_bits = ""
        self.last_cmd = None

    def propose(self, state_or_obs):
        if isinstance(state_or_obs, dict) and "cap" in state_or_obs:
            st = state_or_obs
        else:
            st = state_from_obs(state_or_obs)

        bits = doctrine.decide_bits(
            t=st["t_s"], used=st["used"], cap=st["cap"],
            workers=st["workers"], army=st["army"], gw=st["gw"],
            minerals=st["minerals"], pylons=st["pylons"])

        self.stats["decisions"] += 1
        self.last_bits = bits
        cmd, name = pick_command(bits)
        self.last_cmd = cmd
        return cmd, bits, "doc:%s|w%d a%d g%d py%d" % (
            name, st["workers"], st["army"], st["gw"], st["pylons"])


def _selftest():
    """不依赖游戏：拿几个典型局面验证五位与命令。"""
    cases = [
        # (state, 期望命中) —— 期望值按 doctrine.decide_bits + PRIORITY 手算
        (dict(t_s=0, used=12, cap=15, minerals=50, workers=12, army=0, gw=0, pylons=0), "pylon"),
        (dict(t_s=300, used=40, cap=70, minerals=300, workers=14, army=20, gw=3, pylons=4), "probe"),
        (dict(t_s=300, used=40, cap=70, minerals=300, workers=25, army=9, gw=3, pylons=4), "attack"),
        (dict(t_s=300, used=40, cap=70, minerals=300, workers=25, army=2, gw=1, pylons=4), "gateway"),
        (dict(t_s=300, used=40, cap=70, minerals=50, workers=25, army=2, gw=3, pylons=4), "none"),
        (dict(t_s=300, used=40, cap=47, minerals=300, workers=25, army=0, gw=3, pylons=4), "pylon"),
    ]
    bad = []
    dc = DoctrineCommander()
    for st, want in cases:
        cmd, bits, why = dc.propose(st)
        got = why.split("|")[0].split(":")[1]
        if got != want:
            bad.append((st, want, got, bits))
    if bad:
        raise AssertionError("教条臂自测失败：%s" % bad)
    return len(cases)


if __name__ == "__main__":
    n = _selftest()
    print("doctrine 臂自测通过：%d 例" % n)
    dc = DoctrineCommander()
    for st in (dict(t_s=0, used=12, cap=15, minerals=50, workers=12, army=0, gw=0, pylons=0),
               dict(t_s=420, used=50, cap=70, minerals=300, workers=25, army=6, gw=3, pylons=2)):
        cmd, bits, why = dc.propose(st)
        print(bits, why, "->", (cmd.kind + ":" + cmd.target) if cmd else "-")
