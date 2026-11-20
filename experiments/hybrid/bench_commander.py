#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 指挥官能力基准：先回答"它够不够聪明"，再谈接不接回路。

测四件事：
  A 读状态    —— 给不同数字，问同一个比较问题（如"军队是否<10"）
  B 诊断      —— 给局面，让它从 5 个问题里选最主要的一个
  C 按诊断行动 —— 8 选 1，与"规则给出的答案"比对
  D 致命错    —— 200/200 人口选补农民、无兵选进攻等

另测：重复一致性（同局面问 3 次）、数字引用率（理由里是否引用输入中的数字）、延迟。

判据（写在前面）：准确率≥80% + 致命错=0 + 覆盖≥4/8 + 一致率≥80% 才算"够用"。

用法：
  python experiments/hybrid/bench_commander.py --n 30 --backend openai \
      --base-url http://127.0.0.1:8765/v1 --model-name qwen
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from commander import STRATEGIES, SYSTEM, parse_choice  # noqa: E402
import llm_client                                    # noqa: E402

# ---------- 局面生成 ----------
# 五类局面，每类的正确答案都不同（避免"恒定答案"刷分：
# 若某类占多数，全答那一类的准确率就会虚高，所以这里刻意做成每类 20%）。
REGIMES = {
    "人口卡住": dict(t=300, used=46, cap=47, army=6, mins=300, gas=0, idle=3, workers=24),    # → S2
    "军队不足": dict(t=480, used=55, cap=80, army=4, mins=500, gas=0, idle=2, workers=30),    # → S3
    "农民闲置": dict(t=420, used=50, cap=80, army=12, mins=200, gas=0, idle=10, workers=34),  # → S1
    "无紧迫事": dict(t=200, used=20, cap=40, army=12, mins=100, gas=0, idle=2, workers=18),   # → S5
    "满人口":   dict(t=900, used=198, cap=200, army=40, mins=700, gas=200, idle=5, workers=60),  # → S6
}


def render(st):
    return ("时间 %d:%02d｜人口 %d/%d｜矿 %d 气 %d\n农民 %d｜军队 %d｜空闲农民 %d" %
            (st["t"] // 60, st["t"] % 60, st["used"], st["cap"], st["mins"], st["gas"],
             st["workers"], st["army"], st["idle"]))


def gt_strategy(st):
    """规则给出的"正确答案"（可被人工标签覆盖）。"""
    if st["cap"] > 0 and st["used"] >= 0.95 * st["cap"] and st["army"] >= 15:
        return "S6", "人口已满且兵力充足，该进攻"
    if st["cap"] > 0 and st["used"] >= 0.9 * st["cap"]:
        return "S2", "人口接近上限"
    if st["mins"] >= 800:
        return "S3", "存款过高，应把钱变成兵"
    if st["army"] < 10 and st["t"] >= 300:
        return "S3", "军队太少"
    if st["idle"] >= 8:
        return "S1", "空闲农民多"
    return "S5", "无紧迫问题，攀科技"


def gt_diagnosis(st):
    if st["cap"] > 0 and st["used"] >= 0.95 * st["cap"] and st["army"] >= 15:
        return "其他"
    if st["cap"] > 0 and st["used"] >= 0.9 * st["cap"]:
        return "人口"
    if st["mins"] >= 800:
        return "存款"
    if st["army"] < 10 and st["t"] >= 300:
        return "军队"
    if st["idle"] >= 8:
        return "农民"
    return "其他"


# 致命错：与局面直接矛盾的选择
def is_fatal(st, key):
    if st["used"] >= 0.95 * st["cap"] and key in ("S1", "S3"):   # 满人口还造单位
        return True
    if st["cap"] > 0 and st["used"] < 0.5 * st["cap"] and key == "S2":  # 人口远未满却补水晶
        return True
    return False


def make_states(n_per=6, seed=0):
    rnd = random.Random(seed)
    out = []
    for name, base in REGIMES.items():
        for _ in range(n_per):
            st = dict(base)
            st["used"] = max(1, int(base["used"] + rnd.randint(-2, 2)))
            st["cap"] = max(st["used"], base["cap"])
            st["army"] = max(0, base["army"] + rnd.randint(-2, 2))
            st["mins"] = max(0, base["mins"] + rnd.randint(-150, 150))
            st["idle"] = max(0, base["idle"] + rnd.randint(-2, 2))
            st["workers"] = max(1, base["workers"] + rnd.randint(-2, 2))
            out.append((name, st))
    return out


def nums_in(text):
    return set(re.findall(r"\d+", text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="每个 regime 的样本数")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model-name", default="qwen")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--repeat", type=int, default=3, help="一致性检查重复次数")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--form", default="choice", choices=["choice", "tree"],
                    help="choice=8选1；tree=决策树（3 个二元问题 + few-shot）")
    ap.add_argument("--priming", default="none",
                    choices=["none", "strategy", "strategy_shots"],
                    help="提示词增强：none=原样；strategy=给定固定战术；strategy_shots=再加示例")
    a = ap.parse_args()

    llm = llm_client.build(a.backend, model_path=a.model_path, base_url=a.base_url,
                           model_name=a.model_name)
    states = make_states(n_per=a.n)
    opts = "\n".join("%s: %s" % (k, v[0]) for k, v in STRATEGIES.items())

    # 三种提示词变体：验证"先给背景/战术/示例"到底有没有用
    DOCTRINE = ("本局固定执行**狂热者快攻**，你的职责只是节奏控制：\n"
                "· 人口到上限 90% 就补水晶（S2）；\n"
                "· 军队少于 10 就量产狂热者（S3）；\n"
                "· 空闲农民 ≥8 才补农民（S1）；\n"
                "· 以上都不成立就进攻（S6）。\n")
    SHOTS = ("参考示例：\n"
             "· 人口 46/47、军队 6、矿 300 → S2（人口卡住）\n"
             "· 人口 60/80、军队 4、矿 520 → S3（军队太少）\n"
             "· 人口 200/200、军队 40、矿 800 → S6（人口已满，该打）\n")
    sys_msg = SYSTEM if a.priming == "none" else (DOCTRINE + "\n" + SYSTEM)
    if a.priming == "strategy_shots":
        sys_msg = DOCTRINE + SHOTS + "\n" + SYSTEM

    correct = fatal = 0
    picks = Counter()
    lat = []
    consistent = 0
    grounded = 0
    diag_ok = 0

    for name, st in states:
        text = render(st)
        gt, why = gt_strategy(st)
        if a.form == "tree":
            from commander_tree import TreeCommander
            tc = TreeCommander(llm)
            t0 = time.time()
            cmd, k, raw = tc.propose(text)
            lat.append(time.time() - t0)
            picks[k] += 1
            if k == gt:
                correct += 1
            if k and is_fatal(st, k):
                fatal += 1
            # tree 形式只跑一次决策，无法据此判断重复一致性 → 记为未测，不虚报
            d = llm.chat([{"role": "system", "content": "只回答一个词：人口 / 军队 / 存款 / 农民 / 其他。"},
                          {"role": "user", "content": text + "\n当前最需要解决的问题是哪一类？"}],
                         max_new_tokens=a.max_tokens, temperature=0.0)
            if gt_diagnosis(st) in (d or ""):
                diag_ok += 1
            continue

        picks_this = []
        for _ in range(a.repeat):
            t0 = time.time()
            raw = llm.chat([{"role": "system", "content": sys_msg},
                            {"role": "user", "content": "当前局势：\n%s\n\n可选策略：\n%s\n\n输出编号：" % (text, opts)}],
                           max_new_tokens=a.max_tokens, temperature=0.0)
            lat.append(time.time() - t0)
            picks_this.append(parse_choice(raw))
        k = picks_this[0]
        picks[k] += 1
        if k == gt:
            correct += 1
        if k and is_fatal(st, k):
            fatal += 1
        if len(set(picks_this)) == 1 and picks_this[0] is not None:
            consistent += 1
        # 诊断（单独问一次）
        d = llm.chat([{"role": "system", "content": "只回答一个词：人口 / 军队 / 存款 / 农民 / 其他。"},
                      {"role": "user", "content": text + "\n当前最需要解决的问题是哪一类？"}],
                     max_new_tokens=a.max_tokens, temperature=0.0)
        if gt_diagnosis(st) in (d or ""):
            diag_ok += 1

    n = len(states)
    print("模型: %s（%s）  样本 %d，每题重复 %d 次" % (a.model_name, a.backend, n, a.repeat))
    print("延迟: 平均 %.2fs  中位 %.2fs" % (sum(lat) / len(lat), sorted(lat)[len(lat) // 2]))
    print("A 按诊断行动（8选1）准确率: %d/%d = %.0f%%" % (correct, n, 100.0 * correct / n))
    print("B 诊断准确率:            %d/%d = %.0f%%" % (diag_ok, n, 100.0 * diag_ok / n))
    print("C 致命错:                %d 次" % fatal)
    print("D 策略覆盖:              %d/8  分布 %s" % (len([k for k in picks if k]), dict(picks)))
    if a.form == "tree":
        print("E 重复一致率:            未测（tree 形式只跑一次决策）")
    else:
        print("E 重复一致率:            %d/%d = %.0f%%" % (consistent, n, 100.0 * consistent / n))
    major = max(Counter(gt_strategy(st)[0] for _, st in states).values())
    print("\n多数类基线: %d/%d = %.0f%%（全答同一答案的准确率；超过它才算有判别力）"
          % (major, n, 100.0 * major / n))
    ok = (correct >= 0.8 * n and fatal == 0 and len([k for k in picks if k]) >= 4
          and (a.form == "tree" or consistent >= 0.8 * n) and correct > major)
    print("\n判定: %s" % ("✅ 够用，可以进回路" if ok else "❌ 不够用，先只做离线教练"))


if __name__ == "__main__":
    main()
