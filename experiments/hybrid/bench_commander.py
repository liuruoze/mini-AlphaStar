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
      --base-url http://127.0.0.1:8769/v1 --model-name qwen
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


EN = {
    "S1": "Saturate workers, then add pylons",
    "S2": "Add pylons (raise supply cap)",
    "S3": "Mass Zealots",
    "S4": "Take the natural expansion",
    "S5": "Tech up: Cybernetics Core",
    "S6": "Attack the enemy main with everything",
    "S7": "Fall back and defend the natural",
    "S8": "Add assimilators (gas)",
}
SYS_EN = ("You are a StarCraft II Protoss tactical commander. Pick exactly one option id "
          "from the list and output only that id, no explanation.")


def render_en(st):
    return ("Time %d:%02d | Supply %d/%d | Minerals %d | Vespene %d\n"
            "Workers %d | Army %d | Idle workers %d | Gateways %d" %
            (st["t"] // 60, st["t"] % 60, st["used"], st["cap"], st["mins"], st["gas"],
             st["workers"], st["army"], st["idle"], st.get("gw", 0)))


def render(st):
    return ("时间 %d:%02d｜人口 %d/%d｜矿 %d 气 %d\n农民 %d｜军队 %d｜空闲农民 %d｜兵营 %d 个" %
            (st["t"] // 60, st["t"] % 60, st["used"], st["cap"], st["mins"], st["gas"],
             st["workers"], st["army"], st["idle"], st.get("gw", 0)))


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


# ---- V5 专家的优先级阶梯（取自 experiments/_rule_expert_v5.py 的常量与 P0–P5）----
V5_SUPPLY_RESERVE = 6     # cap - used <= 6 就补 Pylon
V5_PROBE_TARGET = 18      # 农民 < 18 先补农民
V5_ATTACK_AT = 10         # 兵 >= 10 进攻（ATTACK_AT 默认 10）
V5_GATEWAY_TARGET = 4


def gt_v5(st):
    """V5 的决策阶梯（人口 → 农民 → 兵营 → 造兵 → 进攻）。"""
    if st["cap"] - st["used"] <= V5_SUPPLY_RESERVE:
        return "S2", "人口余量≤6，先补水晶"
    if st["workers"] < V5_PROBE_TARGET:
        return "S1", "农民<18，先铺农民"
    if st.get("gw", 99) < V5_GATEWAY_TARGET:
        return "S9", "兵营不足 4 个，先补兵营（狂热者产能）"
    if st["army"] < V5_ATTACK_AT:
        return "S3", "兵<10，继续造兵"
    return "S6", "兵≥10，A-move 进攻"


# V5 阶梯对应的均衡局面（四类各占 25%，避免恒定答案刷分）
# 五类，对应 V5 阶梯的五个分支（含"补兵营"——2026-11-20 用户指出它是快攻命门）
REGIMES_V5 = {
    "人口将满": dict(t=300, used=44, cap=47, army=8, gw=2, mins=250, gas=0, idle=2, workers=24),
    "农民不足": dict(t=360, used=40, cap=70, army=12, gw=3, mins=200, gas=0, idle=1, workers=14),
    "兵营不足": dict(t=300, used=35, cap=70, army=6, gw=1, mins=400, gas=0, idle=2, workers=22),
    "兵少":     dict(t=420, used=50, cap=70, army=4, gw=4, mins=300, gas=0, idle=2, workers=26),
    "兵多":     dict(t=480, used=60, cap=70, army=14, gw=4, mins=300, gas=0, idle=2, workers=28),
}


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
            st["gw"] = max(0, base.get("gw", 0) + rnd.randint(-1, 1))
            out.append((name, st))
    return out


def nums_in(text):
    return set(re.findall(r"\d+", text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="每个 regime 的样本数")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--model-name", default="qwen")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--repeat", type=int, default=3, help="一致性检查重复次数")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--gt", default="rules", choices=["rules", "v5"],
                    help="rules=我手写的8条；v5=按 _rule_expert_v5 的优先级阶梯")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="提问语言")
    ap.add_argument("--votes", type=int, default=1, help="每个二元问题投票次数（多数决）")
    ap.add_argument("--form", default="choice",
                    choices=["choice", "tree", "tree_v5", "tree_d"],
                    help="choice=8选1；tree=决策树（3 个二元问题 + few-shot）")
    ap.add_argument("--priming", default="none",
                    choices=["none", "strategy", "strategy_shots"],
                    help="提示词增强：none=原样；strategy=给定固定战术；strategy_shots=再加示例")
    a = ap.parse_args()

    llm = llm_client.build(a.backend, model_path=a.model_path, base_url=a.base_url,
                           model_name=a.model_name)
    if a.gt == "v5":
        global REGIMES
        REGIMES = REGIMES_V5
        states = make_states(n_per=a.n)
    else:
        states = make_states(n_per=a.n)
    opts = ("\n".join("%s: %s" % (k, EN[k]) for k in STRATEGIES) if a.lang == "en"
            else "\n".join("%s: %s" % (k, v[0]) for k, v in STRATEGIES.items()))

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
        text = render_en(st) if a.lang == "en" else render(st)
        gt, why = (gt_v5(st) if a.gt == "v5" else gt_strategy(st))
        if a.form == "tree_d":
            # 关键改动：把"需不需要减"的量预先算好写进摘要，模型只做一次比较
            text2 = (text + "\n人口余量 %d（上限-已用）" % (st["cap"] - st["used"]))

            def askd(q, shots):
                votes = []
                for _ in range(max(1, a.votes)):
                    raw = llm.chat([{"role": "system", "content": "只答一个字：是 或 否。"},
                                    {"role": "user", "content": shots + text2 + "\n" + q}],
                                   max_new_tokens=8, temperature=0.0)
                    t = (raw or "").strip()
                    if "否" in t[:6]:
                        votes.append(False)
                    elif "是" in t[:6]:
                        votes.append(True)
                if not votes:
                    return None
                return sum(votes) * 2 > len(votes)
            t0 = time.time()
            q1 = askd("人口余量是否小于等于 6？",
                      "示例：人口余量=3，问\"≤6?\"→是。\n示例：人口余量=30，问\"≤6?\"→否。\n")
            if q1:
                k = "S2"
            else:
                q2 = askd("农民数量是否少于 18？",
                          "示例：农民=14，问\"<18?\"→是。\n示例：农民=26，问\"<18?\"→否。\n")
                if q2:
                    k = "S1"
                else:
                    q3 = askd("军队数量是否少于 10？",
                              "示例：军队=4，问\"<10?\"→是。\n示例：军队=14，问\"<10?\"→否。\n")
                    k = "S3" if q3 else "S6"
            lat.append(time.time() - t0)
            picks[k] += 1
            if k == gt:
                correct += 1
            if is_fatal(st, k):
                fatal += 1
            continue

        if a.form == "tree_v5":
            # 逐条执行 V5 的阶梯：三个 yes/no + few-shot
            def ask(q, shots):
                raw = llm.chat([{"role": "system", "content": "只答一个字：是 或 否。"},
                                {"role": "user", "content": shots + text + "\n" + q}],
                               max_new_tokens=8, temperature=0.0)
                t = (raw or "").strip()
                if "否" in t[:6]:
                    return False
                if "是" in t[:6]:
                    return True
                return None
            t0 = time.time()
            q1 = ask("人口上限减去已用人口，是否小于等于 6？",
                     "示例：47-44=3，问\"≤6?\"→是。\n示例：70-40=30，问\"≤6?\"→否。\n")
            if q1:
                k = "S2"
            else:
                q2 = ask("农民数量是否少于 18？",
                         "示例：农民=14，问\"<18?\"→是。\n示例：农民=26，问\"<18?\"→否。\n")
                if q2:
                    k = "S1"
                else:
                    q3 = ask("军队数量是否少于 10？",
                             "示例：军队=4，问\"<10?\"→是。\n示例：军队=14，问\"<10?\"→否。\n")
                    k = "S3" if q3 else "S6"
            lat.append(time.time() - t0)
            picks[k] += 1
            if k == gt:
                correct += 1
            if is_fatal(st, k):
                fatal += 1
            d = llm.chat([{"role": "system", "content": "只回答一个词：人口 / 军队 / 存款 / 农民 / 其他。"},
                          {"role": "user", "content": text + "\n当前最需要解决的问题是哪一类？"}],
                         max_new_tokens=a.max_tokens, temperature=0.0)
            if gt_diagnosis(st) in (d or ""):
                diag_ok += 1
            continue

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
            user_msg = (("Situation://n%s//n//nOptions://n%s//n//nAnswer with the id only:" % (text, opts))
                        if a.lang == "en" else
                        ("当前局势：\n%s\n\n可选策略：\n%s\n\n输出编号：" % (text, opts)))
            raw = llm.chat([{"role": "system", "content": SYS_EN if a.lang == "en" else sys_msg},
                            {"role": "user", "content": user_msg}],
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
    if a.form in ("tree", "tree_v5", "tree_d"):
        print("E 重复一致率:            未测（该形式只跑一次决策）")
    else:
        print("E 重复一致率:            %d/%d = %.0f%%" % (consistent, n, 100.0 * consistent / n))
    gtf = gt_v5 if a.gt == "v5" else gt_strategy
    major = max(Counter(gtf(st)[0] for _, st in states).values())
    print("\n多数类基线: %d/%d = %.0f%%（全答同一答案的准确率；超过它才算有判别力）"
          % (major, n, 100.0 * major / n))
    ok = (correct >= 0.8 * n and fatal == 0 and len([k for k in picks if k]) >= 4
          and (a.form == "tree" or consistent >= 0.8 * n) and correct > major)
    print("\n判定: %s" % ("✅ 够用，可以进回路" if ok else "❌ 不够用，先只做离线教练"))


if __name__ == "__main__":
    main()
