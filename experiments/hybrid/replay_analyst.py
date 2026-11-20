#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线教练：读一局的 15 秒采样 + 胜负，产出"报告卡 + 事件检测 +（可选）中文复盘"。

设计原则：**所有数字由脚本算出，LLM 只负责组织语言**。
这样即使模型乱说，也不会编出"军队 30 个"这种不存在的数字——数字都在提示词里，且要求逐字引用。

用法:
  python experiments/hybrid/replay_analyst.py --trace <jsonl> --out <txt> [--llm] [--model-path ...]
  python experiments/hybrid/replay_analyst.py --dir <trace_dir> --out-dir <eval_out> [--limit 6]
"""

import argparse
import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from obs_to_text import summarize_record, _cn, NAMES  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path, errors="ignore") if l.strip()]


def outcome(txt_path):
    if not os.path.exists(txt_path):
        return None
    m = re.search(r"native_outcome:\s*(-?\d+)", open(txt_path, errors="ignore").read())
    return None if not m else int(int(m.group(1)) == 1)


def near(rows, t, win=20):
    c = [r for r in rows if abs(r["t_s"] - t) <= win]
    return c[0] if c else None


def report_card(rows):
    """3/5/8 分钟的定点快照 + 关键事件。"""
    card = {"snapshots": {}, "events": [], "meta": {}}
    last = rows[-1]
    card["meta"]["duration_s"] = last["t_s"]
    card["meta"]["samples"] = len(rows)

    for t in (180, 300, 480):
        r = near(rows, t)
        if not r:
            continue
        own = r.get("own", {}) or {}
        army = r.get("army", 0)
        workers = r.get("workers", 0)
        budget = r.get("minerals", 0) + r.get("vespene", 0)
        card["snapshots"]["t%d" % t] = {
            "army": army, "workers": workers, "cap": r.get("food_cap", 0),
            "used": r.get("food_used", 0), "minerals": r.get("minerals", 0),
            "vespene": r.get("vespene", 0), "idle": r.get("idle_workers", 0),
            "buildings": sum(v for k, v in own.items()
                             if NAMES.get(int(k)) in ("Pylon", "Gateway", "Nexus", "Assimilator")),
        }

    # 事件：人口顶格 / 军队停滞 / 空闲农民 / 不造科技 / 存款闲置
    capped = [r for r in rows if r["food_cap"] > 0 and r["food_used"] >= 0.95 * r["food_cap"]]
    if len(capped) >= 2:
        card["events"].append("人口顶格 %d 次采样（首次 %.0fs，共 %.1f 分钟）"
                              % (len(capped), capped[0]["t_s"], capped[-1]["t_s"] / 60.0))

    stall = []
    for i, r in enumerate(rows):
        prev = near(rows, r["t_s"] - 120)
        if prev and r.get("army", 0) - prev.get("army", 0) < 1:
            stall.append(r["t_s"])
    if len(stall) >= 3:
        card["events"].append("军队连续 2 分钟不增长：%d 次（首次 %.1f 分钟）"
                              % (len(stall), stall[0] / 60.0))

    idle = [r for r in rows if r.get("idle_workers", 0) >= 8]
    if idle:
        card["events"].append("空闲农民 ≥8：%d 次（峰值 %d）"
                              % (len(idle), max(r.get("idle_workers", 0) for r in idle)))

    tech = set()
    for r in rows:
        for uid in (r.get("own") or {}):
            nm = NAMES.get(int(uid))
            if nm in ("Forge", "CyberneticsCore", "TwilightCouncil", "RoboticsFacility",
                      "Stargate", "TemplarArchives", "FleetBeacon"):
                tech.add(nm)
    card["events"].append("科技建筑：%s" % ("、".join(_cn(t) for t in tech) if tech else "无"))

    rich = [r for r in rows if r.get("minerals", 0) >= 400]
    if rich:
        card["events"].append("存款 ≥400 矿：%d 次（峰值 %d）"
                              % (len(rich), max(r.get("minerals", 0) for r in rich)))
    return card


def render(card, win, rows):
    lines = []
    lines.append("【结果】%s，时长 %.1f 分钟" %
                 ("胜" if win == 1 else ("负" if win == 0 else "未知"),
                  card["meta"]["duration_s"] / 60.0))
    lines.append("【定点快照】")
    for k, v in card["snapshots"].items():
        lines.append("  %s分钟: 军队 %d｜农民 %d｜人口 %d/%d｜矿 %d 气 %d｜空闲农民 %d"
                     % (int(k[1:]) // 60, v["army"], v["workers"], v["used"], v["cap"],
                        v["minerals"], v["vespene"], v["idle"]))
    lines.append("【事件】")
    for e in card["events"]:
        lines.append("  · " + e)
    lines.append("【末帧】")
    lines.append("  " + summarize_record(rows[-1]).replace("\n", "\n  "))
    return "\n".join(lines)




def diagnose(card):
    """规则化的诊断，不依赖模型：按"最能解释败因"的顺序给出问题清单。"""
    out = []
    snaps = card["snapshots"]
    peak_min = max((v["minerals"] for v in snaps.values()), default=0)
    peak_idle = max((v["idle"] for v in snaps.values()), default=0)
    stalled = [e for e in card["events"] if e.startswith("军队连续")]
    tech = [e for e in card["events"] if e.startswith("科技建筑")]
    capped = [e for e in card["events"] if e.startswith("人口顶格")]

    if peak_min >= 400:
        out.append(("资源没花出去", "存款峰值 %d 矿（≥400 视为闲置），训练/建造队列长期空着" % peak_min))
    if peak_idle >= 8:
        out.append(("农民闲置", "峰值 %d 个空闲农民（≥8 视为闲置），采集与建造都在等人" % peak_idle))
    if stalled:
        out.append(("中期停止产兵", stalled[0].replace("军队连续 2 分钟不增长：", "").strip()))
    if capped:
        out.append(("人口顶格", capped[0].replace("人口顶格 ", "").strip()))
    if tech and "无" in tech[0]:
        out.append(("零科技", "全程未建造任何科技建筑，兵种与升级停在起点"))
    if not out:
        out.append(("未见明显宏观缺陷", "在 3/5/8 分钟三个快照上都没有闲置或停滞"))
    return out


def render_diagnosis(card, win):
    tag = "胜" if win == 1 else ("负" if win == 0 else "未知")
    lines = ["【结论】%s局，时长 %.1f 分钟" % (tag, card["meta"]["duration_s"] / 60.0)]
    for name, detail in diagnose(card):
        lines.append("【问题】%s：%s" % (name, detail))
    # 只给一条建议：按问题优先级取第一条
    d = diagnose(card)
    tip = {
        "资源没花出去": "存款超 400 就立刻补产能/出兵，不要攒",
        "农民闲置": "空闲农民超过 8 就拉去采矿或建造",
        "中期停止产兵": "8 分钟后保持兵营不停产，每 2 分钟至少 +5 兵",
        "人口顶格": "人口到 95% 就补水晶，别让人口卡住产兵",
        "零科技": "开一矿后补控制核心，至少解锁一个兵种",
        "未见明显宏观缺陷": "保持当前节奏，重点看交战时机",
    }.get(d[0][0], "")
    if tip:
        lines.append("【建议】" + tip)
    return "\n".join(lines)


SYSTEM = (
    "你是星际争霸2神族教练，正在复盘一局。规则：\n"
    "1) 先看【结果】：胜局就总结“赢在哪里”，负局才总结“输在哪里”，不要弄反；\n"
    "2) 必须原样引用卡片里的至少两个数字（例如“8分钟军队12”“存款1030”）；\n"
    "3) 不得虚构卡片里没有的数量、时间或单位；\n"
    "4) 输出 3 句中文：第一句结论，第二句引用数字支撑，第三句给一条不超过 20 字的建议。"
)


def narrate(card_text, llm):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "这是一局的数据：\n%s\n\n请复盘。" % card_text}]
    return llm.chat(msgs, max_new_tokens=220, temperature=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None)
    ap.add_argument("--out", default=None, help="对应的评测 .txt（给胜负）")
    ap.add_argument("--dir", default=None, help="批量：trace 目录")
    ap.add_argument("--out-dir", default=None, help="批量：评测输出目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--summary", action="store_true", help="只统计问题出现频率，不逐局打印")
    a = ap.parse_args()

    files = []
    if a.dir:
        files = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
        if a.limit:
            files = files[:a.limit]
    elif a.trace:
        files = [a.trace]
    else:
        ap.error("给 --trace 或 --dir")

    llm = None
    if a.llm:
        import llm_client
        llm = llm_client.build(None, model_path=a.model_path)

    if a.summary:
        from collections import Counter
        cnt, tot = Counter(), 0
        by_win = {1: Counter(), 0: Counter()}
        n_win = {1: 0, 0: 0}
        for f in files:
            rows = load(f)
            txt = os.path.join(a.out_dir or "", os.path.basename(f).replace(".jsonl", ".txt"))
            win = outcome(txt)
            card = report_card(rows)
            keys = [k for k, _ in diagnose(card)]
            tot += 1
            if win in (0, 1):
                n_win[win] += 1
            for k in keys:
                cnt[k] += 1
                if win in (0, 1):
                    by_win[win][k] += 1
        print("%-16s %8s %10s %10s" % ("问题", "出现局数", "胜局中", "负局中"))
        for k, v in cnt.most_common():
            print("%-16s %8d %10s %10s" % (
                k, v,
                "%d/%d" % (by_win[1][k], n_win[1]) if n_win[1] else "-",
                "%d/%d" % (by_win[0][k], n_win[0]) if n_win[0] else "-"))
        print("\n总局数 %d（胜 %d / 负 %d）" % (tot, n_win[1], n_win[0]))
        return

    for f in files:
        rows = load(f)
        txt = a.out or os.path.join(a.out_dir or "", os.path.basename(f).replace(".jsonl", ".txt"))
        win = outcome(txt)
        card = report_card(rows)
        text = render(card, win, rows)
        print("\n" + "=" * 70)
        print(os.path.basename(f).replace(".jsonl", ""))
        print(text)
        print(render_diagnosis(card, win))
        if llm is not None:
            print("\n【LLM 复盘】")
            print("  " + narrate(text, llm).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
