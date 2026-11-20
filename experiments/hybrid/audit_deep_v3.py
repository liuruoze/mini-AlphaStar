# -*- coding: utf-8 -*-
"""深度体检：双重录入 + 全文扫描 + 抽样打印。

与 audit_eval_set_v3.py 的分工：
  · 那个查"结构/物理/因果"；
  · 这个查"标签是否真按 system 原文算的"——**独立重写一遍规则**，不复用 doctrine.decide_bits。
    两套实现若结果不同，说明至少有一处与规范不符。
"""

import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "eval_set_v3.json")


def label_from_spec(s):
    """逐字按 prompt_spec_v3.md 的 1..5 条重写，不复用 doctrine.py。"""
    gap = s["cap"] - s["used"]
    b1 = 1 if (s["pylons"] == 0 or (gap <= 9 and s["cap"] < 200)) else 0   # 余量≤9 且上限未满
    b2 = 1 if (s["workers"] <= 14 and gap >= 1) else 0                    # 农民≤14 且余量≥1
    b3 = 1 if (s["gw"] < 3 and s["pylons"] >= 1) else 0                    # 兵营<3 且至少一个水晶
    b4 = 1 if (s["army"] < 24 and s["minerals"] >= 100 and s["workers"] >= 10
               and s["gw"] >= 1 and gap >= 2) else 0                       # 量产五条件
    b5 = 1 if s["army"] >= 8 else 0                                        # 兵≥8
    return "%d%d%d%d%d" % (b1, b2, b3, b4, b5)


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    items = d["items"]
    problems = []

    # 1 双重录入：独立实现 vs 存档标签
    diff = [(it["id"], it["label"], label_from_spec(it["state"])) for it in items
            if label_from_spec(it["state"]) != it["label"]]
    problems += [("标签与规范原文不符", x) for x in diff[:5]]

    # 2 全文扫描：逐条核对 prompt 里应出现的每个数字，且不得出现答案
    for it in items:
        s, p = it["state"], it["prompt"]
        need = ["%d:%02d" % (s["t"] // 60, s["t"] % 60), "%d/%d" % (s["used"], s["cap"]),
                "%d" % (s["cap"] - s["used"]), "%d" % s["minerals"], "%d" % s["workers"],
                "×%d" % s["zealots"] if s["zealots"] else "无"]
        for n in need:
            if n not in p:
                problems.append(("prompt 缺字段", (it["id"], n)))
        # 不得泄漏答案编码
        if re.search(r"\b[01]{5}\b", p):
            problems.append(("prompt 泄漏 5 位编码", it["id"]))
        # 不得出现空字段/None
        if "None" in p or "：\n" in p or "、、" in p:
            problems.append(("prompt 有空字段", it["id"]))

    # 3 字段一致性
    for it in items:
        bits = it["label"]
        names = [n for n, b in zip(["补水晶", "补农民", "补兵营", "量产狂热者", "全面进攻"], bits) if b == "1"]
        if it["label_names"] != (names or ["无"]):
            problems.append(("label_names 与 label 不符", it["id"]))
        if it["state"]["cap"] != min(200, 15 * it["state"]["nexus"] + 8 * it["state"]["pylons"]):
            problems.append(("容量与建筑不符", it["id"]))

    # 4 system 三处是否一致（json / 规范 md / 生成脚本）
    spec = open(os.path.join(HERE, "prompt_spec_v3.md"), encoding="utf-8").read()
    m = re.search(r"```\n(你是星际争霸2.*?)\n```", spec, re.S)
    if not m or m.group(1).strip() != d["system"].strip():
        problems.append(("json 与规范 md 的 system 不一致", "system"))
    gen = open(os.path.join(HERE, "make_eval_set_v3.py"), encoding="utf-8").read()
    m2 = re.search(r'SYSTEM = """(.*?)"""', gen, re.S)
    if not m2 or m2.group(1).strip() != d["system"].strip():
        problems.append(("生成脚本与 json 的 system 不一致", "system"))

    # 5 统计
    print("条目 %d｜失败 %d" % (len(items), len(problems)))
    for p in problems[:10]:
        print("  ✗", p)
    per = {n: sum(1 for it in items if it["label"][i] == "1")
           for i, n in enumerate(["补水晶", "补农民", "补兵营", "量产", "进攻"])}
    print("各位为 1：%s" % per)
    print("每类局面数（按 label 组合）：%s"
          % dict(Counter(it["label"] for it in items).most_common(6)))
    print("时间分布：%s" % dict(Counter(it["state"]["t"] // 60 for it in items)))

    # 6 抽样打印（每类各两条，供人眼复核）
    print("\n===== 抽样复核 =====")
    shown = Counter()
    for it in items:
        if shown[it["label"]] >= 2:
            continue
        shown[it["label"]] += 1
        s = it["state"]
        print("\n[%s] 标签=%s (%s)" % (it["id"], it["label"], "/".join(it["label_names"])))
        print("  t=%ds 人口 %d/%d(余%d) 矿%d 农民%d 空闲%d 兵%d 营%d 晶%d"
              % (s["t"], s["used"], s["cap"], s["cap"] - s["used"], s["minerals"],
                 s["workers"], s["idle"], s["army"], s["gw"], s["pylons"]))
        print("  逐条判定：" + " | ".join([
            "1补水晶=%s" % ("✓" if (s["pylons"] == 0 or (s["cap"] - s["used"] <= 9 and s["cap"] < 200)) else "✗"),
            "2补农民=%s" % ("✓" if (s["workers"] <= 14 and s["cap"] - s["used"] >= 1) else "✗"),
            "3补兵营=%s" % ("✓" if (s["gw"] < 3 and s["pylons"] >= 1) else "✗"),
            "4量产=%s" % ("✓" if (s["army"] < 24 and s["minerals"] >= 100 and s["workers"] >= 10
                                  and s["gw"] >= 1 and s["cap"] - s["used"] >= 2) else "✗"),
            "5进攻=%s" % ("✓" if s["army"] >= 8 else "✗")]))


if __name__ == "__main__":
    main()
