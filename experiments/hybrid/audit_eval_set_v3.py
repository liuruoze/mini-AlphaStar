# -*- coding: utf-8 -*-
"""v3 题集体检：能想到的每一项都写成断言。

检查项：
  1  结构：条数、id 连续且唯一、字段齐全
  2  哈希：与文件内容一致、与 .sha256 一致
  3  标签：与 doctrine.decide_bits 重算一致
  4  渲染：prompt 与 state 一致（数字逐个核对）
  5  物理：容量=15×基地+8×水晶、单位≤人口、空闲≤农民、农民≥1
  6  因果：有气必有气矿、有使徒必有控制核心、有兵营必有水晶
  7  语义：无水晶时第 3 位必为 0；边界项确实只动了一个数
  8  重复：没有两条完全相同的局面
  9  常量：system 文本与 prompt_spec_v3.md 逐字一致
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from doctrine import decide_bits, BIT_NAMES  # noqa: E402

DATA = os.path.join(HERE, "eval_set_v3.json")
SPEC = os.path.join(HERE, "prompt_spec_v3.md")

fails = []


def ck(name, ok, detail=""):
    print("%-34s %s %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        fails.append(name)


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    items = d["items"]

    # 1 结构
    ck("结构/条数=120", len(items) == 120, str(len(items)))
    ids = [it["id"] for it in items]
    ck("结构/id 唯一且连续", len(set(ids)) == 120 and ids == ["v3-%03d" % (i + 1) for i in range(120)])
    need = {"id", "src", "state", "label", "prompt", "label_names"}
    ck("结构/字段齐全", all(need <= set(it) for it in items))

    # 2 哈希
    h = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    ck("哈希/与内容一致", h == d["sha256"])
    ck("哈希/与 .sha256 文件一致", h == open(DATA + ".sha256").read().strip())

    # 3 标签
    bad = [it["id"] for it in items if decide_bits(**it["state"]) != it["label"]]
    ck("标签/重算一致", not bad, ",".join(bad[:3]))

    # 4 渲染
    mism = []
    for it in items:
        s, p = it["state"], it["prompt"]
        want = [
            "时间 %d:%02d" % (s["t"] // 60, s["t"] % 60),
            "人口 %d/%d" % (s["used"], s["cap"]),
            "余量 %d" % (s["cap"] - s["used"]),
            "矿 %d 气 %d" % (s["minerals"], s["gas"]),
            "农民 %d" % s["workers"],
            "空闲农民 %d" % s["idle"],
        ]
        if not all(w in p for w in want):
            mism.append(it["id"])
        # 部队构成里的总数必须等于 army
        m = re.search(r"部队构成：(.+)", p)
        if m:
            txt = m.group(1)
            tot = sum(int(x) for x in re.findall(r"×(\d+)", txt))
            if txt != "无" and tot != s["army"]:
                mism.append(it["id"] + "(部队总数)")
    ck("渲染/数字一致", not mism, ",".join(mism[:3]))

    # 5 物理
    v = []
    for it in items:
        s = it["state"]
        cap_want = min(200, 15 * s["nexus"] + 8 * s["pylons"])
        if s["cap"] != cap_want:
            v.append(it["id"] + "(容量)")
        if s["workers"] + s["army"] > s["used"]:
            v.append(it["id"] + "(单位>人口)")
        if s["used"] > s["cap"]:
            v.append(it["id"] + "(人口超上限)")
        if s["idle"] > s["workers"]:
            v.append(it["id"] + "(空闲>农民)")
        if s["workers"] < 1:
            v.append(it["id"] + "(无农民)")
    ck("物理/约束", not v, ",".join(v[:3]))

    # 6 因果
    c = []
    for it in items:
        s = it["state"]
        bld = it["prompt"].split("建筑：")[1].split("\n")[0]
        if s["gas"] > 0 and "气矿" not in bld:
            c.append(it["id"] + "(有气无气矿)")
        if s["stalkers"] > 0 and s["cyber"] == 0:
            c.append(it["id"] + "(有使徒无控制核心)")
        if s["cyber"] > 0 and s["gw"] == 0:
            c.append(it["id"] + "(有控制核心无兵营)")
        if s["stalkers"] > 0 and s["cyber"] == 0:
            c.append(it["id"] + "(有使徒无核心)")
        if s["gw"] > 0 and s["pylons"] == 0 and it["src"] != "boundary:无水晶有兵营":
            c.append(it["id"] + "(有兵营无水晶)")
    ck("因果/资源与建筑", not c, ",".join(c[:3]))

    # 7 语义
    s7 = []
    for it in items:
        s = it["state"]
        if s["pylons"] == 0 and it["label"][2] != "0":
            s7.append(it["id"] + "(无水晶却要造兵营)")
    ck("语义/无水晶→兵营位=0", not s7, ",".join(s7[:3]))

    # 8 重复
    seen = Counter(json.dumps(it["state"], sort_keys=True) for it in items)
    dup = [k for k, n in seen.items() if n > 1]
    ck("重复/无相同局面", not dup, "%d 组重复" % len(dup))

    # 9 system 与规范一致
    spec = open(SPEC, encoding="utf-8").read()
    blk = re.search(r"```\n(你是星际争霸2.*?)\n```", spec, re.S)
    same = bool(blk) and blk.group(1).strip() == d["system"].strip()
    ck("常量/system 与规范一致", same)

    # 10 名称：题目里出现的单位必须全部来自术语表，且只允许纯狂热者体系的单位
    from sc2_terms import resolve
    ALLOWED = {"狂热者", "农民", "基地", "水晶", "兵营", "无"}
    name_bad = []
    for it in items:
        body = it["prompt"].split("输出")[0]
        seg = []
        for line in body.splitlines():
            if line.startswith("部队构成：") or line.startswith("建筑："):
                seg.append(line.split("：", 1)[1])
        for chunk in "、".join(seg).split("、"):
            name = chunk.split("×")[0].strip()
            if not name:
                continue
            if name == "无":
                continue
            if name not in ALLOWED:
                name_bad.append(it["id"] + ":" + name)
            elif resolve(name) is None:
                name_bad.append(it["id"] + ":" + name + "(表里查不到)")
    ck("名称/仅允许狂热者体系且可解析", not name_bad, ",".join(name_bad[:3]))

    # 11 时间可行性：局面在该时间点造得出来
    tw, tg, ta = [], [], []
    for it in items:
        st = it["state"]
        t = st["t"]
        if st["workers"] > (12 + t / 12.0) * 1.15:
            tw.append(it["id"])
        if st["gw"] > 1 + t // 45:
            tg.append(it["id"])
        # 兵营被拆是允许的；兵力上限按"曾经可能有过的兵营数"估
        max_gw_ever = 1 + int(t // 45)
        cap_army = (t / 27.0) * (st["gw"] if st["gw"] > 0 else max_gw_ever)
        if st["army"] > cap_army * 1.2:
            ta.append(it["id"] + "(兵超出产能)")
        if t < 45 and st["army"] > 0:
            ta.append(it["id"] + "(开局就有兵)")
    ck("时间可行性/农民", not tw, ",".join(tw[:3]))
    ck("时间可行性/兵营", not tg, ",".join(tg[:3]))
    ck("时间可行性/兵力", not ta, ",".join(ta[:3]))

    # 12 经济可行性：花掉的矿不超过采到的
    eco = []
    for it in items:
        st = it["state"]
        mined = 0.75 * (12 + st["workers"]) / 2.0 * st["t"] + 50
        spent = 50 * st["workers"] + 150 * st["gw"] + 100 * st["pylons"] + 100 * st["army"]
        if spent > mined * 1.3:
            eco.append("%s(花%d/采%d)" % (it["id"], spent, mined))
    ck("经济可行性/矿不超采", not eco, ",".join(eco[:3]))

    # 附加：分布与边界翻转
    print("\n各位为 1：" + str({n: sum(1 for it in items if it["label"][i] == "1")
                              for i, n in enumerate(BIT_NAMES)}))
    print("来源分布：" + str(Counter(it["src"].split(":")[0] for it in items)))
    # 边界对照：按声明的期望位逐对校验
    print("\n边界对照（每对必须只翻一位，且是指定那一位）：")
    by_name = {it["src"].split(":", 1)[1]: it for it in items if ":" in it["src"]}
    PAIRS = [("农民14", "农民15", "补农民"), ("兵营2", "兵营3", "补兵营"),
             ("兵7", "兵8", "全面进攻"), ("兵23", "兵24", "量产狂热者"),
             ("余量9", "余量10", "补水晶"), ("矿99", "矿100", "量产狂热者"),
             ("余量1", "余量2", "量产狂热者"), ("农民10", "农民9", "量产狂热者")]
    bad_pair = []
    for a, b, bit in PAIRS:
        la, lb = by_name[a]["label"], by_name[b]["label"]
        diff = [i for i in range(5) if la[i] != lb[i]]
        ok = (len(diff) == 1 and BIT_NAMES[diff[0]] == bit)
        if not ok:
            bad_pair.append("%s/%s" % (a, b))
        print("  %-8s %s → %-8s %s  期望翻「%s」 实际翻 %s  %s"
              % (a, la, b, lb, bit, BIT_NAMES[diff[0]] if len(diff) == 1 else diff,
                 "OK" if ok else "FAIL"))
    ck("边界/成对翻转正确", not bad_pair, ",".join(bad_pair))

    print("\n总计失败项：%d %s" % (len(fails), fails if fails else ""))


if __name__ == "__main__":
    main()
