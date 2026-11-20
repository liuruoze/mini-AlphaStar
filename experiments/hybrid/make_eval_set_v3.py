# -*- coding: utf-8 -*-
"""生成并冻结评测题集 v3（多选：五位独立判断）。

规则（2026-11-20 19:33 定稿）：
  1. 补水晶：没有水晶，或（人口余量 ≤9 且人口上限未满 200）
  2. 补农民：农民 ≤14 且人口余量 ≥1
  3. 补兵营：兵营 <3 且至少有一个水晶
  4. 量产狂热者：兵 <24 且矿 ≥100 且农民 ≥10 且兵营 ≥1 且余量 ≥2
  5. 全面进攻：兵 ≥8

题集：120 条 = 边界 28 条（薄弱阈值各 4 条）+ 随机 92 条。
约束：容量 15×基地+8×水晶；单位 ≤ 人口；时间可行性；经济可行性；无重复。
"""

import argparse
import hashlib
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from doctrine import decide_bits, BIT_NAMES  # noqa: E402
from obs_to_text import CN as UNIT_CN        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v3.json")
SEED = 20261120
TARGET_ONES = 60        # 每位目标 60 个 1（共 120 题）

SYSTEM = """你是星际争霸2神族的战术指挥官。本局固定执行狂热者快攻。以下五件事要分别判断“现在是否该做”，可以同时为真：

1. 补水晶：没有水晶，或（人口余量（上限−已用）≤ 9 且人口上限未满 200）
2. 补农民：农民 ≤ 14 且人口余量 ≥ 1
3. 补兵营：兵营 < 3 且至少有一个水晶（没有能量场就放不下）
4. 量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2
5. 全面进攻：兵 ≥ 8

按 1..5 的顺序输出 5 位 0/1（1 = 现在该做），不要解释。"""


def render(st):
    comp = []
    if st["zealots"]:
        comp.append("%s×%d" % (UNIT_CN["Zealot"], st["zealots"]))
    bld = ["基地×%d" % st["nexus"]]
    if st["pylons"]:
        bld.append("%s×%d" % (UNIT_CN["Pylon"], st["pylons"]))
    if st["gw"]:
        bld.append("%s×%d" % (UNIT_CN["Gateway"], st["gw"]))
    return ("时间 %d:%02d｜人口 %d/%d（余量 %d）\n"
            "矿 %d 气 0｜农民 %d\n"
            "部队构成：%s\n"
            "建筑：%s\n\n输出（5 位 0/1）："
            % (st["t"] // 60, st["t"] % 60, st["used"], st["cap"], st["cap"] - st["used"],
               st["minerals"], st["workers"],
               "、".join(comp) if comp else "无", "、".join(bld)))


def _afford_t(t, workers, army, gw, pylons):
    """把时间抬到"这套建造负担得起"为止；时间不影响任何一位标签。"""
    while t < 900:
        mined = 0.75 * (12 + workers) / 2.0 * t + 50
        if 50 * workers + 150 * gw + 100 * pylons + 100 * army <= mined * 1.25:
            break
        t += 60
    return t


def make(rnd, t, used, workers, army, gw, pylons, minerals, cyber=0, nexus=1,
         idle=0, raw=False):
    cap = min(200, 15 * nexus + 8 * pylons)
    used = min(used, cap)
    if workers + army > used:
        army = max(0, used - workers)
        if workers + army > used:
            workers = max(1, used - army)

    if not raw:
        t = _afford_t(t, workers, army, gw, pylons)
        # 时间可行性：农民 12s/个（开局 12 个）、兵营 45s/座、狂热者 27s/个/兵营
        workers = min(workers, int(12 + t / 12.0))
        gw = min(gw, 1 + int(t // 45))
        max_gw_ever = 1 + int(t // 45)
        if t < 45:
            army = 0
        else:
            army = min(army, int((t / 27.0) * (gw if gw > 0 else max_gw_ever)))
        # 经济可行性：花的矿不超过采到的
        mined = 0.75 * (12 + workers) / 2.0 * t + 50
        spent_fixed = 50 * workers + 150 * gw + 100 * pylons
        army = min(army, int(max(0, (mined * 1.25 - spent_fixed) / 100.0)))
        if workers + army > used:
            army = max(0, used - workers)

    return dict(t=t, used=used, cap=cap, workers=workers, army=army, zealots=army,
                stalkers=0, gw=gw, pylons=pylons, cyber=0, nexus=nexus, idle=idle,
                minerals=minerals, gas=0, geysers=0)


def boundary(rnd):
    """28 条：六对（各 2 条）+ 四个薄弱阈值（各 4 条）+ 三条单独。"""
    out = []

    def one(name, pylons, gap, workers, army, gw, minerals, expect=None, note=""):
        cap = min(200, 15 + 8 * pylons) if pylons else 15
        t = _afford_t(240, workers, army, gw, pylons)
        st = make(rnd, t=t, used=cap - gap, workers=workers, army=army, gw=gw,
                  pylons=pylons, minerals=minerals, raw=True)
        out.append({"name": name, "state": st, "expect": expect, "note": note})

    one("农民14", 4, 7, 14, 20, 3, 300, "补农民")
    one("农民15", 4, 7, 15, 20, 3, 300, "补农民")
    one("兵营2", 4, 7, 22, 10, 2, 300, "补兵营")
    one("兵营3", 4, 7, 22, 10, 3, 300, "补兵营")
    one("兵7", 5, 10, 24, 7, 4, 400, "全面进攻")
    one("兵8", 5, 10, 24, 8, 4, 400, "全面进攻")
    one("余量9", 6, 9, 20, 8, 3, 300, "补水晶")
    one("余量10", 6, 10, 20, 8, 3, 300, "补水晶")
    one("余量1", 6, 1, 24, 10, 3, 300, "量产狂热者")
    one("余量2", 6, 2, 24, 10, 3, 300, "量产狂热者")
    one("农民10", 4, 17, 10, 6, 2, 200, "量产狂热者")
    one("农民9", 4, 17, 9, 6, 2, 200, "量产狂热者")

    for i, (pyl, gap, w, g, m) in enumerate(
            [(4, 4, 20, 4, 300), (5, 10, 22, 4, 350), (5, 8, 24, 5, 450), (6, 14, 26, 4, 500)], 1):
        one("兵23-%d" % i, pyl, gap, w, 23, g, m, "量产狂热者")
    for i, (pyl, gap, w, g, m) in enumerate(
            [(4, 3, 20, 4, 300), (5, 9, 22, 4, 350), (5, 7, 24, 5, 450), (6, 13, 26, 4, 500)], 1):
        one("兵24-%d" % i, pyl, gap, w, 24, g, m, "量产狂热者")
    for i, (pyl, gap, w, g, a) in enumerate(
            [(5, 10, 20, 3, 5), (5, 12, 22, 4, 8), (6, 14, 24, 4, 6), (4, 9, 20, 3, 3)], 1):
        one("矿99-%d" % i, pyl, gap, w, a, g, 99, "量产狂热者")
    for i, (pyl, gap, w, g, a) in enumerate(
            [(5, 10, 20, 3, 5), (5, 12, 22, 4, 8), (6, 14, 24, 4, 6), (4, 9, 20, 3, 3)], 1):
        one("矿100-%d" % i, pyl, gap, w, a, g, 100, "量产狂热者")

    one("无水晶", 0, 1, 12, 0, 0, 80, None, "无能量场，兵营位强制 0")
    one("兵营被拆", 3, 5, 18, 6, 0, 250, None, "兵营 0 但仍有兵")
    cap24 = min(200, 15 + 8 * 24)
    out.append({"name": "满人口",
                "state": make(rnd, t=_afford_t(900, 60, 60, 6, 24), used=cap24, workers=60,
                              army=60, gw=6, pylons=24, minerals=800, raw=True),
                "expect": None, "note": "人口顶格，补水晶无意义"})
    return out


def random_state(rnd):
    pylons = rnd.choice([0, 1, 1, 2, 3, 3, 4, 4, 5, 6])
    cap = min(200, 15 + 8 * pylons) if pylons else 15
    used = rnd.randint(max(1, int(cap * 0.4)), cap)
    workers = rnd.randint(max(1, min(used - 1, 8)), max(2, used - 1))
    army = rnd.randint(0, max(0, used - workers))
    gw = rnd.choice([0, 1, 1, 2, 2, 3, 3, 4, 5])
    if gw and pylons == 0:
        gw = 0
    return make(rnd, rnd.choice([60, 120, 180, 240, 300, 360, 420, 480, 540]),
                used, workers, army, gw, pylons, rnd.randint(0, 700))


def build():
    rnd = random.Random(SEED)
    items, seen = [], set()

    for b in boundary(rnd):
        key = json.dumps(b["state"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        items.append({"id": "v3-%03d" % (len(items) + 1), "src": "boundary:" + b["name"],
                      "state": b["state"], "label": decide_bits(**b["state"]),
                      "expect": b["expect"], "note": b["note"]})

    # ---- 配平：每位接近 50%，同时限制单一组合的条数，避免"全 0/全 1"刷分 ----
    pool = []
    guard = 0
    while len(pool) < 6000 and guard < 3000000:
        guard += 1
        st = random_state(rnd)
        key = json.dumps(st, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        pool.append(st)

    slots = 120 - len(items)
    base_ones = [sum(1 for it in items if it["label"][i] == "1") for i in range(5)]
    bit_ones = list(base_ones)
    label_count = {}
    for it in items:
        label_count[it["label"]] = label_count.get(it["label"], 0) + 1

    rnd.shuffle(pool)
    picked = []
    for st in pool:
        if len(picked) >= slots:
            break
        bits = decide_bits(**st)
        if label_count.get(bits, 0) >= 8:                     # 单一组合最多 8 条
            continue
        if any(bits[i] == "1" and bit_ones[i] >= TARGET_ONES for i in range(5)):
            continue                                          # 某位超过目标
        picked.append(st)
        label_count[bits] = label_count.get(bits, 0) + 1
        for i in range(5):
            bit_ones[i] += 1 if bits[i] == "1" else 0

    for st in picked:
        items.append({"id": "v3-%03d" % (len(items) + 1), "src": "random",
                      "state": st, "label": decide_bits(**st)})

    for it in items:
        it["prompt"] = render(it["state"])
        it["label_names"] = [n for n, b in zip(BIT_NAMES, it["label"]) if b == "1"] or ["无"]
    return items


def digest(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        d = json.load(open(OUT, encoding="utf-8"))
        bad = [it["id"] for it in d["items"] if decide_bits(**it["state"]) != it["label"]]
        print("条目 %d｜哈希一致 %s｜标签自洽 %s"
              % (len(d["items"]), digest(d["items"]) == d["sha256"], not bad))
        return
    items = build()
    d = {"version": "v3", "frozen_at": "2026-11-20T19:35+08:00",
         "system": SYSTEM, "n": len(items), "sha256": digest(items), "items": items}
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(OUT + ".sha256", "w").write(digest(items) + "\n")
    print("写入 %s  sha256=%s" % (OUT, d["sha256"]))
    print("每位为 1：%s" % {n: sum(1 for it in items if it["label"][i] == "1")
                            for i, n in enumerate(BIT_NAMES)})


if __name__ == "__main__":
    main()
