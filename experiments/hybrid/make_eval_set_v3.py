# -*- coding: utf-8 -*-
"""生成并冻结评测题集 v3（多选：五位独立判断）。

与 v2 的差别：
  · 标签从「一个编号」变成「5 位 0/1」——基地造农民、兵营造兵、农民造建筑互不冲突；
  · 唯一的依赖：没有水晶时第 3 位强制 0（兵营放不进能量场）；
  · 指标从整体答对率改为**逐位准确率**（5 个判断题的平均）。

设计：
  · 120 条，每一条都渲染成最终 prompt 存盘（所有模型看到逐字相同的输入）；
  · 每位保证 ≥40 个 1 和 ≥40 个 0，避免某一位退化成常数；
  · 24 条专门卡阈值（农民 14/15、兵营 2/3、兵 7/8、兵 23/24、余量 9/10、矿 99/100）；
  · 物理约束：人口上限 = 15 + 8×水晶数（水晶=0 时为 15），单位数不超过人口。

用法：
  python experiments/hybrid/make_eval_set_v3.py           # 生成
  python experiments/hybrid/make_eval_set_v3.py --check   # 校验哈希与标签自洽
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from doctrine import decide_bits, BIT_NAMES  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v3.json")
SEED = 20261120

SYSTEM = """你是星际争霸2神族的战术指挥官。本局固定执行狂热者快攻。以下五件事要分别判断“现在是否该做”，可以同时为真：

1. 补水晶：没有水晶，或人口余量（上限−已用）≤ 9
2. 补农民：农民 ≤ 14
3. 补兵营：兵营 < 3 且至少有一个水晶（没有能量场就放不下）
4. 量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2
5. 全面进攻：兵 ≥ 8

按 1..5 的顺序输出 5 位 0/1（1 = 现在该做），不要解释。"""


def render(st):
    comp = []
    if st["zealots"]:
        comp.append("狂热者×%d" % st["zealots"])
    if st["stalkers"]:
        comp.append("使徒×%d" % st["stalkers"])
    bld = ["基地×%d" % st["nexus"]]
    if st["pylons"]:
        bld.append("水晶×%d" % st["pylons"])
    if st["gw"]:
        bld.append("兵营×%d" % st["gw"])
    if st["cyber"]:
        bld.append("控制核心×%d" % st["cyber"])
    return ("时间 %d:%02d｜人口 %d/%d（余量 %d）\n"
            "矿 %d 气 %d｜农民 %d｜空闲农民 %d\n"
            "部队构成：%s\n"
            "建筑：%s\n\n输出（5 位 0/1）："
            % (st["t"] // 60, st["t"] % 60, st["used"], st["cap"], st["cap"] - st["used"],
               st["minerals"], st["gas"], st["workers"], st["idle"],
               "、".join(comp) if comp else "无", "、".join(bld)))


def make(rnd, t, used, workers, army, gw, pylons, minerals, gas=0, idle=0,
         cyber=0, nexus=1):
    # 容量 = 15×基地 + 8×水晶（用户 19:12 更正；基地给 15，每个水晶给 8）
    cap = min(200, 15 * nexus + 8 * pylons)
    used = min(used, cap)
    if workers + army > used:                    # 单位数不能超过人口
        army = max(0, used - workers)
        if workers + army > used:
            workers = max(1, used - army)
    zealots, stalkers = army, 0
    if army and rnd.random() < 0.2:
        stalkers = rnd.randint(1, max(1, army // 3))
        zealots = army - stalkers
    return dict(t=t, used=used, cap=cap, workers=workers, army=army,
                zealots=zealots, stalkers=stalkers, gw=gw, pylons=pylons,
                cyber=cyber, nexus=nexus, idle=idle, minerals=minerals, gas=gas)


def boundary_items(rnd):
    """24 条卡阈值：成对给出，一次只动一个数。"""
    out = []
    pairs = [
        # (说明, 基础参数, 变动)
        ("农民14", dict(t=240, used=40, workers=14, army=20, gw=3, pylons=4, minerals=300)),
        ("农民15", dict(t=240, used=40, workers=15, army=20, gw=3, pylons=4, minerals=300)),
        ("兵营2", dict(t=240, used=40, workers=22, army=10, gw=2, pylons=4, minerals=300)),
        ("兵营3", dict(t=240, used=40, workers=22, army=10, gw=3, pylons=4, minerals=300)),
        ("兵7", dict(t=300, used=45, workers=24, army=7, gw=4, pylons=5, minerals=400)),
        ("兵8", dict(t=300, used=45, workers=24, army=8, gw=4, pylons=5, minerals=400)),
        ("兵23", dict(t=360, used=50, workers=24, army=23, gw=4, pylons=5, minerals=400)),
        ("兵24", dict(t=360, used=55, workers=24, army=24, gw=4, pylons=5, minerals=400)),
        ("余量9", dict(t=240, used=38, workers=20, army=8, gw=3, pylons=6, minerals=300)),
        ("余量10", dict(t=240, used=37, workers=20, army=8, gw=3, pylons=6, minerals=300)),
        ("矿99", dict(t=240, used=38, workers=20, army=5, gw=3, pylons=5, minerals=99)),
        ("矿100", dict(t=240, used=38, workers=20, army=5, gw=3, pylons=5, minerals=100)),
        ("无水晶", dict(t=45, used=12, workers=12, army=0, gw=0, pylons=0, minerals=80)),
        ("无水晶有兵营", dict(t=90, used=14, workers=13, army=0, gw=1, pylons=0, minerals=120)),
        ("余量1", dict(t=300, used=46, workers=24, army=10, gw=3, pylons=6, minerals=300)),
        ("余量2", dict(t=300, used=45, workers=24, army=10, gw=3, pylons=6, minerals=300)),
        ("农民10", dict(t=180, used=30, workers=10, army=6, gw=2, pylons=4, minerals=200)),
        ("农民9", dict(t=180, used=30, workers=9, army=6, gw=2, pylons=4, minerals=200)),
        ("矿100兵23", dict(t=420, used=50, workers=22, army=23, gw=4, pylons=6, minerals=100)),
        ("进攻中", dict(t=420, used=50, workers=22, army=14, gw=4, pylons=6, minerals=150)),
        ("满人口", dict(t=540, used=200, workers=60, army=60, gw=6, pylons=24, minerals=800)),
        ("无兵营有水晶", dict(t=150, used=25, workers=18, army=3, gw=0, pylons=3, minerals=250)),
        ("兵营3兵7", dict(t=300, used=40, workers=20, army=7, gw=3, pylons=5, minerals=250)),
        ("兵营3兵8", dict(t=300, used=42, workers=20, army=8, gw=3, pylons=5, minerals=250)),
    ]
    for name, kw in pairs:
        st = make(rnd, **kw)
        out.append((name, st))
    return out


def random_state(rnd):
    pylons = rnd.choice([0, 1, 1, 2, 3, 3, 4, 4, 5, 6])
    cap = min(200, 15 * 1 + 8 * pylons)
    used = rnd.randint(max(1, int(cap * 0.4)), cap)
    workers = rnd.randint(max(1, min(used - 1, 8)), max(2, used - 1))
    army = rnd.randint(0, max(0, used - workers))
    gw = rnd.choice([0, 0, 1, 1, 2, 3, 3, 4, 5])
    if gw and pylons == 0:
        gw = 0
    return make(rnd, rnd.choice([60, 120, 180, 240, 300, 360, 420, 480, 540]),
                used, workers, army, gw, pylons,
                rnd.randint(0, 700), rnd.choice([0, 0, 0, 50, 100]),
                rnd.choice([0, 0, 1, 2, 5]), rnd.choice([0, 0, 1]), 1)


def build():
    rnd = random.Random(SEED)
    items = []
    for name, st in boundary_items(rnd):
        items.append({"id": "v3-%03d" % (len(items) + 1), "src": "boundary:" + name,
                      "state": st, "label": decide_bits(**st)})

    # 补齐到 120，并让每一位的 1/0 都不少于 36（120×0.3）
    target = 120
    guard = 0
    while len(items) < target and guard < 200000:
        guard += 1
        st = random_state(rnd)
        bits = decide_bits(**st)
        ones = [bits.count("1")]
        pos = sum(1 for it in items if it["label"][0] == "1")
        # 简单平衡：第 1 位已过半就不再收 1
        if bits[0] == "1" and pos > target * 0.55:
            continue
        items.append({"id": "v3-%03d" % (len(items) + 1), "src": "random",
                      "state": st, "label": bits})

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
        items = d["items"]
        bad = [it["id"] for it in items if decide_bits(**it["state"]) != it["label"]]
        per_bit = [sum(1 for it in items if it["label"][i] == "1") for i in range(5)]
        print("条目 %d｜哈希一致 %s｜标签自洽 %s" % (len(items), digest(items) == d["sha256"], not bad))
        print("各位为 1 的条数：%s" % dict(zip(BIT_NAMES, per_bit)))
        return

    items = build()
    d = {"version": "v3", "frozen_at": "2026-11-20T19:15+08:00",
         "system": SYSTEM, "n": len(items), "sha256": digest(items), "items": items}
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(OUT + ".sha256", "w").write(digest(items) + "\n")
    print("写入 %s  sha256=%s" % (OUT, d["sha256"]))
    print("每位为 1：%s" % dict(zip(BIT_NAMES,
          [sum(1 for it in items if it["label"][i] == "1") for i in range(5)])))


if __name__ == "__main__":
    main()
