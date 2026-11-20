# -*- coding: utf-8 -*-
"""生成并冻结评测题集（v2 协议，120 题，5 类各 24）。

设计要点：
  · 每类 24 条，保证"恒定答案"只能拿 20%；
  · 重点覆盖阈值边界：农民 14/15、兵营 2/3、兵 7/8、人口余量 9/10；
  · 每条题目**预先渲染成最终 user 文本**并存盘 —— 所有模型看到逐字相同的输入；
  · 标签由 doctrine.decide 计算（v2 六条规则），不是人工写的；
  · 生成一次后写 sha256，之后不得修改（改 = v3）。

用法：
  python experiments/hybrid/make_eval_set.py            # 生成 eval_set_v2.json + .sha256
  python experiments/hybrid/make_eval_set.py --check    # 只校验哈希与标签自洽
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

from doctrine import decide, CODE_OF, NAMES  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set_v2.json")
SEED = 20261120

# 系统提示（与 prompt_spec_v2.md 逐字一致）
SYSTEM = """你是星际争霸2神族的战术指挥官。本局固定执行狂热者快攻，你的唯一职责是节奏控制。

注意：兵营必须建在水晶的能量场里，所以没有任何水晶时，一切其他建造都无从谈起。

1. 水晶 = 0 → 补水晶
2. 人口余量（上限−已用）≤ 9 → 补水晶
3. 农民 ≤ 14 → 补农民
4. 兵营 < 3 → 补兵营
5. 兵 ≥ 8 → 全面进攻
6. 其余 → 量产狂热者

只输出一个 5 位编码（由 0 和 1 组成），不要解释。"""

OPTIONS = [("00001", "补水晶"), ("00010", "补农民"), ("00100", "补兵营"),
           ("01000", "量产狂热者"), ("10000", "全面进攻")]


def render(st):
    """按规范 v2 的 user 模板渲染。"""
    comp = []
    if st["zealots"]:
        comp.append("狂热者×%d" % st["zealots"])
    if st["stalkers"]:
        comp.append("使徒×%d" % st["stalkers"])
    comp_s = "、".join(comp) if comp else "无"
    bld = []
    if st["nexus"]:
        bld.append("基地×%d" % st["nexus"])
    if st["pylons"]:
        bld.append("水晶×%d" % st["pylons"])
    if st["gw"]:
        bld.append("兵营×%d" % st["gw"])
    if st["cyber"]:
        bld.append("控制核心×%d" % st["cyber"])
    bld_s = "、".join(bld) if bld else "无"
    opts = "\n".join("%s: %s" % (c, n) for c, n in OPTIONS)
    return ("时间 %d:%02d｜人口 %d/%d（余量 %d）\n"
            "矿 %d 气 %d｜农民 %d｜空闲农民 %d\n"
            "部队构成：%s\n"
            "建筑：%s\n\n可选：\n%s\n输出："
            % (st["t"] // 60, st["t"] % 60, st["used"], st["cap"], st["cap"] - st["used"],
               st["minerals"], st["gas"], st["workers"], st["idle"],
               comp_s, bld_s, opts))


def base_state(rnd, t, used, cap, workers, army, gw, pylons, cyber=0, nexus=1,
               minerals=None, gas=0, idle=0):
    zealots = army
    stalkers = 0
    if army and rnd.random() < 0.2:          # 少量局面带使徒，避免"兵=狂热者"太单一
        stalkers = rnd.randint(1, max(1, army // 3))
        zealots = army - stalkers
    return dict(t=t, used=used, cap=cap, workers=workers, army=army,
                zealots=zealots, stalkers=stalkers, gw=gw, pylons=pylons,
                cyber=cyber, nexus=nexus, idle=idle,
                minerals=rnd.randint(0, 600) if minerals is None else minerals,
                gas=rnd.choice([0, 0, 0, 50, 100]) if gas == 0 else gas)


def gen_pylon(rnd):
    """补水晶：一半是"没有水晶"的前置局面，一半是人口余量 ≤9。"""
    if rnd.random() < 0.25:
        # 没有水晶（开局或被打光）
        cap = rnd.choice([15, 15, 23])
        used = rnd.randint(max(1, cap - 6), cap)
        workers = rnd.randint(max(1, used - 2), used)
        st = base_state(rnd, rnd.choice([30, 45, 60, 90]), used, cap, workers,
                        army=0, gw=0, pylons=0, minerals=rnd.randint(50, 150))
    else:
        cap = rnd.choice([23, 31, 47, 63, 70, 80])
        gap = rnd.choice([0, 1, 3, 5, 7, 8, 9])       # 卡在 9 附近
        used = cap - gap
        workers = rnd.randint(max(12, used - 20), max(13, used - 6))
        army = rnd.randint(0, max(0, used - workers))
        gw = rnd.choice([0, 1, 2, 3, 4])
        st = base_state(rnd, rnd.choice([120, 180, 240, 300, 360]),
                        used, cap, workers, army, gw,
                        pylons=rnd.randint(1, 5), cyber=rnd.choice([0, 1]))
    return st


def gen_probe(rnd):
    """补农民：农民 ≤14，其余条件不满足（余量>9）。"""
    cap = rnd.choice([47, 63, 70, 80])
    used = rnd.randint(max(15, cap - 40), cap - 10)
    workers = rnd.choice([8, 10, 11, 12, 13, 14])     # 卡在 14
    army = rnd.randint(0, max(0, used - workers))
    gw = rnd.choice([0, 1, 2, 3, 4])
    return base_state(rnd, rnd.choice([120, 180, 240, 300, 360]),
                      used, cap, workers, army, gw,
                      pylons=rnd.randint(1, 5), cyber=rnd.choice([0, 1]))


def gen_gateway(rnd):
    """补兵营：农民 >14、余量 >9、兵营 <3。"""
    cap = rnd.choice([47, 63, 70, 80])
    used = rnd.randint(max(20, cap - 40), cap - 10)
    workers = rnd.randint(15, 30)
    army = rnd.randint(0, max(0, used - workers))
    gw = rnd.choice([0, 1, 2])                        # 卡在 2
    return base_state(rnd, rnd.choice([150, 210, 270, 330, 400]),
                      used, cap, workers, army, gw,
                      pylons=rnd.randint(1, 5), cyber=rnd.choice([0, 1]))


def gen_zealot(rnd):
    """量产：农民 >14、余量 >9、兵营 ≥3、兵 <8。"""
    cap = rnd.choice([47, 63, 70, 80])
    used = rnd.randint(max(25, cap - 40), cap - 10)
    workers = rnd.randint(15, 30)
    army = rnd.choice([0, 1, 2, 3, 4, 5, 6, 7])       # 卡在 7
    gw = rnd.choice([3, 3, 4, 5])                     # 卡在 3
    return base_state(rnd, rnd.choice([240, 300, 360, 420, 480]),
                      used, cap, workers, army, gw,
                      pylons=rnd.randint(2, 7), cyber=rnd.choice([0, 1]))


def gen_attack(rnd):
    """进攻：兵 ≥8，其余条件不满足。"""
    cap = rnd.choice([47, 63, 70, 80])
    used = rnd.randint(max(30, cap - 35), cap - 10)
    workers = rnd.randint(15, 30)
    army = rnd.choice([8, 8, 9, 10, 12, 14, 16, 18, 20])
    gw = rnd.choice([3, 4, 5])
    st = base_state(rnd, rnd.choice([300, 360, 420, 480, 540]),
                    used, cap, workers, army, gw,
                    pylons=rnd.randint(2, 8), cyber=rnd.choice([0, 1]))
    return st


GENS = [gen_pylon, gen_probe, gen_gateway, gen_zealot, gen_attack]


def build():
    rnd = random.Random(SEED)
    items = []
    for gi, gen in enumerate(GENS):
        made = 0
        guard = 0
        while made < 24 and guard < 2000:
            guard += 1
            st = gen(rnd)
            # 物理约束：单位数不超过人口；农民不超过人口
            if st["workers"] + st["army"] > st["used"]:
                continue
            if st["used"] > st["cap"] or st["used"] < 1:
                continue
            sid = decide(**st)
            want = ["S2", "S1", "S9", "S3", "S6"][gi]
            if sid != want:                     # 只收该类，保证类别纯净
                continue
            items.append({
                "id": "v2-%03d" % (len(items) + 1),
                "group": want,
                "state": st,
                "label": sid,
                "code": CODE_OF[sid],
                "label_name": NAMES[sid],
                "prompt": render(st),
            })
            made += 1
    return items


def digest(items):
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.check:
        data = json.load(open(OUT, encoding="utf-8"))
        items = data["items"]
        h = digest(items)
        ok = (h == data["sha256"])
        bad = [it["id"] for it in items if decide(**it["state"]) != it["label"]]
        dist = collections.Counter(it["group"] for it in items)
        print("条目 %d｜哈希一致 %s｜标签自洽 %s｜分布 %s"
              % (len(items), ok, not bad, dict(dist)))
        return

    items = build()
    data = {"version": "v2", "frozen_at": "2026-11-20T19:08+08:00",
            "system": SYSTEM, "n": len(items), "sha256": digest(items), "items": items}
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(OUT + ".sha256", "w").write(digest(items) + "\n")
    print("写入 %s" % OUT)
    print("sha256 %s" % data["sha256"])
    print("分布 %s" % dict(collections.Counter(it["group"] for it in items)))


if __name__ == "__main__":
    main()
