#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 hybrid 日志的 `why` 字段还原建造时序。

`why` 的格式（commander_v3 / commander_doctrine 都写）：
    v3:pylon|w12 a0 g0 py0 cap15/15 min50
    doc:gateway|w25 a2 g1 py4

由此可以算出：第一个水晶、第一个兵营、军队首次 ≥8 的时间点。
这些量比"最终击杀数"更贴近"指挥官有没有真的改变执行"。

用法：
    python build_timeline.py /tmp/doc30            # 单目录
    python build_timeline.py /tmp/doc30 /tmp/ab6   # 多目录合并
"""

import glob
import json
import os
import re
import statistics as st
import sys

PAT = re.compile(r"(\w+):(\w+)\|w(\d+) a(\d+) g(\d+) py(\d+)")


def timeline(path):
    """返回 (首次 py>=1, 首次 g>=1, 首次 a>=8) 的游戏内秒数；没有则 None。"""
    first = {"pylon": None, "gateway": None, "army8": None}
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        why = r.get("why") or ""
        m = PAT.search(why)
        if not m:
            continue
        t = r.get("t", 0)
        _, _, _, a, g, py = m.groups()
        if first["pylon"] is None and int(py) >= 1:
            first["pylon"] = t
        if first["gateway"] is None and int(g) >= 1:
            first["gateway"] = t
        if first["army8"] is None and int(a) >= 8:
            first["army8"] = t
    return first


def summarize(root, label):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "d*_*_s*.hybrid.jsonl"))):
        b = os.path.basename(p)
        m = re.match(r"d(\d)_(\w+)_s(\d+)\.hybrid\.jsonl$", b)
        if not m:
            continue
        f = timeline(p)
        rows.append((int(m.group(1)), m.group(2), int(m.group(3)), f))
    if not rows:
        print("%s：无日志" % label)
        return
    print("== %s（%d 局）" % (label, len(rows)))
    for key, name in (("pylon", "首水晶"), ("gateway", "首兵营"), ("army8", "8 兵")):
        vals = [r[3][key] for r in rows if r[3][key] is not None]
        if vals:
            print("   %s：达成 %d/%d ｜ 中位 %.1f 分（最快 %.1f / 最慢 %.1f）" % (
                name, len(vals), len(rows), st.median(vals) / 60,
                min(vals) / 60, max(vals) / 60))
        else:
            print("   %s：0/%d 达成" % (name, len(rows)))


def main(dirs):
    for d in dirs:
        summarize(d, d)


if __name__ == "__main__":
    main(sys.argv[1:] or ["/tmp/doc30"])
