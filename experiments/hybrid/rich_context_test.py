# -*- coding: utf-8 -*-
"""上下文加满后，LLM 能不能按用户的战术做决策？

对比三种输入（同一批真实局面、同一模型）：
  A 薄上下文：7 个数字（时间/人口/矿气/农民/军队/空闲/兵营）
  B 厚上下文：完整单位构成（按类型计数）+ 建筑 + 经济
  C 厚上下文 + 战术说明书（用户六条）

标准答案：doctrine.decide()（由用户 33 条标注拟合，33/33）

用法：
  python experiments/hybrid/rich_context_test.py --n 24 --backend openai \
      --base-url http://127.0.0.1:8769/v1 --model-name qwen
"""

import argparse
import collections
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

import llm_client                        # noqa: E402
from doctrine import decide              # noqa: E402
from obs_to_text import NAMES, CN        # noqa: E402

WORKER_IDS = {84: "Probe"}          # 84 = Probe（pysc2 id）
BUILDINGS = {"Nexus", "Pylon", "Gateway", "Assimilator", "Forge", "CyberneticsCore",
             "TwilightCouncil", "RoboticsFacility", "Stargate", "TemplarArchives"}

OPTS = {
    "S1": "补农民", "S2": "补水晶（人口）", "S9": "补兵营", "S3": "量产狂热者",
    "S6": "全面进攻",
}

DOCTRINE = """你是星际争霸2神族的战术指挥官，本局固定打**狂热者快攻**。你的唯一职责是节奏控制：

1. 人口余量（上限-已用）≤ 9 → 补水晶
2. 农民 ≤ 14 → 补农民
3. 兵营 < 3 → 补兵营
4. 兵营 < 4 且矿 ≥ 250 且兵 < 10 → 再补一个兵营
5. 时间 ≥ 7:00 且兵 ≥ 10 → 全面进攻
6. 其余 → 量产狂热者

只输出一个编号（S1/S2/S9/S3/S6），不要解释。"""


def unit_name(uid):
    try:
        return NAMES.get(int(uid), str(uid))
    except Exception:
        return str(uid)


def parse_own(own):
    """返回 (workers, army, buildings) 的计数字典。"""
    workers = 0
    army = 0
    buildings = {}
    for uid, cnt in (own or {}).items():
        nm = unit_name(uid)
        cnt = int(cnt)
        if nm == "Probe":
            workers += cnt
        elif nm in BUILDINGS:
            buildings[nm] = buildings.get(nm, 0) + cnt
        else:
            army += cnt
    return workers, army, buildings


def render_thin(r):
    return ("时间 %d:%02d｜人口 %d/%d｜矿 %d 气 %d\n农民 %d｜军队 %d｜空闲农民 %d｜兵营 %d 个"
            % (r["t_s"] // 60, r["t_s"] % 60, r["food_used"], r["food_cap"],
               r["minerals"], r["vespene"], r["workers"], r["army"], r["idle_workers"],
               (r.get("own") or {}).get("62", 0)))


def render_rich(r):
    wk, army, bld = parse_own(r.get("own"))
    comp = collections.Counter()
    for uid, cnt in (r.get("own") or {}).items():
        nm = unit_name(uid)
        if nm in BUILDINGS or nm == "Probe":
            continue
        comp[nm] += int(cnt)
    comp_s = "、".join("%s×%d" % (CN.get(k, k), v) for k, v in comp.most_common()) or "无"
    bld_s = "、".join("%s×%d" % (CN.get(k, k), v) for k, v in bld.items()) or "无"
    return ("时间 %d:%02d｜人口 %d/%d（余量 %d）\n"
            "矿 %d 气 %d｜农民 %d｜空闲农民 %d\n"
            "部队构成：%s（共 %d）\n"
            "建筑：%s"
            % (r["t_s"] // 60, r["t_s"] % 60, r["food_used"], r["food_cap"],
               r["food_cap"] - r["food_used"], r["minerals"], r["vespene"],
               wk, r["idle_workers"], comp_s, army, bld_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/bvb/macro_trace"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--model-name", default="qwen")
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        rows += [json.loads(l) for l in open(f, errors="ignore") if l.strip()]
    # 只取 4:00 之后（有兵营、能判断局面）
    rows = [r for r in rows if r["t_s"] >= 240]
    step = max(1, len(rows) // a.n)
    pick = rows[::step][:a.n]

    llm = llm_client.build(a.backend, base_url=a.base_url, model_name=a.model_name)
    opts = "\n".join("%s: %s" % (k, v) for k, v in OPTS.items())
    results = {}
    for name, render_fn, with_doc in [("A 薄上下文", render_thin, False),
                                      ("B 厚上下文", render_rich, False),
                                      ("C 厚+战术书", render_rich, True)]:
        ok = 0
        dist = collections.Counter()
        for r in pick:
            wk, army, bld = parse_own(r.get("own"))
            gt = decide(t=r["t_s"], used=r["food_used"], cap=r["food_cap"],
                        workers=wk, army=army, gw=bld.get("Gateway", 0),
                        minerals=r["minerals"])
            sysmsg = (DOCTRINE if with_doc else
                      "你是星际争霸2神族指挥官，只输出一个编号，不要解释。")
            raw = llm.chat([{"role": "system", "content": sysmsg},
                            {"role": "user", "content": render_fn(r) + "\n\n可选：" + opts + "\n输出："}],
                           max_new_tokens=a.max_tokens, temperature=0.0)
            k = next((x for x in OPTS if x in (raw or "")), None)
            dist[k] += 1
            ok += (k == gt)
        results[name] = (ok, dist)

    print("样本 %d 条真实局面（t≥4:00）" % len(pick))
    for name, (ok, dist) in results.items():
        print("%-12s 准确率 %2d/%d = %3.0f%%   分布 %s"
              % (name, ok, len(pick), 100.0 * ok / len(pick), dict(dist)))


if __name__ == "__main__":
    main()
