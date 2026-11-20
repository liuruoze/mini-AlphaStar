#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线评估指挥官：喂历史快照，看它选的策略有没有区分度（不需要开局）。

指标：
  · 不同策略覆盖数（8 个选项里用到几个）
  · 最常见选项占比（>80% 说明基本固定策略）
  · 与规则版的一致率（规则给的是"该干什么"，LLM 给的是"选哪个"）

用法:
  python experiments/hybrid/eval_commander.py --dir <trace_dir> --n 30 [--backend openai --base-url ...]
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))

from obs_to_text import summarize_record          # noqa: E402
from commander import Commander, STRATEGIES, build_prompt  # noqa: E402
import llm_client                                  # noqa: E402
from policy import decide_from_stats               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8765/v1"))
    ap.add_argument("--model-path", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    rows = []
    for f in files:
        for r in json.load(open(f)) if False else [json.loads(l) for l in open(f, errors="ignore") if l.strip()]:
            rows.append(r)
    rnd = random.Random(a.seed)
    sample = rnd.sample(rows, min(a.n, len(rows)))

    llm = llm_client.build(a.backend, model_path=a.model_path, base_url=a.base_url)
    cmd = Commander(llm, max_new_tokens=8, temperature=0.0)

    picks, lat = Counter(), []
    agree = 0
    for r in sample:
        text = summarize_record(r)
        t0 = time.time()
        _, key, raw = cmd.propose(text)
        lat.append(time.time() - t0)
        picks[key] += 1
        st = {"t_s": r.get("t_s", 0), "used": r.get("food_used", 0), "cap": r.get("food_cap", 0),
              "minerals": r.get("minerals", 0), "vespene": r.get("vespene", 0),
              "idle": r.get("idle_workers", 0), "army": r.get("army", 0)}
        rcmd, rkey = decide_from_stats(st)
        if rkey and key and STRATEGIES.get(key) and rcmd and STRATEGIES[key][1].kind == rcmd.kind:
            agree += 1

    print("样本 %d 条，平均延迟 %.2fs" % (len(sample), sum(lat) / len(lat)))
    print("策略分布：", dict(picks))
    print("覆盖 %d/8 种；最常见占比 %.0f%%" %
          (len(picks), 100.0 * max(picks.values()) / max(sum(picks.values()), 1)))
    print("与规则版类型一致：%d/%d" % (agree, len(sample)))


if __name__ == "__main__":
    main()
