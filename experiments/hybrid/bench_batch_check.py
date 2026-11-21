# -*- coding: utf-8 -*-
"""复核批量接口：① 结果是否与串行一致；② 批量大小变化时耗时如何增长。

① 是前提——如果左填充把答案弄错了，0.55 秒毫无意义。
② 决定 8 局并发到底够不够。

用法：
  python experiments/hybrid/bench_batch_check.py --base-url http://127.0.0.1:8769/v1
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RULES = [
    "补水晶：没有水晶，或者人口余量（上限−已用）≤ 9 且人口上限未满 200。",
    "补农民：农民 ≤ 14 且人口余量 ≥ 1。",
    "补兵营：兵营 < 3 且至少有一个水晶（没有水晶就没有能量场，兵营放不下）。",
    "量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2。",
    "全面进攻：兵 ≥ 8。",
]
SYS = ("你是星际争霸2神族的战术指挥官，本局打狂热者快攻。现在只判断一件事：\n\n%s\n\n"
       "结合下面的局面，回答“现在是否应该执行这件事”。只输出一个字：是 或 否，不要解释。")


def post(url, obj, timeout=600):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--n-states", type=int, default=4)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    d = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
    items = d["items"][:a.n_states]
    sysmsgs = [SYS % r for r in RULES]

    msgs_batch = [[{"role": "system", "content": sm}, {"role": "user", "content": it["prompt"]}]
                  for it in items for sm in sysmsgs]

    # ① 一致性：批量 vs 串行
    t0 = time.time()
    ref = []
    for it in items:
        row = []
        for sm in sysmsgs:
            r = post(base + "/chat/completions",
                     {"messages": [{"role": "system", "content": sm},
                                   {"role": "user", "content": it["prompt"]}],
                      "max_tokens": 8, "temperature": 0})
            row.append(r["choices"][0]["message"]["content"].strip())
        ref.append(row)
    t_seq = time.time() - t0

    t0 = time.time()
    got = post(base + "/batch", {"messages_batch": msgs_batch, "max_tokens": 8, "temperature": 0})
    t_bat = time.time() - t0
    g = got["texts"]

    diff = 0
    for i in range(len(items)):
        for j in range(len(sysmsgs)):
            if g[i * len(sysmsgs) + j].strip() != ref[i][j]:
                diff += 1
    print("① 一致性：%d 条里 %d 条与串行不同（串行 %.2fs / 批量 %.2fs）"
          % (len(msgs_batch), diff, t_seq, t_bat))
    if diff:
        for i in range(len(items)):
            print("   请求:", items[i]["id"], ref[i], "→", g[i * 5:(i + 1) * 5])

    # ② 批量扩展性
    print("\n② 批量大小 vs 耗时（同为 8 token 的 是/否）")
    print("   %-8s %-10s %-12s %s" % ("批量", "耗时", "每条均摊", "每秒可处理"))
    base_msgs = msgs_batch * 4
    for n in (1, 5, 10, 20, 40):
        reps = 3
        ts = []
        for _ in range(reps):
            t0 = time.time()
            post(base + "/batch", {"messages_batch": base_msgs[:n], "max_tokens": 8,
                                   "temperature": 0})
            ts.append(time.time() - t0)
        m = statistics.median(ts)
        print("   %-8d %-10.2f %-12.3f %.1f 条/秒" % (n, m, m / n, n / m))


if __name__ == "__main__":
    main()
