# -*- coding: utf-8 -*-
"""量三种调用方式的真实开销：串行 / 客户端并发 / 服务端批量。

目的：判断"能不能靠异步把 5 条规则问快一点"。
结论只看数字，不看推测。

用法：
  python experiments/hybrid/bench_async.py --base-url http://127.0.0.1:8769/v1
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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


def post(url, obj, timeout=300):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def one_call(base, prompt, sys_msg):
    t0 = time.time()
    post(base + "/chat/completions",
         {"messages": [{"role": "system", "content": sys_msg},
                       {"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0})
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    d = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
    prompt = d["items"][0]["prompt"]
    sysmsgs = [SYS % r for r in RULES]

    seq, conc, batch, per = [], [], [], []
    for _ in range(a.repeats):
        # ① 串行：5 次请求依次发
        t0 = time.time()
        for sm in sysmsgs:
            one_call(base, prompt, sm)
        seq.append(time.time() - t0)

        # ② 客户端并发：5 个线程同时发（服务端有全局锁）
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=5) as ex:
            list(ex.map(lambda sm: one_call(base, prompt, sm), sysmsgs))
        conc.append(time.time() - t0)

        # ③ 服务端批量：一次请求算 5 条
        t0 = time.time()
        r = post(base + "/batch", {"messages_batch": [
            [{"role": "system", "content": sm}, {"role": "user", "content": prompt}]
            for sm in sysmsgs], "max_tokens": 8, "temperature": 0})
        batch.append(time.time() - t0)
        per.append(r.get("latency_s"))

    def show(name, xs):
        print("%-14s 中位 %5.2fs ｜ 最小 %5.2fs ｜ 最大 %5.2fs"
              % (name, statistics.median(xs), min(xs), max(xs)))

    print("重复 %d 次，每次都是「5 条规则判同一个局面」\n" % a.repeats)
    show("① 串行", seq)
    show("② 客户端并发", conc)
    show("③ 服务端批量", batch)

    m_seq = statistics.median(seq)
    m_con = statistics.median(conc)
    m_bat = statistics.median(batch)
    print("\n相对串行的加速：并发 %.2fx ｜ 批量 %.2fx" % (m_seq / m_con, m_seq / m_bat))
    # 供给与容量的正确算法（早期版本印的是 8/m_bat，把"墙钟秒"当成了"游戏秒"，差 16 倍）
    qps = 5.0 / m_bat
    for interval, speed in ((8, 16), (30, 16), (60, 16)):
        print("供给 %.1f 条/秒 ｜ 间隔 %d 游戏秒、目标 %d× 时可支撑 %.1f 局"
              % (qps, interval, speed, qps * interval / (5.0 * speed)))


if __name__ == "__main__":
    main()
