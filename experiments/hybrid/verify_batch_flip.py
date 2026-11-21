# -*- coding: utf-8 -*-
"""复查 batch 与串行的差异。HY 2026-11-21 审阅后要求补三处：
  ① 标注"批内自洽"不代表真实批量组成；
  ② 边界样本补齐（兵 23/24、矿 99/100、兵营 2/3），去掉早退；
  ③ 把 n=1..5 的震荡实验并入本脚本留档。

用法：
  python experiments/hybrid/verify_batch_flip.py [--base-url ...] [--max-items 16]

三条判据全过才算干净：
  A. 真实批量组成（5 条不同规则）下，串行与批量逐位一致；
  B. 边界样本上也不翻转；
  C. 同一问题在不同批量大小下答案不变。
"""

import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

RULES = {
    "补水晶": "补水晶：没有水晶，或者人口余量（上限−已用）≤ 9 且人口上限未满 200。",
    "补农民": "补农民：农民 ≤ 14 且人口余量 ≥ 1。",
    "补兵营": "补兵营：兵营 < 3 且至少有一个水晶（没有水晶就没有能量场，兵营放不下）。",
    "量产": "量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2。",
    "进攻": "全面进攻：兵 ≥ 8。",
}
KEYS = list(RULES)
SYS = ("你是星际争霸2神族的战术指挥官，本局打狂热者快攻。现在只判断一件事：\n\n%s\n\n"
       "结合下面的局面，回答“现在是否应该执行这件事”。只输出一个字：是 或 否，不要解释。")

# 各规则阈值（与 doctrine.py 同一组常数），用来挑边界样本
THRESH = {
    "workers": (14, 15),      # 补农民：≤14
    "gap": (9, 10),           # 补水晶：余量 ≤9
    "army_attack": (7, 8),    # 进攻：≥8
    "army_cap": (23, 24),     # 量产：兵 <24
    "minerals": (99, 100),    # 量产：矿 ≥100
    "gw": (2, 3),             # 补兵营：<3
}


def post(url, obj, timeout=600):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def serial(base, sysmsg, prompt):
    r = post(base + "/chat/completions",
             {"messages": [{"role": "system", "content": sysmsg},
                           {"role": "user", "content": prompt}],
              "max_tokens": 8, "temperature": 0})
    return r["choices"][0]["message"]["content"].strip()


def batch(base, sysmsgs, prompt):
    r = post(base + "/batch", {"messages_batch": [
        [{"role": "system", "content": sm}, {"role": "user", "content": prompt}]
        for sm in sysmsgs], "max_tokens": 8, "temperature": 0})
    return [t.strip() for t in r["texts"]]


def is_boundary(s):
    gap = s["cap"] - s["used"]
    return (s["workers"] in THRESH["workers"] or gap in THRESH["gap"]
            or s["army"] in THRESH["army_attack"] or s["army"] in THRESH["army_cap"]
            or s["minerals"] in THRESH["minerals"] or s["gw"] in THRESH["gw"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--max-items", type=int, default=16)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    d = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
    items = d["items"]
    by_id = {it["id"]: it for it in items}

    # ① 批内自洽性（只说明"同一批量下稳定"，不代表真实组成）
    it = by_id["v3-001"]
    sm = SYS % RULES["补农民"]
    seq5 = [serial(base, sm, it["prompt"]) for _ in range(5)]
    bat5 = [batch(base, [sm] * 5, it["prompt"]) for _ in range(5)]
    print("① 批内自洽性（同一条重复 5 份；**不代表真实批量组成**）")
    print("   v3-001 补农民（农民=14）：串行 %s ｜ 批量 %s ｜ 批内自洽 %s"
          % (set(seq5), {b[0] for b in bat5}, all(len(set(b)) == 1 for b in bat5)))

    # ② 真实批量组成（5 条不同规则），边界样本全覆盖
    print("\n② 真实批量组成（5 条不同规则），边界样本全覆盖")
    bnd = [it2 for it2 in items if is_boundary(it2["state"])][:a.max_items]
    sysmsgs = [SYS % RULES[k] for k in KEYS]
    total = diff = 0
    flips = []
    for it2 in bnd:
        ref = [serial(base, sm2, it2["prompt"]) for sm2 in sysmsgs]
        got = batch(base, sysmsgs, it2["prompt"])
        bad = [KEYS[i] for i in range(5) if ref[i] != got[i]]
        total += 5
        diff += len(bad)
        if bad:
            flips.append((it2["id"], bad, ref, got))
    print("   边界样本 %d 个，共 %d 条，不一致 %d 条" % (len(bnd), total, diff))
    for fid, bad, ref, got in flips:
        print("   %s 翻转 %s：%s" % (fid, bad, ", ".join(
            "%s 串行=%s 批量=%s" % (k, ref[KEYS.index(k)], got[KEYS.index(k)]) for k in bad)))

    # ③ 批量大小震荡（不同规则优先填充，贴近真实组成）
    print("\n③ 同一问题在不同批量大小下的答案（n=1..5）")
    probe = by_id["v3-001"]
    probe_sys = SYS % RULES["补农民"]
    pad = [SYS % RULES[k] for k in ("进攻", "补兵营", "量产", "补水晶")]
    for n in range(1, 6):
        if n == 1:
            ans = set(batch(base, [probe_sys], probe["prompt"]))
        else:
            ans = {batch(base, [probe_sys] + pad[:n - 1], probe["prompt"])[0]}
        print("   n=%d -> %s" % (n, ans))


if __name__ == "__main__":
    main()
