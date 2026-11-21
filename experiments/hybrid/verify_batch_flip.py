# -*- coding: utf-8 -*-
"""复核 HY 的核心指控：batch 与串行是否在边界局面上给出不同答案。

HY 报告：v3-001 的「补农民」串行答"是"、batch 答"否"，30 条里 1 条翻转。

本脚本做三件事：
  ① 把 v3-001 的补农民问 5 次串行、5 次批量，看是否稳定翻转；
  ② 扫所有边界局面（农民=14/15、余量=9/10、兵=24 等），串行 vs 批量逐条比；
  ③ 报告 batch 内部是否自身一致（同一批里重复同一条 3 次，答案是否相同）。

用法：
  python experiments/hybrid/verify_batch_flip.py --base-url http://127.0.0.1:8769/v1
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

RULES = {
    "补水晶": "补水晶：没有水晶，或者人口余量（上限−已用）≤ 9 且人口上限未满 200。",
    "补农民": "补农民：农民 ≤ 14 且人口余量 ≥ 1。",
    "补兵营": "补兵营：兵营 < 3 且至少有一个水晶（没有水晶就没有能量场，兵营放不下）。",
    "量产": "量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2。",
    "进攻": "全面进攻：兵 ≥ 8。",
}
SYS = ("你是星际争霸2神族的战术指挥官，本局打狂热者快攻。现在只判断一件事：\n\n%s\n\n"
       "结合下面的局面，回答“现在是否应该执行这件事”。只输出一个字：是 或 否，不要解释。")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    d = json.load(open(os.path.join(HERE, "eval_set_v3.json"), encoding="utf-8"))
    items = {it["id"]: it for it in d["items"]}

    # ---------- ① 复现 HY 指出的那一条 ----------
    it = items["v3-001"]
    sm = SYS % RULES["补农民"]
    print("① v3-001 的「补农民」，真值 bit=%s" % it["label"][1])
    print("   局面：农民 %d，人口 %d/%d，真值 %s"
          % (it["state"]["workers"], it["state"]["used"], it["state"]["cap"], it["label"]))
    seq = [serial(base, sm, it["prompt"]) for _ in range(5)]
    bat = [batch(base, [sm] * 5, it["prompt"]) for _ in range(5)]
    print("   串行 ×5: %s" % seq)
    print("   批量 ×5（每次批里同一条重复 5 份）: %s" % [b[0] for b in bat])
    print("   批内是否自洽: %s" % all(len(set(b)) == 1 for b in bat))

    # ---------- ② 扫边界局面 ----------
    print("\n② 边界局面：串行 vs 批量（5 条规则全比）")
    ids = ["v3-001", "v3-002", "v3-003", "v3-004", "v3-005", "v3-006"]
    # 再补几个已知临界局面
    for iid, it2 in items.items():
        s = it2["state"]
        if s["workers"] in (14, 15) or (s["cap"] - s["used"]) in (9, 10) or s["army"] in (7, 8):
            if iid not in ids:
                ids.append(iid)
        if len(ids) >= 12:
            break

    sysmsgs = [SYS % RULES[k] for k in RULES]
    keys = list(RULES)
    total = diff = 0
    for iid in ids:
        it2 = items[iid]
        ref = [serial(base, sm, it2["prompt"]) for sm in sysmsgs]
        got = batch(base, sysmsgs, it2["prompt"])
        bad = [keys[i] for i in range(5) if ref[i] != got[i]]
        total += 5
        diff += len(bad)
        if bad:
            print("   %s 翻转：%s" % (iid, ", ".join(
                "%s 串行=%s 批量=%s" % (k, ref[keys.index(k)], got[keys.index(k)]) for k in bad)))
    print("   合计 %d 条，不一致 %d 条" % (total, diff))


if __name__ == "__main__":
    main()
