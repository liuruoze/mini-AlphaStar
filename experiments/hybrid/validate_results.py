#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A/B 结果的完整性校验。用法：python validate_results.py /tmp/ab6 [--hybrid]

检查项（任何一项异常都会打印出来，退出码非零表示有严重问题）：

  1. 日志缺失 / 没有 statistic 行；
  2. steps 异常（< 1000，说明对局没真正跑起来）；
  3. 同一 (难度, 臂, seed) 重复；
  4. seed 覆盖有缺口（按臂分别检查）；
  5. hybrid 局：`.hybrid.jsonl` 里所有指令的 bias_n 都是 0（干预未生效）；
  6. hybrid 局：`mask_n > 0`（应为 0 —— 修复后 bias 模式不得施加遮罩）。
"""

import glob
import json
import os
import re
import sys

PAT = re.compile(r"d(\d)_(sl|hyb|doc)_s(\d+)\.log$")


def main(root):
    problems = []
    seen = {}
    seeds = {}
    logs = sorted(glob.glob(os.path.join(root, "d*_*.log")))
    if not logs:
        print("没有找到日志：%s" % root)
        return 1

    for p in logs:
        m = PAT.search(os.path.basename(p))
        if not m:
            continue
        d, arm, seed = int(m.group(1)), m.group(2), int(m.group(3))
        key = (d, arm, seed)
        if key in seen:
            problems.append("重复：d%d %s seed%d（%s 与 %s）" % (d, arm, seed, seen[key], p))
        seen[key] = p
        seeds.setdefault((d, arm), set()).add(seed)

        txt = open(p, errors="ignore").read()
        if "statistic" not in txt:
            problems.append("无结果：%s" % os.path.basename(p))
            continue
        st = [float(x) for x in re.findall(r"steps:\s*([0-9.]+)", txt)]
        steps = int(max(st)) if st else 0
        if steps < 1000:
            problems.append("步数异常：%s steps=%d" % (os.path.basename(p), steps))

        if arm == "hyb":
            jl = p[:-4] + ".hybrid.jsonl"
            if not os.path.exists(jl):
                problems.append("缺 hybrid 日志：%s" % os.path.basename(jl))
            else:
                cmds = 0
                bias_ok = 0
                mask_leak = 0
                for line in open(jl):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "cmd" not in r:
                        continue
                    # 注意：泄漏只发生在"无指令"的行上，所以不能先跳过空指令
                    if r.get("mask_n", 0) > 0:
                        mask_leak += 1
                    if r["cmd"]:
                        cmds += 1
                        if r.get("bias_n", 0) > 0:
                            bias_ok += 1
                if cmds and bias_ok == 0:
                    problems.append("干预未生效：%s（%d 条指令全为空偏置）" % (os.path.basename(jl), cmds))
                if mask_leak:
                    problems.append("遮罩泄漏：%s（%d 次 mask_n>0）" % (os.path.basename(jl), mask_leak))

    print("扫描 %d 个日志" % len(logs))
    for (d, arm), ss in sorted(seeds.items()):
        gaps = sorted(set(range(min(ss), max(ss) + 1)) - ss)
        print("  d%-2d %-4s n=%3d ｜ seed %d–%d%s" % (
            d, arm, len(ss), min(ss), max(ss),
            ("｜缺口 " + str(gaps[:6]) + ("…" if len(gaps) > 6 else "")) if gaps else ""))

    if problems:
        print("\n发现 %d 个问题：" % len(problems))
        for x in problems[:40]:
            print("  - " + x)
        return 1
    print("\n未发现异常")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab6"))
