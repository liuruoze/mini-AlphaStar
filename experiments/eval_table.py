#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测结果汇总表（替代手写 sed/awk 拼装）。

用法：
  python3 experiments/eval_table.py cwB_ep4 cwH5_s20261171
  python3 experiments/eval_table.py --seeds 3038-3047 cwB_ep4
  python3 experiments/eval_table.py --csv cwB_ep4
  python3 experiments/eval_table.py --detail cwH5_s20261171

口径（与项目永久裁定一致）：
  * native 真胜 = raw_reward == +1
  * 官方判胜   = outcome == +1  或 (outcome == 0 且 killed_points > 4000)
  两个数必须分开报，不可混用。

文件命名兼容两种：`{tag}s{seed}_{seed}.txt`（run_p8_eval）与 `{tag}_{seed}.txt`（直接调 run_eval_batch）。
"""
import argparse
import glob
import os
import re
import statistics as st
import sys

OUT_DIR = os.path.expanduser("~/mini-AlphaStar-off/outputs")
PAT = re.compile(
    r"outcome:\s*([+-]?\d+).*?raw_reward:\s*([+-]?\d+).*?food_used:\s*([\d.]+).*?"
    r"army_count:\s*([\d.]+).*?killed_points:\s*([\d.]+).*?our_units:\s*(\d+).*?steps:\s*([\d.]+)"
)


def find_file(tag, seed):
    for name in ("%ss%d_%d.txt" % (tag, seed, seed), "%s_%d.txt" % (tag, seed)):
        p = os.path.join(OUT_DIR, name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def parse(path):
    with open(path, "r", errors="ignore") as f:
        head = f.readline()
    m = PAT.search(head)
    if not m:
        return None
    o, r, food, army, killed, our, steps = m.groups()
    return {
        "outcome": int(o), "raw": int(r), "food": float(food), "army": float(army),
        "killed": float(killed), "our": int(our), "steps": float(steps),
    }


def official_win(d):
    return d["outcome"] == 1 or (d["outcome"] == 0 and d["killed"] > 4000)


def collect(tag, seeds):
    rows = []
    for s in seeds:
        p = find_file(tag, s)
        if not p:
            continue
        d = parse(p)
        if d is None:
            continue
        d["seed"] = s
        rows.append(d)
    return rows


def collect_merged(tags, seeds):
    """把同一模型的多个 tag 别名按 seed 合并（同一 seed 只取第一个有结果的别名）。

    场景：同一个模型在多次评测里用过不同 OFF_TAG（如 cwH5_s20261171_e / h5v / h5v3），
    结果分散在多个文件里，必须合并后才能得到完整 n。
    """
    seen = {}
    cover = {t: 0 for t in tags}
    for t in tags:
        for s in seeds:
            if s in seen:
                continue
            p = find_file(t, s)
            if not p:
                continue
            d = parse(p)
            if d is None:
                continue
            d["seed"] = s
            seen[s] = d
            cover[t] += 1
    return [seen[s] for s in sorted(seen)], cover


def suggest_tags(tag):
    """找不到输出时，从 outputs 里找相近的 tag 前缀（今天多次因为别名对不上而白跑）。"""
    try:
        names = os.listdir(OUT_DIR)
    except Exception:
        return []
    stems = set()
    for nm in names:
        if not nm.endswith(".txt"):
            continue
        m = re.match(r"(.+?)s(\d+)_\1?\.txt$", nm) or re.match(r"(.+?)_(\d+)\.txt$", nm)
        if m:
            stems.add(m.group(1))
    stem = tag.split("_s")[0] if "_s" in tag else tag
    out = sorted(x for x in stems if tag in x or (len(stem) > 5 and stem in x))
    return out[:5]


def fmt(tag, rows, seeds, detail=False, csv=False):
    n = len(rows)
    if n == 0:
        sug = suggest_tags(tag)
        print("%-24s 0/%d  (无输出文件)%s" % (tag, len(seeds),
              ("  可能是别名: " + " ".join(sug)) if sug else ""))
        return
    w = sum(1 for d in rows if d["raw"] == 1)
    ow = sum(1 for d in rows if official_win(d))
    km = st.median([d["killed"] for d in rows])
    sm = st.median([d["steps"] for d in rows])
    miss = len(seeds) - n
    if csv:
        print("%s,%d,%d,%d,%.0f,%.0f,%d" % (tag, n, w, ow, km, sm, miss))
        return
    print("%-24s n=%-3d native=%2d/%-3d (%4.1f%%)  官方=%2d/%-3d  killed中位=%-7.0f steps中位=%-7.0f%s"
          % (tag, n, w, n, 100.0 * w / n, ow, n, km, sm,
             ("  缺 %d 局" % miss) if miss else ""))
    if detail:
        for d in sorted(rows, key=lambda x: x["seed"]):
            print("    s%-5d raw=%+d outcome=%+d killed=%-7.0f army=%-5.0f food=%-6.1f our=%-3d steps=%.0f"
                  % (d["seed"], d["raw"], d["outcome"], d["killed"], d["army"], d["food"], d["our"], d["steps"]))


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", help="模型 tag（可多个）；用 --compare-pairs 时可省略")
    ap.add_argument("--seeds", default="3038-3077", help="种子范围，如 3038-3077 或 3038,3040")
    ap.add_argument("--detail", action="store_true", help="打印逐局明细")
    ap.add_argument("--csv", action="store_true", help="CSV 输出")
    ap.add_argument("--glob", action="store_true", help="把 tag 当通配符展开")
    ap.add_argument("--merge", action="store_true",
                    help="把多个 tag 当同一模型的别名，按 seed 合并成一行")
    ap.add_argument("--compare", action="store_true",
                    help="两两配对比较（同一批种子上谁赢），tag 数量须为 2 的倍数")
    ap.add_argument("--compare-pairs", default="",
                    help="显式指定配对，如 'A1,A2|B1,B2'（每组内逗号分隔为别名）")
    args = ap.parse_args()

    if not args.tags and not args.compare_pairs:
        raise SystemExit("至少要给一个 tag，或用 --compare-pairs")
    seeds = parse_seeds(args.seeds)
    tags = []
    for t in args.tags:
        if args.glob:
            # 从 outputs 里反推有哪些 tag
            found = set()
            for p in glob.glob(os.path.join(OUT_DIR, "*_%d.txt" % seeds[0])):
                base = os.path.basename(p)
                m = re.match(r"(.+?)s?%d_%d\.txt$" % (seeds[0], seeds[0]), base)
                if m:
                    found.add(m.group(1))
            tags.extend(sorted(x for x in found if re.search(t, x)))
        else:
            tags.append(t)
    if args.csv:
        print("tag,n,native_wins,official_wins,killed_median,steps_median,missing")
    if args.compare or args.compare_pairs:
        pairs = []
        if args.compare_pairs:
            for grp in args.compare_pairs.split("|"):
                pairs.append([x.strip() for x in grp.split(",") if x.strip()])
        else:
            if len(tags) % 2 != 0:
                raise SystemExit("--compare 需要偶数个 tag（成对），或用 --compare-pairs")
            pairs = [[tags[i], tags[i + 1]] for i in range(0, len(tags), 2)]
        for gi in range(0, len(pairs), 2):
            ga = pairs[gi]
            gb = pairs[gi + 1] if gi + 1 < len(pairs) else None
            if gb is None:
                raise SystemExit("--compare 需要成对的组")
            ra, _ = collect_merged(ga, seeds)
            rb, _ = collect_merged(gb, seeds)
            da = {d["seed"]: d for d in ra}
            db = {d["seed"]: d for d in rb}
            common = sorted(set(da) & set(db))
            wa = sum(1 for s in common if da[s]["raw"] == 1)
            wb = sum(1 for s in common if db[s]["raw"] == 1)
            both = sum(1 for s in common if da[s]["raw"] == 1 and db[s]["raw"] == 1)
            flip_ab = [s for s in common if da[s]["raw"] == 1 and db[s]["raw"] != 1]
            flip_ba = [s for s in common if db[s]["raw"] == 1 and da[s]["raw"] != 1]
            print("A=%-28s vs B=%-28s  共同种子=%d" % ("+".join(ga)[:28], "+".join(gb)[:28], len(common)))
            print("  A胜=%-3d B胜=%-3d 双方都胜=%-3d  A独胜=%-3d B独胜=%-3d"
                  % (wa, wb, both, len(flip_ab), len(flip_ba)))
            if flip_ab:
                print("  A 独胜种子: %s" % " ".join(str(x) for x in flip_ab[:20]))
            if flip_ba:
                print("  B 独胜种子: %s" % " ".join(str(x) for x in flip_ba[:20]))
            only = len(common) - both - len(flip_ab) - len(flip_ba)
            print("  双方都负=%-3d" % only)
        return 0

    if args.merge and len(tags) >= 1:
        rows, cover = collect_merged(tags, seeds)
        fmt("+".join(tags)[:40], rows, seeds, detail=args.detail, csv=args.csv)
        print("    别名覆盖: " + "  ".join("%s=%d" % (k, v) for k, v in cover.items()))
        return 0
    for t in tags:
        rows = collect(t, seeds)
        fmt(t, rows, seeds, detail=args.detail, csv=args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
