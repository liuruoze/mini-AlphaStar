#!/usr/bin/env python3
"""从采集日志里解析胜负，输出 P 胜的 replay 路径清单。

用法:
  python experiments/replay_gen/bvb_winners.py --logdir /home/liuruoze/bvb/even_logs \
      --repldir /home/liuruoze/bvb/even --out /tmp/pwin.txt
"""
import argparse
import glob
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--repldir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reps = {}
    for p in glob.glob(os.path.join(args.repldir, "*.SC2Replay")):
        base = os.path.basename(p)
        tag = re.sub(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.SC2Replay$", "", base)
        reps[tag] = p

    rows = []
    for lg in sorted(glob.glob(os.path.join(args.logdir, "collect_*.log"))):
        tag = os.path.basename(lg)[len("collect_"):-4]
        txt = open(lg, errors="ignore").read()
        m = re.search(r"player_result = \[\((\d+), (\d+)\), \((\d+), (\d+)\)\]", txt)
        if not m:
            rows.append((tag, "NO_RESULT", ""))
            continue
        _, r1, _, r2 = map(int, m.groups())
        res = "P_WIN" if r1 == 1 else ("P_LOSE" if r1 == 2 else "R%d" % r1)
        rows.append((tag, res, reps.get(tag, "")))

    win = [r for r in rows if r[1] == "P_WIN" and r[2]]
    lose = [r for r in rows if r[1] == "P_LOSE"]
    other = [r for r in rows if r[1] not in ("P_WIN", "P_LOSE")]

    print("总对局 = %d   P胜 = %d   P负 = %d   无结果 = %d"
          % (len(rows), len(win), len(lose), len(other)))
    for tag, res, p in rows:
        print("  %-8s %-10s %s" % (tag, res, os.path.basename(p) if p else "-"))

    if args.out:
        with open(args.out, "w") as f:
            for _, _, p in win:
                f.write(p + "\n")
        print("P 胜清单 ->", args.out, "（%d 条）" % len(win))


if __name__ == "__main__":
    main()
