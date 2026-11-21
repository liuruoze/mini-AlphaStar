# -*- coding: utf-8 -*-
"""把 A/B 矩阵的日志汇总成 CSV。用法：python collect_ab.py /tmp/ab3"""
import glob, os, re, sys, csv

def parse(path):
    txt = open(path, errors="ignore").read()
    m = re.search(r"killed_minerals (\d+)", txt)
    s = re.search(r"statistic:.*?win_rate: ([0-9.]+).*?win_rate_custom: ([0-9.]+)"
                  r".*?food_used: ([0-9.]+).*?army_count: ([0-9.]+)", txt, re.S)
    sts = [float(x) for x in re.findall(r"steps:\s*([0-9.]+)", txt)]   # 日志里多处，取最大值
    if not (m and s):
        return None
    steps = int(max(sts)) if sts else 0
    return dict(killed=int(m.group(1)), native=float(s.group(1)), custom=float(s.group(2)),
                food=float(s.group(3)), army=float(s.group(4)), steps=steps,
                minutes=round(steps / 22.4 / 60, 1) if steps else 0)

def main(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "d*_*.log"))):
        b = os.path.basename(p)[:-4]
        mm = re.match(r"d(\d)_(sl|hyb)_s(\d+)$", b)
        if not mm:
            continue
        r = parse(p)
        if r is None:
            continue
        r.update(d=int(mm.group(1)), arm=mm.group(2), seed=int(mm.group(3)))
        rows.append(r)
    out = os.path.join(root, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "arm", "seed", "killed", "native", "custom",
                                          "food", "army", "steps", "minutes"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["d"], x["arm"], x["seed"])):
            w.writerow({k: r[k] for k in w.fieldnames})
    print("写入 %s，共 %d 局" % (out, len(rows)))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab3")
