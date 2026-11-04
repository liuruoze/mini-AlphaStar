#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长跑曲线分析器（第 57 轮，DeepSeek）。

输入 `_soak_monitor.py` 产出的 CSV，输出**可以直接写进结论**的数字：

1. 每个数值列的 min / max / 末段均值；
2. **前 20% 均值 vs 后 20% 均值**的差 —— 判断"单调上升"用这个，不用看最大最小值
   （最大值可能只是启动瞬间的一次性峰值，容易被误判成泄漏）；
3. 线性斜率（用最小二乘，单位 /min）—— 泄漏是**线性**的，噪声是**随机**的，
   斜率是最能区分两者的统计量；
4. batch 侧指标从 run.log 里抽（平均 batch、峰值 batch、rounds、稳态 step/s）。

用法：
    python _soak_analyze.py log/soak_batch30/monitor.csv [log/soak_batch30/run.log]
"""

import re
import sys


def read_csv(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return [], []
    header = lines[0].split(",")
    rows = []
    for l in lines[1:]:
        parts = l.split(",")
        if len(parts) != len(header):
            continue
        rows.append(parts)
    return header, rows


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def slope_per_min(vals, ts):
    """最小二乘斜率（单位：列值/min），ts 为秒。"""
    n = len(vals)
    if n < 3:
        return 0.0
    mx = sum(ts) / n
    my = sum(vals) / n
    num = sum((t - mx) * (v - my) for t, v in zip(ts, vals))
    den = sum((t - mx) ** 2 for t in ts)
    if den == 0:
        return 0.0
    return num / den * 60.0


def median5(vals):
    """5 点中值滤波。长跑曲线里单点抖动（扫 /proc 时进程刚起/刚退）很常见，
    直接看 min/max 会把抖动当成泄漏；中值滤波对"平台 + 尖刺"最稳。"""
    out = []
    n = len(vals)
    for i in range(n):
        w = vals[max(0, i - 2):i + 3]
        out.append(sorted(w)[len(w) // 2])
    return out


def analyze_csv(path):
    header, rows = read_csv(path)
    if not rows:
        print("(csv 为空)")
        return
    ts = [to_float(r[1]) for r in rows]
    total = ts[-1] or 1.0
    print("样本数 %d，时长 %.0f s（%.1f min）" % (len(rows), total, total / 60.0))
    print("（5 点中值滤波后按时间分位取点；\n      斜率只取 20%~90% 时间窗：前 10% 是启动预热，后 10% 是收尾）")
    print("%-13s %8s %8s %8s %8s %8s %8s %10s"
          % ("列", "10%", "25%", "50%", "75%", "90%", "末样本", "20-90%斜率/min"))
    for ci in range(2, len(header)):
        vals, tv = [], []
        for i, r in enumerate(rows):
            v = to_float(r[ci])
            if v is None or v < 0:
                continue
            vals.append(v)
            tv.append(ts[i])
        if not vals:
            continue
        sm = median5(vals)
        marks = []
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            target = frac * total
            j = min(range(len(tv)), key=lambda k: abs(tv[k] - target))
            marks.append(sm[j])
        # 斜率只取 20%~90% 时间窗。
        # ⚠️ 第 70 轮修正：原来用"后半段"（50%~100%），但最后几个样本是**收尾瞬间**——
        #    进程退出、显存/RSS 断崖下跌，会把斜率拉成负数。实测本轮 py_rss 的斜率
        #    算出 -574 MB/min，而实际 10%->90% 是 +134 MB。把收尾算进"稳态趋势"，
        #    既会掩盖真泄漏，也会伪造"内存在下降"。
        _lo, _hi = 0.2 * total, 0.9 * total
        _win = [k for k in range(len(tv)) if _lo <= tv[k] <= _hi]
        if len(_win) >= 4:
            sl = slope_per_min([sm[k] for k in _win], [tv[k] for k in _win])
        else:
            sl = 0.0
        print("%-13s %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %+10.2f"
              % (header[ci], marks[0], marks[1], marks[2], marks[3], marks[4], marks[5], sl))


# 新格式（第 55 轮起，带分段计时）
BATCH_RE = re.compile(
    r"\[async\] rounds=(\d+) steps=(\d+) \| 平均 batch ([\d.]+)（峰值 (\d+) / 上限 (\d+)）"
    r"\| 每轮 sync ([\d.]+) \+ build ([\d.]+) \+ forward ([\d.]+) \+ write ([\d.]+) = ([\d.]+) ms"
    r"\s*\|\s*稳态 ([\d.]+) step/s")
# 旧格式（第 55 轮之前），保留以便分析历史日志
BATCH_RE_OLD = re.compile(
    r"\[async\] rounds=(\d+) steps=(\d+) \| 平均 batch ([\d.]+)（峰值 (\d+) / 上限 (\d+)）"
    r"\| forward 每轮 ([\d.]+) ms \| 稳态 ([\d.]+) step/s")

# 第 71 轮：write 段拆分（追加在"稳态"之后，不影响上面的匹配）
WRITE_SPLIT_RE = re.compile(r"回写拆分 h/c ([\d.]+) \+ 动作 ([\d.]+) ms")
# 第 70 轮：episode 边界契约（bad 必须为 0）
H0_RE = re.compile(r"新episode首步h=0: ok (\d+) / bad (\d+)")

STEP_RE = re.compile(r"\[(agent_\d+)\] steps=(\d+) \| 总耗时 ([\d.]+) s \| 吞吐 ([\d.]+) step/s（mode=(\w+)）")


def classify_tracebacks(txt):
    """把 Traceback 分类。

    第 71 轮新增。动机：之前所有"稳定性判据"里都写着 `grep -c Traceback == 0`，
    但实测**每一次跑完的日志里都有 1 个** —— 它来自 tensorboardX 的
    event_file_writer 守护线程（进程退出时队列被关，`_recv` 抛 EOFError），
    不是训练代码的异常。之所以以前测到 0，是因为 grep 发生在**跑完之前**，
    收尾那几行还没写进日志。

    返回 (总数, tensorboardX 收尾 EOFError 数, 其它异常的首行列表)。
    """
    chunks = txt.split("Traceback (most recent call last):")
    total = len(chunks) - 1
    tb_writer = 0
    others = []
    for c in chunks[1:]:
        seg = c[:1500]
        if ("tensorboardX" in seg or "event_file_writer" in seg) and "EOFError" in seg:
            tb_writer += 1
            continue
        # 取第一个看起来像 "XxxError: ..." 的行
        m = re.search(r"^([A-Za-z_.]*(?:Error|Exception)[^\n]*)", seg, re.M)
        others.append(m.group(1)[:100] if m else seg.strip().splitlines()[0][:100])
    return total, tb_writer, others


def analyze_log(path):
    with open(path, errors="ignore") as f:
        txt = f.read()
    print("\n--- batch server ---")
    m = BATCH_RE.search(txt)
    if m:
        print("rounds=%s  steps=%s  平均 batch=%s（峰值 %s / 上限 %s）"
              % (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)))
        print("每轮耗时拆分：sync %s + build %s + forward %s + write %s = %s ms"
              % (m.group(6), m.group(7), m.group(8), m.group(9), m.group(10)))
        print("稳态 %s step/s" % m.group(11))
        ws = WRITE_SPLIT_RE.search(txt)
        if ws:
            print("回写拆分：h/c %s + 动作 %s ms" % (ws.group(1), ws.group(2)))
        h0 = H0_RE.search(txt)
        if h0:
            bad = int(h0.group(2))
            print("episode 边界：首步 hidden=0 命中 %s 次，违反 %s 次%s"
                  % (h0.group(1), h0.group(2),
                     "" if bad == 0 else "  ← **契约被破坏**"))
    else:
        m2 = BATCH_RE_OLD.search(txt)
        if m2:
            print("rounds=%s  steps=%s  平均 batch=%s（峰值 %s / 上限 %s）  forward(build+net) %.2f ms/轮  稳态 %.1f step/s"
                  % (m2.group(1), m2.group(2), m2.group(3), m2.group(4), m2.group(5),
                     float(m2.group(6)), float(m2.group(7))))
        else:
            print("(无 async 汇总行：local 模式不经过 server，属正常)")

    print("\n--- 异常分类 ---")
    _tot, _tb, _oth = classify_tracebacks(txt)
    print("Traceback 总数 %d：tensorboardX 收尾 EOFError %d，其它 %d"
          % (_tot, _tb, len(_oth)))
    for o in _oth[:5]:
        print("  ⚠️ %s" % o)
    if not _oth:
        print("  （除 tensorboardX 收尾噪声外，无训练代码异常）")

    print("--- 各 actor ---")
    tot_s = tot_st = 0.0
    max_tt = 0.0
    n = 0
    for name, st, tt, thr, mode in STEP_RE.findall(txt):
        print("%-10s steps=%-6s %6.1f s  %6.2f step/s  mode=%s" % (name, st, float(tt), float(thr), mode))
        tot_st += float(st)
        tot_s += float(tt)
        max_tt = max(max_tt, float(tt))
        n += 1
    if n:
        # 「聚合吞吐」= 总步数 / 最长那个 actor 的耗时。
        # 不能用 Σ(每 actor 吞吐) —— 那是把每个 actor 单独跑的速度加起来，
        # 隐含「它们完全并行」的假设，而 GIL 下多线程 actor 并不是。
        print("合计 %d 步 / %d 个 actor / 最长 %.1f s -> 聚合 %.2f step/s"
              % (tot_st, n, max_tt, tot_st / max_tt if max_tt else 0.0))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyze_csv(sys.argv[1])
    if len(sys.argv) > 2:
        analyze_log(sys.argv[2])
