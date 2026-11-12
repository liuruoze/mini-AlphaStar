#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BENCH v1：统一基准测试执行器。

用法（建议在 tmux 里跑，因为全程 35–40 分钟）：
    tmux new -s bench1 "cd ~/mini-AlphaStar && python -u experiments/sl/bench.py run dropout01 --env MAS_SL_DROPOUT=0.1"

子命令：
    run <name> [--env K=V ...] [--runs 3] [--epochs 10] [--lr 1e-4] [--ss 5] [--gamma 0.2]
    report <name>
    list

流程（run）：
    1) 并行启动 N 个训练（各占一张卡、各自 model_bench_<name>_<r>/ 目录）
    2) 等 epXX 落盘；读最终 loss，标出是否落在 [6.7, 7.2] 安全区
    3) 用固定 seed 3038 各评 1 局
    4) 汇总：每 run 的 killed_points / 过线与否 / 中位数 / 判定
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

TRAIN = os.path.expanduser("~/mini-AlphaStar")
EVAL = os.path.expanduser("~/mini-AlphaStar-off")
DATA_DEFAULT = "./data/replay_data_v3_90_nonop/"
DATA = DATA_DEFAULT
RESULT_DIR = os.path.join(TRAIN, "bench_results")
SEED = 3038
LOSS_LO, LOSS_HI = 6.7, 7.2
KILL_LINE = 4000


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def tmux_alive(sess):
    r = sh("tmux has-session -t %s 2>/dev/null" % sess)
    return r.returncode == 0


def kill_sess(sess):
    sh("tmux kill-session -t %s 2>/dev/null" % sess)


def tail_loss(logfile):
    """日志格式是 `loss: 9.355849.`（带尾随句号），必须去掉再转 float。"""
    if not os.path.exists(logfile):
        return None
    r = sh("grep -o 'loss: [0-9.]*' %s | tail -1" % logfile)
    m = re.search(r"loss:\s*([0-9.]+)", r.stdout)
    if not m:
        return None
    txt = m.group(1).rstrip(".")
    try:
        return float(txt)
    except ValueError:
        return None


def ep_file(model_dir, epochs):
    hits = sorted(glob.glob(os.path.join(model_dir, "*_ep%02d.pth" % epochs)))
    return hits[-1] if hits else None


def start_train(name, r, args, envs):
    model_dir = os.path.join(TRAIN, "model_bench_%s_%d" % (name, r))
    log = os.path.join(TRAIN, "log", "p2_bench_%s_%d.log" % (name, r))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    sh("rm -rf %s && mkdir -p %s" % (model_dir, model_dir))

    ev = " ".join(envs) if envs else ""
    cmd = (
        "cd {train} && {ev} "
        "MAS_SL_EPOCHS={ep} MAS_SL_LR={lr} MAS_SL_STEP_SIZE={ss} MAS_SL_GAMMA={gm} "
        "MAS_SL_WORKERS=4 MAS_SL_MODEL_PATH=./model_bench_{name}_{r}/ "
        "MAS_DATA_DIR={data} P2_GPU={gpu} "
        "bash run_data_env.sh log/p2_bench_{name}_{r}.log"
    ).format(train=TRAIN, ev=ev, ep=args.epochs, lr=args.lr, ss=args.ss, gm=args.gamma,
             name=name, r=r, data=args.data, gpu=(r % 4))
    sess = "bt_%s_%d" % (name, r)
    kill_sess(sess)
    r0 = sh('tmux new-session -d -s %s "%s"' % (sess, cmd))
    return model_dir, log, r0.returncode


def start_eval(name, r):
    model_dir = os.path.join(TRAIN, "model_bench_%s_%d" % (name, r))
    tag = "bench_%s_%d" % (name, r)
    out = os.path.join(EVAL, "outputs", "%s_%d.txt" % (tag, SEED))
    if os.path.exists(out):
        os.remove(out)          # 避免追加导致重复统计
    cmd = ("cd %s && OFF_GPUS=%d bash run_eval_batch.sh %s/ %s %d 1 1 > log/%s.log 2>&1"
           % (EVAL, (r % 4), model_dir, tag, SEED, tag))
    sess = "be_%s_%d" % (name, r)
    kill_sess(sess)
    sh('tmux new-session -d -s %s "%s"' % (sess, cmd))
    return out


def parse_killed(path):
    if not os.path.exists(path):
        return None
    r = sh("grep 'Episode:' %s | tail -1" % path)
    m = re.search(r"killed_points:\s*([0-9.]+)", r.stdout)
    k = float(m.group(1)) if m else None
    m2 = re.search(r"steps:\s*([0-9.]+)", r.stdout)
    st = float(m2.group(1)) if m2 else None
    m3 = re.search(r"outcome:\s*([-+]?\d+)", r.stdout)
    oc = int(m3.group(1)) if m3 else None
    return (k, st, oc)


def cmd_run(args):
    os.makedirs(RESULT_DIR, exist_ok=True)
    envs = args.env or []
    print("=" * 78)
    print("BENCH run: %s  (runs=%d, epochs=%d, lr=%s, ss=%s, gamma=%s)"
          % (args.name, args.runs, args.epochs, args.lr, args.ss, args.gamma))
    if envs:
        print("extra env: " + " ".join(envs))
    print("=" * 78)

    info = {}
    for r in range(1, args.runs + 1):
        md, log, rc = start_train(args.name, r, args, envs)
        info[r] = {"model_dir": md, "log": log}
        print("[train] run%d 启动 (rc=%d) -> %s" % (r, rc, os.path.basename(md)))
        time.sleep(3)

    # ---- 等训练 ----
    t0 = time.time()
    deadline = t0 + args.train_timeout
    while time.time() < deadline:
        done = [r for r in info if ep_file(info[r]["model_dir"], args.epochs)]
        print("  [%s] 训练完成 %d/%d" % (time.strftime("%H:%M:%S"), len(done), args.runs))
        if len(done) == args.runs:
            break
        time.sleep(60)
    else:
        print("!! 训练超时（%d 分钟），已完成的继续" % (args.train_timeout // 60))

    ok_runs = []
    for r in info:
        f = ep_file(info[r]["model_dir"], args.epochs)
        loss = tail_loss(info[r]["log"])
        info[r]["loss"] = loss
        info[r]["ckpt"] = f
        if not f:
            print("[train] run%d 无 ep%d 权重 -> 跳过" % (r, args.epochs))
            continue
        flag = "OK" if (loss is not None and LOSS_LO <= loss <= LOSS_HI) else "LOSS异常(可丢)"
        print("[train] run%d loss=%s %s" % (r, loss, flag))
        ok_runs.append(r)

    if not ok_runs:
        print("!! 没有任何可用 run，终止")
        return 1

    # ---- 评测 ----
    outs = {}
    for r in ok_runs:
        outs[r] = start_eval(args.name, r)
        print("[eval ] run%d -> %s" % (r, os.path.basename(outs[r])))
        time.sleep(2)

    t1 = time.time()
    edead = t1 + args.eval_timeout
    while time.time() < edead:
        got = [r for r in outs if parse_killed(outs[r])]
        print("  [%s] 评测完成 %d/%d" % (time.strftime("%H:%M:%S"), len(got), len(outs)))
        if len(got) == len(outs):
            break
        time.sleep(30)

    # ---- 汇总 ----
    rows = []
    for r in ok_runs:
        res = parse_killed(outs[r])
        if not res:
            continue
        k, st, oc = res
        rows.append({"run": r, "killed": k, "steps": st, "outcome": oc,
                     "loss": info[r]["loss"], "win": k > KILL_LINE})

    if not rows:
        print("!! 评测全部失败")
        return 1

    kills = sorted(x["killed"] for x in rows)
    n = len(kills)
    med = kills[n // 2] if n % 2 else (kills[n // 2 - 1] + kills[n // 2]) / 2
    nwin = sum(1 for x in rows if x["win"])

    print()
    print("=" * 78)
    print("结果: %s" % args.name)
    print("=" * 78)
    print("%-6s %-10s %-10s %-8s %-6s" % ("run", "loss", "killed", "steps", "过线"))
    for x in rows:
        print("%-6d %-10s %-10.0f %-8.0f %-6s"
              % (x["run"], ("%.3f" % x["loss"]) if x["loss"] else "-",
                 x["killed"], x["steps"] or 0, "YES" if x["win"] else "no"))
    print("-" * 78)
    print("过线 %d/%d   中位数 killed = %.0f   均值 = %.0f"
          % (nwin, len(rows), med, sum(kills) / len(kills)))
    verdict = "通过(3/3)" if nwin == len(rows) and len(rows) >= 3 else (
        "存疑" if nwin >= 2 else "否证")
    print("判定: %s" % verdict)

    rec = {"name": args.name, "seed": SEED, "epochs": args.epochs, "lr": args.lr,
           "ss": args.ss, "gamma": args.gamma, "env": envs,
           "rows": rows, "nwin": nwin, "median": med, "verdict": verdict,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_json = os.path.join(RESULT_DIR, "%s.json" % args.name)
    with open(out_json, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULT_DIR, "%s.txt" % args.name), "w") as f:
        f.write(json.dumps(rec, ensure_ascii=False, indent=2))
    print("已保存: %s" % out_json)
    return 0


def cmd_report(args):
    p = os.path.join(RESULT_DIR, "%s.json" % args.name)
    if not os.path.exists(p):
        print("没有找到 %s" % p)
        return 1
    rec = json.load(open(p))
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


def cmd_list(args):
    os.makedirs(RESULT_DIR, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(RESULT_DIR, "*.json"))):
        rec = json.load(open(f))
        print("%-22s 过线 %d/%d  中位 %8.0f  %s"
              % (rec["name"], rec["nwin"], len(rec["rows"]), rec["median"], rec["verdict"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--epochs", type=int, default=10)
    r.add_argument("--lr", default="1e-4")
    r.add_argument("--ss", type=int, default=5)
    r.add_argument("--gamma", default="0.2")
    r.add_argument("--data", default=DATA_DEFAULT)
    r.add_argument("--env", nargs="*", default=[])
    r.add_argument("--train-timeout", type=int, default=2700)
    r.add_argument("--eval-timeout", type=int, default=1500)

    p = sub.add_parser("report")
    p.add_argument("name")

    sub.add_parser("list")

    args = ap.parse_args()
    if args.cmd == "run":
        sys.exit(cmd_run(args))
    if args.cmd == "report":
        sys.exit(cmd_report(args))
    if args.cmd == "list":
        sys.exit(cmd_list(args))


if __name__ == "__main__":
    main()
