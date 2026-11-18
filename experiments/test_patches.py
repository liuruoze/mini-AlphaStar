#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""针对本周新增/改动代码路径的离线单测（不需要 GPU、不需要 SC2）。

覆盖：
  1. （已移除）开局帧加权：该开关经实测会加速塌缩，2026-11-15 删除代码
  2. _balance_window_weights 动作均衡采样（稀有类权重更高、上限生效）
  3. _WindowSubset           fp16 存储 -> fp32 上采样，且 fp32 数据零拷贝语义不变
  4. to_fp16                 转换后 dtype 正确、数值可逆（float() 后与原文一致）
  5. eval_table              解析与判胜口径（native / 官方两种）

运行：python3 experiments/test_patches.py
返回码 0 = 全部通过。
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "mini-AlphaStar"))
os.environ.setdefault("MAS_ON_SERVER", "1")

import alphastarmini.core.sl.sl_train_by_tensor as T  # noqa: E402
from pysc2.lib.actions import RAW_FUNCTIONS as RF  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print("[%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAIL.append(name)


def act_id(substr, must=True):
    for i, fn in enumerate(RF):
        if fn.name.startswith(substr):
            return i
    if must:
        raise RuntimeError("找不到动作: %s" % substr)
    return None


def make_dataset(lens, action_cycle, feat_dim=32, lab_dim=564):
    """构造 (features, labels) 列表：labels 是 one-hot。"""
    tensors = []
    for n in lens:
        f = torch.randn(n, feat_dim)
        l = torch.zeros(n, lab_dim)
        for j in range(n):
            l[j, action_cycle[j % len(action_cycle)]] = 1.0
        tensors.append((f, l))
    return tensors


def make_windows(tensors, seq):
    wins = []
    for i, (f, _) in enumerate(tensors):
        for s in range(max(f.size(0) - seq + 1, 0)):
            wins.append((i, s))
    return wins


def test_split():
    """train/val 划分：不重叠、并集覆盖全部、（train_end-1,val_from) 之间无缝隙。
    这个函数一旦悄悄变了，所有历史对比都会失效，所以单独锁死。"""
    for total in (2, 3, 5, 60):
        (tf, te), (vf, ve) = T.split_indices(total)
        check("split.%d.不重叠" % total, te <= vf, "train=[%d,%d) val=[%d,%d)" % (tf, te, vf, ve))
        check("split.%d.覆盖" % total, tf == 0 and ve == total, "train=[%d,%d) val=[%d,%d)" % (tf, te, vf, ve))
        check("split.%d.无缝隙" % total, te == vf, "gap=%d" % (vf - te))
    # 60 文件时 val 应等于 VAL_NUM（默认 12 = 20%）
    (_, te), (vf, _) = T.split_indices(60)
    check("split.60.val数", vf == 60 - min(T.VAL_NUM, 59), "VAL_NUM=%d vf=%d" % (T.VAL_NUM, vf))
    # 单文件退化
    (tf2, te2), (vf2, ve2) = T.split_indices(1)
    check("split.1.退化", (tf2, te2) == (0, 1) and (vf2, ve2) == (0, 1))


def test_balance():
    seq = T.SEQ_LEN
    # 让 gateway/probe 真的"稀有"（各 1/22），否则按 目标/实际 的公式它们反而会被降权
    ids = [act_id("Attack_pt")] * 20 + [act_id("Build_Gateway_pt"), act_id("Train_Probe_quick")]
    tensors = make_dataset([200], ids)
    ds = T._WindowSubset(tensors, make_windows(tensors, seq), seq)

    os.environ["MAS_SL_BALANCE_MAXW"] = "10"
    w = T._balance_window_weights(ds, seq)
    os.environ.pop("MAS_SL_BALANCE_MAXW")

    gid = act_id("Build_Gateway_pt")
    atk = act_id("Attack_pt")
    w_gw = [float(w[k]) for k, (i, s) in enumerate(ds.windows)
            if int(tensors[i][1][min(s + seq - 1, tensors[i][1].size(0) - 1)].argmax()) == gid]
    w_atk = [float(w[k]) for k, (i, s) in enumerate(ds.windows)
             if int(tensors[i][1][min(s + seq - 1, tensors[i][1].size(0) - 1)].argmax()) == atk]
    check("balance.长度", len(w) == len(ds.windows))
    check("balance.上限生效", float(w.max()) <= 10.0 + 1e-6, "max=%.2f" % float(w.max()))
    check("balance.稀有类权重更高", (min(w_gw) if w_gw else 0) > (max(w_atk) if w_atk else 0),
          "gw=%.2f atk=%.2f" % (w_gw[0] if w_gw else -1, w_atk[0] if w_atk else -1))


def test_upcast():
    seq = T.SEQ_LEN
    tensors = [(torch.randn(20, 8).half(), torch.zeros(20, 564).half()) for _ in range(2)]
    ds = T._WindowSubset(tensors, make_windows(tensors, seq), seq)
    f, l = ds[0]
    check("upcast.fp16->fp32", f.dtype == torch.float32 and l.dtype == torch.float32,
          "%s/%s" % (f.dtype, l.dtype))

    tensors32 = [(torch.randn(20, 8), torch.zeros(20, 564)) for _ in range(2)]
    ds32 = T._WindowSubset(tensors32, make_windows(tensors32, seq), seq)
    f2, l2 = ds32[0]
    check("upcast.fp32不变", f2.dtype == torch.float32 and torch.equal(f2, tensors32[0][0][:seq]))


def test_to_fp16():
    tmp = tempfile.mkdtemp(prefix="fp16test_")
    src = os.path.join(tmp, "src")
    dst = os.path.join(tmp, "dst")
    os.makedirs(src)
    f = torch.randn(30, 16)
    l = torch.zeros(30, 564)
    l[:, 2] = 1.0
    torch.save((f, l), os.path.join(src, "a.pt"))
    with open(os.path.join(src, "manifest.json"), "w") as fh:
        fh.write('{"dagger": true, "note": "unit-test"}')

    r = subprocess.run([sys.executable, os.path.join(HOME, "mini-AlphaStar/experiments/to_fp16.py"),
                        "--src", src, "--dst", dst], capture_output=True, text=True)
    ok = r.returncode == 0
    check("to_fp16.退出码", ok, r.stderr[-200:] if not ok else "")
    if ok:
        f16, l16 = torch.load(os.path.join(dst, "a.pt"), map_location="cpu")
        check("to_fp16.dtype", f16.dtype == torch.float16 and l16.dtype == torch.float16)
        check("to_fp16.数值可逆", torch.allclose(f16.float(), f, atol=2e-3),
              "maxdiff=%.4g" % float((f16.float() - f).abs().max()))
        import json
        m = json.load(open(os.path.join(dst, "manifest.json")))
        check("to_fp16.继承dagger", m.get("dagger") is True)
    shutil.rmtree(tmp, ignore_errors=True)


def test_bad_file():
    """坏文件必须报错退出，而不是静默少用数据（R7）。"""
    tmp = tempfile.mkdtemp(prefix="badds_")
    f = torch.randn(20, 8); l = torch.zeros(20, 564); l[:, 2] = 1.0
    torch.save((f, l), os.path.join(tmp, "good.pt"))
    with open(os.path.join(tmp, "bad.pt"), "wb") as fh:
        fh.write(b"not a torch file")
    files = ["good.pt", "bad.pt"]

    os.environ.pop("MAS_SL_ALLOW_BAD_FILES", None)
    raised = False
    try:
        T.getReplayData(tmp + "/", files)
    except RuntimeError as e:
        raised = "文件读取失败" in str(e)
    check("badfile.默认报错", raised)

    os.environ["MAS_SL_ALLOW_BAD_FILES"] = "1"
    try:
        ds = T.getReplayData(tmp + "/", files)
        check("badfile.显式放行", len(ds) == 1, "datasets=%d" % len(ds))
    except Exception as e:
        check("badfile.显式放行", False, repr(e))
    finally:
        os.environ.pop("MAS_SL_ALLOW_BAD_FILES", None)
    shutil.rmtree(tmp, ignore_errors=True)


def test_eval_table():
    p = os.path.join(HOME, "mini-AlphaStar/experiments/eval_table.py")
    spec = importlib.util.spec_from_file_location("eval_table", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    line = ("2026-11-14 04:33:05 Episode: [1/1]| outcome: +1 | raw_reward: +0 | food_used: 103.0 | "
            "army_count: 34.0 | collected_points: 10612.0 | used_points: 7175.0 | killed_points: 11500.0 | "
            "our_units: 85 | enemy_units: 1 | enemy_army: 0 | steps: 39992.000s \n")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    tmp.write(line)
    tmp.close()
    d = m.parse(tmp.name)
    os.unlink(tmp.name)
    check("eval_table.解析", d is not None and d["killed"] == 11500.0 and d["our"] == 85,
          str(d))
    check("eval_table.官方判胜(outcome=1)", m.official_win(d) is True)
    d2 = {"outcome": 0, "killed": 5000.0}
    d3 = {"outcome": 0, "killed": 100.0}
    check("eval_table.官方判胜(killed>4000)", m.official_win(d2) is True)
    check("eval_table.官方判负", m.official_win(d3) is False)


def test_bad_norm():
    """is_bad_norm：NaN/±Inf 为坏，有限值为好（2026-11-19 新增）。"""
    from alphastarmini.lib.utils import is_bad_norm as f
    check("is_bad_norm(nan)", f(float("nan")) is True)
    check("is_bad_norm(inf)", f(float("inf")) is True)
    check("is_bad_norm(-inf)", f(float("-inf")) is True)
    check("is_bad_norm(0.0)", f(0.0) is False)
    check("is_bad_norm(1e-3)", f(1e-3) is False)
    check("is_bad_norm(None)", f(None) is True)



def test_validate_lr():
    """validate_lr：(0,1] 内通过，其余抛错（2026-11-19 新增）。"""
    from alphastarmini.lib.utils import validate_lr as f
    check("validate_lr(1e-6)", abs(f("1e-6") - 1e-6) < 1e-18)
    check("validate_lr(3e-6)", abs(f(3e-6) - 3e-6) < 1e-18)
    for bad in (0, -1e-6, 1.5, "abc"):
        try:
            f(bad)
            check("validate_lr(%r) 应抛错" % (bad,), False)
        except Exception:
            check("validate_lr(%r) 抛错" % (bad,), True)


def main():
    print("=== 新增代码路径离线单测 ===")
    print("SEQ_LEN=%d" % T.SEQ_LEN)
    test_split()
    test_balance()
    test_upcast()
    test_to_fp16()
    test_bad_file()
    test_eval_table()
    test_bad_norm()
    test_validate_lr()
    print("\n=== 结果：%d 项失败 ===" % len(FAIL))
    if FAIL:
        print("失败项: %s" % ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
