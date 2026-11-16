#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描仓库里的环境变量开关，生成/校验开关索引（避免文档与代码脱节）。

用法：
  python3 experiments/env_index.py                 # 打印所有开关
  python3 experiments/env_index.py --md out.md     # 写成 markdown 表
  python3 experiments/env_index.py --check         # 只检查新增/消失（对比 ENV_KNOBS.md）

识别三类写法：
  Python: os.environ.get("X", "default") / os.environ["X"]
  Shell : ${X:-default} / ${X:?msg} / "$X" 赋值使用
输出按前缀分组（MAS_SL_ / MAS_EXPERT_ / OFF_ / 其它），并标注出现次数与默认值。
"""
import argparse
import collections
import os
import re
import sys

ROOTS = [
    os.path.expanduser("~/mini-AlphaStar"),
    os.path.expanduser("~/mini-AlphaStar-off"),
]
SKIP_DIRS = {".git", "__pycache__", "model", "data", "outputs", "log", "log_night", "revised_simple64_replays"}

PY_GET = re.compile(r"os\.environ\.get\(\s*[\"']([A-Z][A-Z0-9_]+)[\"']\s*(?:,\s*([^)]{0,40}))?\)")
PY_IDX = re.compile(r"os\.environ\[\s*[\"']([A-Z][A-Z0-9_]+)[\"']\s*\]")
SH_DEF = re.compile(r"\$\{([A-Z][A-Z0-9_]+):-([^}]{0,40})\}")
SH_REQ = re.compile(r"\$\{([A-Z][A-Z0-9_]+):\?")
SH_BARE = re.compile(r"\$\{([A-Z][A-Z0-9_]+)\}")   # 仅用于检测，不入表（本地变量噪声太大）
ENV_PREFIX = re.compile(r"^(MAS_|OFF_|PF_|CUDA_|CUBLAS_|SC2|DISPLAY|OMP_|MKL_|OPENBLAS_|NUMEXPR_)")


def scan():
    hits = collections.defaultdict(lambda: {"files": set(), "defaults": set(), "count": 0})
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith((".py", ".sh")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    txt = open(p, "r", errors="ignore").read()
                except Exception:
                    continue
                rel = p.replace(os.path.expanduser("~") + "/", "")
                for rx, kind in ((PY_GET, "py"), (PY_IDX, "py"), (SH_DEF, "sh"), (SH_REQ, "sh")):
                    for m in rx.finditer(txt):
                        name = m.group(1)
                        # shell 侧的 ${X:-...} / ${X:?...} 大量是本地变量（$TAG/$CONC/$BASE），
                        # 只保留带环境变量前缀的，避免把 283 个"变量"里的 200 个本地变量也算进来。
                        if kind == "sh" and not ENV_PREFIX.match(name):
                            continue
                        hits[name]["files"].add(rel)
                        hits[name]["count"] += 1
                        if kind in ("py", "sh") and rx in (PY_GET, SH_DEF) and m.lastindex and m.lastindex >= 2:
                            d = (m.group(2) or "").strip().strip('"\'')
                            if d:
                                hits[name]["defaults"].add(d)
    return hits


CURATED = [
 ("MAS_SL_CLASS_WEIGHTS", "0.15,build_pylon=8,...", "按动作名前缀给损失加权（必须精确写 pylon/gateway，写 build= 会连带 Build_Nexus）"),
 ("MAS_SL_BALANCE", "0", "按动作类型做 batch 内均衡重采样（只改采样不改损失）；压 other 会崩，别乱用"),
 ("MAS_SL_LR / MAS_SL_EPOCHS", "1e-4 / -", "学习率与轮数；cwB 配方是 1e-3 / 4（ep4 是峰值，ep8 崩）"),
 ("MAS_SL_SEED", "20261108", "训练 seed；同配方 7 个 run 双峰（3/7 可用），必须多 run"),
 ("MAS_DAGGER", "0", "声明这是 DAgger 数据；与 manifest 的 dagger 双向校验，不一致直接拒跑"),
 ("MAS_SL_LOAD_STATE / MAS_SL_RESTORE_PATH", "0 / -", "热启动：从指定 ckpt 继续训（[SL-restore] 会打印加载了多少 key）"),
 ("MAS_ALLOW_MORE", "0", "跳过『训练数上限 + 内存』前置检查（可能 OOM，慎用）"),
 ("MAS_PF_SKIP", "0", "跳过全部前置检查（只在你明确知道后果时用）"),
 ("OFF_DIFFICULTY", "3", "内置 AI 难度；采集/评测基准都是 3"),
 ("OFF_GAME_STEPS", "30000", "单局步数上限；**评测必须 40000**，30k 会把优势局打成平"),
 ("OFF_SEED / OFF_MAX_EP", "- / 1", "评测种子与局数；一进程一局，种子固定不变"),
 ("OFF_GPUS / OFF_GPU", "-", "评测用哪几张卡 / 单进程用哪张；超过 8 进程/卡会 CUDNN_STATUS_NOT_INITIALIZED"),
 ("OFF_ACT_STATS", "0", "导出每局的 function 调用计数（诊断动作分布）"),
 ("OFF_ACT_BIAS", "", "推理端动作 logits 偏置，如 Attack_pt=+2.0；实测平均无增益，默认关"),
 ("OFF_MAX_JOBS / OFF_MAX_PER_GPU", "NG*6 / 8", "评测并发上限（本周新增），防止打爆单卡显存"),
 ("MAS_EXPERT_IMPL", "v5", "采集用哪个规则专家（v5=随机化 fast_rush，v4 仅回退）"),
 ("MAS_DUAL_MODE", "append", "learner+专家的协作方式：append=专家叠加命令，replace=专家接管"),
]


def group_of(name):
    for pre in ("MAS_SL_", "MAS_EXPERT_", "MAS_DUAL_", "MAS_", "OFF_", "PF_", "CUDA_", "SC2", "DISPLAY"):
        if name.startswith(pre):
            return pre
    return "其它"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    hits = scan()
    rows = []
    for name in sorted(hits, key=lambda n: (group_of(n), n)):
        d = hits[name]
        rows.append((name, group_of(name), len(d["files"]), d["count"], "/".join(sorted(d["defaults"])) or "-"))

    print("共 %d 个环境变量" % len(rows))
    cur = None
    for name, g, nf, cnt, dflt in rows:
        if g != cur:
            print("\n## %s*" % g)
            cur = g
        print("  %-32s files=%-2d uses=%-3d default=%s" % (name, nf, cnt, dflt))

    if args.md:
        with open(args.md, "w") as f:
            f.write("# 环境变量开关索引（由 experiments/env_index.py 自动生成）\n\n")
            f.write("> 修改代码后重跑 `python3 experiments/env_index.py --md <此文件>` 即可更新。\n\n")
            f.write("## 常用开关（人工维护）\n\n| 变量 | 默认 | 说明 |\n|---|---|---|\n")
            for k, d, desc in CURATED:
                f.write("| `%s` | %s | %s |\n" % (k, d, desc))
            f.write("\n## 全量索引（自动生成）\n\n")
            f.write("| 变量 | 分组 | 文件数 | 出现次数 | 默认值 |\n|---|---|---:|---:|---|\n")
            for name, g, nf, cnt, dflt in rows:
                f.write("| `%s` | %s | %d | %d | %s |\n" % (name, g, nf, cnt, dflt))
        print("\n[env_index] 已写入 %s" % args.md)

    if args.check:
        doc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ENV_KNOBS.md")
        if not os.path.exists(doc):
            print("[env_index] 还没有 ENV_KNOBS.md，先跑 --md 生成")
            return 1
        old = set(re.findall(r"\| `([A-Z][A-Z0-9_]+)`", open(doc).read()))
        new = {r[0] for r in rows}
        added, gone = sorted(new - old), sorted(old - new)
        print("\n[check] 新增: %s" % (added or "无"))
        print("[check] 消失: %s" % (gone or "无"))
        return 0 if not added and not gone else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
