# -*- coding: utf-8 -*-
"""第 71 轮：`_pack_actions` 的等价性测试。

**它要证明什么**：把 8 个动作字段"先 cat 再一次 .cpu()、再在 CPU 上切片"
得到的结果，与原来"每个字段各自 .cpu()"**逐元素完全相同**。

为什么值得单独测：打包版把「字段边界」从 Python 变量变成了**列偏移量**。
一旦某个宽度算错（尤其是 `units` 会被 max_selected 截断、`target_location` 固定 2 列），
写进共享内存的动作就会**整体错位一格** —— 而 server 不报错，
actor 拿到的 action 仍然"形状合法"，只是语义全错。

覆盖点：
  [A] 多种批大小 k（1 / 3 / 5）与 max_selected（4 / 8）
  [B] units 列数 > max_selected（必须截断）与 == max_selected（不截断）
  [C] 大整数（2^40 量级）：`.long()` 不能经由 float 中转而丢精度
  [D] 输出 dtype/设备：int64 + CPU
  [E] 8 个字段的列宽之和 == 打包张量宽度（偏移量自洽）

跑法：python experiments/_test_r71_pack.py
"""
from __future__ import print_function

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer

Acts = collections.namedtuple(
    "Acts", "action_type delay queue units target_unit target_location")


def make_case(k, units_cols, max_selected, big=False, mixed=False):
    g = torch.Generator().manual_seed(1234 + k * 10 + units_cols)
    hi = 1 << 40 if big else 1000
    acts = Acts(
        action_type=torch.randint(0, hi, (k,), generator=g),
        delay=torch.randint(0, 8, (k,), generator=g),
        queue=torch.randint(0, 3, (k,), generator=g),
        units=torch.randint(0, hi, (k, units_cols), generator=g),
        target_unit=torch.randint(0, hi, (k,), generator=g),
        target_location=torch.randint(0, 64, (k, 2), generator=g),
    )
    sun = torch.randint(0, 48, (k,), generator=g)
    en = torch.randint(0, 128, (k,), generator=g)
    if mixed:
        # 真实模型里各 head 的 dtype 不一定都是 int64（有的经 argmax 出来是 int32）。
        # 打包版在 cat 之前必须逐个 .long()，否则 torch.cat 会直接抛 dtype 不一致。
        acts = acts._replace(delay=acts.delay.int(), target_unit=acts.target_unit.int())
        sun = sun.int()
    return acts, sun, en


def reference(acts, sun, en, k, max_selected):
    """原来那 8 行"各自 .cpu()"的写法，逐字复刻，作为基准。"""
    return {
        "a_type": acts.action_type.reshape(-1).cpu().long(),
        "a_delay": acts.delay.reshape(-1).cpu().long(),
        "a_queue": acts.queue.reshape(-1).cpu().long(),
        "a_units": acts.units.reshape(k, -1)[:, :max_selected].cpu().long(),
        "a_tunit": acts.target_unit.reshape(-1).cpu().long(),
        "a_tloc": acts.target_location.reshape(k, -1)[:, :2].cpu().long(),
        "a_sun": sun.reshape(-1).cpu().long(),
        "a_en": en.reshape(-1).cpu().long(),
    }


def main():
    res = [True]

    def chk(name, cond, detail):
        res[0] = res[0] and bool(cond)
        print("  %-46s %s   %s" % (name, "PASS" if cond else "FAIL", detail), flush=True)

    #            k  units_cols max_selected  big    mixed(dtype)
    cases = [(1, 4, 4, False, False),
             (3, 8, 4, False, False),
             (5, 4, 4, False, False),
             (5, 12, 8, True, False),     # 截断 + 大整数
             (3, 4, 4, False, True)]      # 混用 int32/int64

    for (k, uc, ms, big, mixed) in cases:
        print("\n=== k=%d, units 列数=%d, max_selected=%d, 大整数=%s, 混合dtype=%s ==="
              % (k, uc, ms, big, mixed), flush=True)
        acts, sun, en = make_case(k, uc, ms, big, mixed)
        ref = reference(acts, sun, en, k, ms)
        pk = AsyncBatchInferenceServer._pack_actions(acts, sun, en, k, ms)

        chk("字段数一致（8）", len(pk) == len(ref) == 8, "%d vs %d" % (len(pk), len(ref)))
        bad = []
        for name in ref:
            a, b = ref[name], pk[name]
            if a.shape != b.shape:
                bad.append("%s 形状 %s vs %s" % (name, tuple(a.shape), tuple(b.shape)))
            elif not torch.equal(a, b):
                d = int((a.to(torch.float64) - b.to(torch.float64)).abs().max())
                bad.append("%s 值差 %d" % (name, d))
        chk("8 个字段逐元素相同", not bad, "；".join(bad) if bad else "全部 identical")

        widths = [int(pk[n].shape[1]) if pk[n].dim() == 2 else 1 for n in pk]
        chk("列宽之和自洽", sum(widths) == sum(
            [int(ref[n].shape[1]) if ref[n].dim() == 2 else 1 for n in ref]),
            "widths=%s sum=%d" % (widths, sum(widths)))

        chk("dtype 全为 int64", all(pk[n].dtype == torch.int64 for n in pk),
            str({n: str(pk[n].dtype) for n in pk}))
        chk("设备全为 CPU", all(pk[n].device.type == "cpu" for n in pk),
            str({n: pk[n].device.type for n in pk}))

        if uc > ms:
            chk("units 被截断到 %d 列" % ms, pk["a_units"].shape[1] == ms,
                str(tuple(pk["a_units"].shape)))
        if big:
            chk("大整数未丢精度（>2^32）", int(pk["a_type"].max()) >= (1 << 32),
                "max=%d" % int(pk["a_type"].max()))
        if mixed:
            chk("混合 dtype 输入仍产出 int64", all(pk[n].dtype == torch.int64 for n in pk),
                str({n: str(pk[n].dtype) for n in pk}))

    print("\n=== 结论 ===", flush=True)
    if res[0]:
        print(" 打包版与逐字段版逐元素相同，宽度/偏移/dtype 全部自洽。", flush=True)
        return 0
    print(" 存在 FAIL —— 动作字段可能整体错位，不可上线。", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
