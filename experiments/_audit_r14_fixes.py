"""第 14 轮审计：确认第 12/13 轮两处最复杂的改动**真的正确**。

前两轮是"改完看到 rc=0 就收工"，本轮改成"用独立实现对照"。七个检查项：

[A] pad_teacher_units_logits 补的第 13 列 == change_units_and_logits 的原生 padding 列
    （逐位比对，覆盖 entity_num = 1 / 12 / 55 / 127 四种取值）
[B] pad 不修改传入的 entity_num（第 14 轮修掉的 clamp_ in-place 隐患回归测试）
[C] pad 对已经是 13 宽的输入幂等（不重复补）
[D] kl() 的 valid mask 能消除「target 被 mask、teacher 未 mask」造成的 1e9 量级伪贡献，
    且结果 == 只在共同支撑集上手工算的 KL
[E] kl() 在无 mask 时与「不做 valid 过滤」的原公式逐位一致（不改变正常路径）
[F] 全位置都被 mask 时 kl == 0（不能出 nan）
[G] 1e9 的来源归因验证：分别构造「target 被 mask」与「teacher 被 mask」两种情形，
    确认只有前者会爆 —— 这决定了修法是否对症

用法：python _audit_r14_fixes.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib import utils as L
from alphastarmini.core.rl import rl_algo as RA
from alphastarmini.core.rl import rl_loss as RLoss

DEV = torch.device("cpu")     # 纯数值审计，CPU 足够且无显存压力
SEED = 20261102

MAX_SEL = AHP.max_selected       # 12
MAX_ENT = AHP.max_entities       # 128


def hr(title):
    print("\n=== %s ===" % title, flush=True)


def main():
    torch.manual_seed(SEED)
    ok = True
    results = []

    def record(name, passed, detail=""):
        nonlocal ok
        results.append((name, passed, detail))
        if not passed:
            ok = False
        print("  %-6s %s%s" % ("PASS" if passed else "FAIL", name,
                               ("  " + detail) if detail else ""), flush=True)

    print("AHP.max_selected = %d, AHP.max_entities = %d" % (MAX_SEL, MAX_ENT), flush=True)

    # ---------------------------------------------------------------- [A]
    hr("[A] pad 的第 13 列 == change_units_and_logits 的原生 padding 列")
    B = 8
    units_logits = torch.randn(B, MAX_SEL, MAX_ENT) * 2.0
    for en_val in (1, 12, 55, MAX_ENT - 1):
        en = torch.full((B,), en_val, dtype=torch.int64)

        # 参照实现：直接调用原函数（它对第 13 列赋 (one_hot(en)-1)*1e9）
        # 注意它返回 tuple (logits, gt_units)，取 [0]
        ref = RLoss.change_units_and_logits(
            units_logits.clone(), torch.zeros(B, MAX_SEL, 1, dtype=torch.int64),
            torch.full((B,), MAX_SEL, dtype=torch.int64), en.clone())[0]
        ref_pad = ref[:, -1, :]

        # 待测实现
        got = RLoss.pad_teacher_units_logits(units_logits.clone(), en.clone(), DEV)
        got_pad = got[:, -1, :]

        same_shape = tuple(got.shape) == (B, MAX_SEL + 1, MAX_ENT)
        d = float((got_pad - ref_pad).abs().max())
        # 前 12 列必须原样保留
        d_body = float((got[:, :MAX_SEL, :] - units_logits).abs().max())
        record("[A] entity_num=%-3d" % en_val, same_shape and d == 0.0 and d_body == 0.0,
               "第13列 max|Δ| = %.3e, 前12列 max|Δ| = %.3e, shape=%s"
               % (d, d_body, tuple(got.shape)))

    # ---------------------------------------------------------------- [B]
    hr("[B] pad 不修改传入的 entity_num（clamp_ → clamp 的回归）")
    for en_val in (1, 55, 120):
        en = torch.full((B,), en_val, dtype=torch.int64)
        before = en.clone()
        _ = RLoss.pad_teacher_units_logits(units_logits.clone(), en, DEV)
        d = float((en - before).abs().max())
        record("[B] entity_num=%-3d 未被改写" % en_val, d == 0.0, "max|Δ| = %.3e" % d)

    # 越界值：clamp 应该只在内部生效，不回写
    en_bad = torch.tensor([0, 1, 500, -3, 127, 128, 999, 12], dtype=torch.int64)
    before = en_bad.clone()
    out = RLoss.pad_teacher_units_logits(units_logits.clone(), en_bad, DEV)
    d = float((en_bad - before).abs().max())
    record("[B] 越界 entity_num 不被 clamp 回写", d == 0.0,
           "输入 max|Δ| = %.3e（内部 clamp 保护 one_hot 不越界）" % d)
    # 越界时 padding 列应仍是"恰好一个 0、其余 -1e9"的形态
    col = out[:, -1, :]
    n_zero = int((col.abs() < 1e-6).sum(dim=1).max())
    n_neg = int((col < -1e8).sum(dim=1).min())
    record("[B] 越界时 padding 列形态正确", n_zero == 1 and n_neg == MAX_ENT - 1,
           "每行 0 的个数最多 %d、-1e9 的个数最少 %d" % (n_zero, n_neg))

    # ---------------------------------------------------------------- [C]
    hr("[C] pad 对已是 13 宽的输入幂等")
    already = torch.randn(B, MAX_SEL + 1, MAX_ENT)
    out2 = RLoss.pad_teacher_units_logits(already.clone(), torch.full((B,), 55, dtype=torch.int64), DEV)
    d = float((out2 - already).abs().max())
    record("[C] 幂等", tuple(out2.shape) == tuple(already.shape) and d == 0.0,
           "shape=%s, max|Δ| = %.3e" % (tuple(out2.shape), d))

    # ---------------------------------------------------------------- [D][E][F][G]
    hr("[D]/[G] kl 的 valid mask：定位 1e9 的真实来源并对症")
    # 构造 4 个位置：0/1 正常；2 = target 被 mask；3 = teacher 被 mask
    s_logits = torch.tensor([[0.5, -0.3, 1.2, 0.8]])
    t_logits = torch.tensor([[0.1, 0.4, -0.6, 0.2]])

    t_masked = t_logits.clone()
    t_masked[0, 3] = -1e9                 # teacher 被 mask（情形 G-2）
    s_masked = s_logits.clone()
    s_masked[0, 2] = -1e9                 # target 被 mask（情形 G-1）

    def kl_no_valid(student, teacher):
        """原公式（第 13 轮修复前的等价实现）。"""
        s_lp = F.log_softmax(student, dim=-1)
        t_lp = F.log_softmax(teacher, dim=-1)
        return float((torch.exp(t_lp) * (t_lp - s_lp)).sum())

    # G-2：teacher 被 mask
    k2_no = kl_no_valid(s_logits, t_masked)
    k2_new = float(RA.kl([s_logits, t_masked]).sum())   # kl() 返回逐元素张量
    print("  [G-2] teacher 位置 3 被 mask：原公式 %.6f | 新实现 %.6f" % (k2_no, k2_new), flush=True)
    record("[G] teacher 被 mask 时原公式本来就不爆", abs(k2_no) < 1e3,
           "因为 exp(-1e9)=0，伪贡献被概率权重吃掉")

    # G-1：target 被 mask —— 这才是 1e9 的来源
    k1_no = kl_no_valid(s_masked, t_logits)
    k1_new = float(RA.kl([s_masked, t_logits]).sum())
    print("  [G-1] target 位置 2 被 mask：原公式 %.6e | 新实现 %.6f" % (k1_no, k1_new), flush=True)
    record("[G] **target 被 mask 时原公式会爆到 1e8+**", k1_no > 1e7,
           "实测 %.3e —— 证实第 13 轮的归因：不是「两边都 mask」，是「target 单边 mask」"
           % k1_no)
    record("[G] 新实现把它压回合理量级", abs(k1_new) < 1e2, "%.6f" % k1_new)

    # [D] 新结果必须等于「只在共同支撑集上手工算的 KL」
    keep = [0, 1, 3]                     # 位置 2 因 target 被 mask 而排除
    s_lp = F.log_softmax(s_masked, dim=-1)[0, keep]
    t_lp = F.log_softmax(t_logits, dim=-1)[0, keep]
    manual = float((torch.exp(t_lp) * (t_lp - s_lp)).sum())
    record("[D] 新实现 == 共同支撑集上手工 KL", abs(k1_new - manual) < 1e-6,
           "新 %.8f vs 手工 %.8f（差 %.2e）" % (k1_new, manual, abs(k1_new - manual)))

    # 两侧都 mask（同一位置）
    both = s_logits.clone()
    both[0, 2] = -1e9
    t_both = t_logits.clone()
    t_both[0, 2] = -1e9
    k_both = float(RA.kl([both, t_both]).sum())
    keep2 = [0, 1, 3]
    md = float((torch.exp(F.log_softmax(t_both, dim=-1)[0, keep2])
                * (F.log_softmax(t_both, dim=-1)[0, keep2]
                   - F.log_softmax(both, dim=-1)[0, keep2])).sum())
    record("[D] 两侧同时 mask 同一位置", abs(k_both - md) < 1e-6,
           "新 %.8f vs 手工 %.8f" % (k_both, md))

    # ---------------------------------------------------------------- [E]
    hr("[E] 无 mask 时 kl 与修复前逐位一致（不改变正常路径）")
    worst = 0.0
    for _ in range(20):
        a = torch.randn(16, 64) * 3.0
        b = torch.randn(16, 64) * 3.0
        worst = max(worst, abs(float(RA.kl([a, b]).sum()) - kl_no_valid(a, b)))
    record("[E] 20 组随机输入 max|Δ|", worst < 1e-9, "%.3e" % worst)

    # ---------------------------------------------------------------- [F]
    hr("[F] 全位置被 mask 时 kl == 0，且不出 nan")
    s_all = torch.full((4, 32), -1e9)
    t_all = torch.full((4, 32), -1e9)
    k_all = float(RA.kl([s_all, t_all]).sum())
    record("[F] 全 mask → 0 且非 nan", k_all == 0.0 and not np.isnan(k_all), "%.6f" % k_all)

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 68, flush=True)
    passed = sum(1 for _, p, _ in results if p)
    print("第14轮审计汇总：%d/%d PASS%s"
          % (passed, len(results), "" if ok else "  <-- 有 FAIL"), flush=True)
    for name, p, detail in results:
        if not p:
            print("  FAIL  %s  %s" % (name, detail), flush=True)
    print("==> %s" % ("全部 PASS" if ok else "有 FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
