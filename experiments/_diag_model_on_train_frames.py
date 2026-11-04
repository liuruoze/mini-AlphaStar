#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 79 轮（DS）诊断：把**训练数据自己的帧**喂给模型，看它输出什么动作。

为什么要做（DS12 79.8）：
  关闭 Smart_unit 替换后训练出的模型，val action acc 0.9392，
  但实战 19/20 局 killed=0、collected 还降了。
  这有两种完全不同的解释：
    H1「模型塌缩」：模型本身就只会输出 no_op（训练没学好）。
    H2「评测输入不一致」：模型在训练帧上正常，但实战时喂进去的 state 与训练时不同
        （归一化/通道/entity 排列差异），于是退化成 no_op。

  区分方法：用**训练数据本身的帧**（模型见过的分布）做前向，统计输出动作分布。
    · 若输出接近标签分布（有 Smart_unit / Attack_pt / Build / Train）-> H1 不成立，问题在评测输入（H2）
    · 若输出 95%+ 是 no_op -> H1 成立，模型确实塌缩了

用法：
  python experiments/_diag_model_on_train_frames.py --model model/xxx_best.pth --frames 200
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl.feature import Feature
from alphastarmini.lib.hyper_parameters import Label_Size as LS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--path", default="./data/replay_data_v3/")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="从第 start 帧开始取（开局没有攻击，要看中后期用 500）")
    ap.add_argument("--file-index", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ArchModel()
    sd = torch.load(args.model, map_location="cpu")
    miss, unexpected = net.load_state_dict(sd, strict=False)
    net.to(device)
    net.eval()
    print("加载 %s（missing=%d unexpected=%d）" % (args.model, len(miss), len(unexpected)))

    files = sorted(f for f in os.listdir(args.path) if f.endswith(".pt"))
    fp = os.path.join(args.path, files[args.file_index])
    feat, lab = torch.load(fp, map_location="cpu")
    s0 = min(args.start, feat.shape[0] - 1)
    n = min(args.frames, feat.shape[0] - s0)
    print("用 %s 的第 %d..%d 帧（共 %d）" % (fp, s0, s0 + n - 1, feat.shape[0]))

    pred_cnt = collections.Counter()
    gt_cnt = collections.Counter()

    # ⚠️ 第 79 轮修正：hidden 必须在**帧间传递**，否则测的是"单帧 + 零状态"，
    #    与实战（连续几千步的 LSTM 状态）不同 —— 第一版就犯了这个错，
    #    结果"训练帧上输出正常"的结论只在零状态下成立。
    hs = (torch.zeros(1, 1, 128, device=device),
          torch.zeros(1, 1, 128, device=device))

    with torch.no_grad():
        for t in range(s0, s0 + n):
            f = feat[t:t + 1].to(device).float()          # [1, 343892]
            st = Feature.feature2state(f)                 # MsState
            if t == s0:
                # 与评测侧 [STATE-DUMP] 同口径，用于对比"训练输入"与"实战输入"的数值分布
                print("[TRAIN-STATE] entity=%s map=%s scalar_n=%d"
                      % (tuple(st.entity_state.shape), tuple(st.map_state.shape),
                         len(st.statistical_state)))
                print("[TRAIN-STATE] entity absmax=%.3f mean=%.3f nonzero=%.1f%%"
                      % (float(st.entity_state.abs().max()), float(st.entity_state.mean()),
                         100.0 * float((st.entity_state != 0).float().mean())))
            # forward 签名: (state, batch_size, sequence_length, hidden_state,
            #                return_logits, ..., return_baseline)
            # return_logits=True 且 return_baseline=False 时返回
            #   (action_logits, action, hidden_state, select_units_num, entity_nums)
            out = net.forward(st, batch_size=1, sequence_length=1,
                              hidden_state=hs, return_logits=True)
            # 用返回值更新 hidden（与评测侧 `player_memory = tuple(h.detach() ...)` 一致）
            try:
                hs = tuple(h.detach() for h in out[2])
            except Exception:
                pass
            logits = out[0]
            at = getattr(logits, "action_type", None)
            if at is None:
                raise RuntimeError("forward 返回里没有 action_type: %r" % (type(logits),))
            pred_cnt[int(at.argmax(dim=-1).reshape(-1)[0].item())] += 1

            gt = int(lab[t, :LS.action_type_encoding].argmax().item())
            gt_cnt[gt] += 1

    names = {}
    try:
        from pysc2.lib.actions import RAW_FUNCTIONS as F
        for nm in F._func_dict:
            fn = F._func_dict[nm]
            names[int(fn.id.value)] = nm
    except Exception:
        pass

    def show(tag, c):
        print("\n--- %s (n=%d) ---" % (tag, sum(c.values())))
        for aid, k in c.most_common(12):
            print("  %-30s id=%-4d %5d  %6.2f%%" % (names.get(aid, "?"), aid, k,
                                                    100.0 * k / max(sum(c.values()), 1)))

    show("模型输出", pred_cnt)
    show("数据标签", gt_cnt)

    # 关键判据：预测里有多少"非 no_op"
    tot = max(sum(pred_cnt.values()), 1)
    noop = pred_cnt.get(0, 0)
    smart = pred_cnt.get(12, 0)
    atk = pred_cnt.get(2, 0)
    print("\n=== 判据 ===")
    print("  模型输出 no_op 占比: %.1f%%" % (100.0 * noop / tot))
    print("  Smart_unit 占比:     %.1f%%" % (100.0 * smart / tot))
    print("  Attack_pt 占比:      %.1f%%" % (100.0 * atk / tot))
    print("  -> 若 no_op > 90%%：模型塌缩（H1）；若接近标签分布：评测输入不一致（H2）")


if __name__ == "__main__":
    main()
