#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 79 轮（DS）诊断：replay_data_v3 的**特征**解析是否正确。

动机（DS12 78.19 / 79.x）：
  用 v3 数据训出的模型 val acc 0.9663，但实战 killed=0；
  而用旧格式（replay_data_tensor_new_small，1,056,596 维）训出的模型 val acc 只有 0.60，
  实战却有 killed=145。两者最大的差异是 max_entities 512 -> 128（特征维度差 712,704）。
  如果 v3 的特征排布与 `Feature.feature2state` 的解析不一致，
  模型看到的就是错乱状态 —— 训练/验证同源看不出来，实战直接废。

做法：取一帧 features，用 feature2state 解析，检查各字段的 shape 与统计量是否落在
SC2 的合理范围（minimap 0..1/0..255、实体数 <= max_entities、scalar 非全零等）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

from alphastarmini.core.sl.feature import Feature
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Label_Size as LS


def stat(name, t):
    if t is None:
        print("  %-30s None" % name)
        return
    try:
        f = t.float()
        print("  %-30s shape=%-22s min=%8.3f max=%8.3f mean=%8.3f 非零=%.1f%%"
              % (name, str(tuple(t.shape)), float(f.min()), float(f.max()),
                 float(f.mean()), 100.0 * float((f != 0).float().mean())))
    except Exception as e:
        print("  %-30s shape=%s (统计失败: %r)" % (name, tuple(getattr(t, "shape", ())), e))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "./data/replay_data_v3/"
    files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))[:1]
    fp = os.path.join(path, files[0])
    feat, lab = torch.load(fp, map_location="cpu")
    print("文件:", fp)
    print("features.shape:", tuple(feat.shape), " labels.shape:", tuple(lab.shape))
    print("AHP: max_entities=%d max_selected=%d batch=%d seq=%d"
          % (AHP.max_entities, AHP.max_selected, AHP.batch_size, AHP.sequence_length))
    print("LS fields:", LS._fields)
    print("LS values:", [getattr(LS, f) for f in LS._fields])

    for t in (0, 1, feat.shape[0] // 2, feat.shape[0] - 1):
        print("\n===== 第 %d 帧 =====" % t)
        try:
            st = Feature.feature2state(feat[t:t + 1])
        except Exception as e:
            print("  feature2state 失败: %r" % (e,))
            import traceback
            traceback.print_exc()
            return

        for name in ("minimap", "screen", "entities", "scalar_features",
                     "entity_position", "selected_units"):
            stat(name, getattr(st, name, None))

        # 实体数：有多少条实体是非零的
        ent = getattr(st, "entities", None)
        if ent is not None:
            nz = (ent.reshape(-1, ent.shape[-1]).abs().sum(-1) > 0).sum().item() \
                 if ent.dim() >= 2 else 0
            print("  %-30s 非零实体数=%d（<= max_entities %d ? %s）"
                  % ("entities 非零行数", nz, AHP.max_entities, nz <= AHP.max_entities))

    # 标签侧的检查：units 标签有多少非零（战斗/选中单位相关）
    print("\n===== 标签检查 =====")
    at = lab[:, :LS.action_type_encoding]
    print("  action_type one-hot 有效帧: %d / %d" % (int((at.sum(-1) > 0.5).sum()), lab.shape[0]))
    off = LS.action_type_encoding + LS.delay_encoding + LS.queue_encoding
    un = lab[:, off:off + int(AHP.max_selected) * int(AHP.max_entities)]
    print("  units 标签 shape=%s 非零比例=%.2f%% 每帧非零均值=%.2f"
          % (tuple(un.shape), 100.0 * float((un != 0).float().mean()),
             float((un != 0).float().sum(-1).mean())))


if __name__ == "__main__":
    main()
