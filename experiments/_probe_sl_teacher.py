"""T10：用**真实 SL 模型**产 teacher_logits，验证缺陷③ 是否真的会炸。

为什么必须做这一步
------------------
第 9/10 轮的 T9 里，`teacher_logits` 是我用 `torch.zeros` 造的占位，宽度 13
（为了和 target 对齐）。于是：
  - 结构跑通了，但 loss = 1.24e8 完全是假的
  - **无法判断缺陷③ 在真实训练里到底会不会触发**

真实训练里 `rl_vs_inner_bot_mp.py` 写的是 `teacher_logits = player_logits`，
而 `player_logits` 来自 `step_logits -> action_logits_by_state`（`forward`），
units 宽度是 **12**（不是 13）。

所以本脚本用 `model/` 里真实的 SL 权重跑 forward 产 teacher_logits，然后：
  [A] 检查 teacher 的 units 实际宽度（应该是 12）
  [B] 直接喂给 `loss_function`，看是否复现
      `shape '[64, 13, 128]' is invalid for input of size 98304`
  [C] 若复现，说明缺陷③ 是真的（第一步），下一轮再修

这是"确认改动正确"的必要一环：**第 10 轮我只是推测缺陷③ 存在，没有实证。**
"""
import os
import sys
import glob

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from pysc2.env.sc2_env import Race

from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
from alphastarmini.core.rl.rl_utils import get_supervised_agent
from alphastarmini.core.rl.batch_server import FakeRawUnits
from alphastarmini.core.sl.feature import Feature
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")


def main():
    torch.manual_seed(31)
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    print("数据集: %s  帧数: %d" % (f0, feats.shape[0]), flush=True)

    print("\n=== 加载真实 SL 模型 ===", flush=True)
    sl_agent = get_supervised_agent(race=Race.protoss, device=DEV, model_type="sl")
    sl_agent.agent_nn.to(DEV)
    sl_agent.agent_nn.model.eval()
    print("  SL agent 就绪：%s" % type(sl_agent).__name__, flush=True)

    # 用真实 SL 模型跑 forward，拿 teacher_logits
    # step_from_state 的返回：(func_call, action, action_logits, hidden, sun, entity_num)
    st = Feature.feature2state(feats[0:1].float())
    st = type(st)(entity_state=st.entity_state.float().to(DEV),
                  statistical_state=[x.float().to(DEV) for x in st.statistical_state],
                  map_state=st.map_state.float().to(DEV))
    h = torch.zeros(1, 1, 128, device=DEV)
    c = torch.zeros(1, 1, 128, device=DEV)

    # ⚠️ 不能用 step_from_state(obs=None)：它内部 arch_model.forward 会调
    #    get_batch_unit_type_mask(action_type, obs_list)，obs=None 直接
    #    `TypeError: 'NoneType' object is not subscriptable`。
    #    用 batch_server 里现成的 FakeRawUnits 顶替（它只被访问一列 unit_type）。
    obs_list = [{"raw_units": FakeRawUnits(np.zeros(64, dtype=np.int64))}]
    with torch.no_grad():
        teacher = sl_agent.agent_nn.batch_action_logits_by_state(
            st, hidden_state=(h, c), obs_list=obs_list)[0]

    print("\n=== [A] 真实 SL 的 teacher_logits 各 head 宽度 ===", flush=True)
    widths = {}
    for name in ("action_type", "delay", "queue", "units", "target_unit", "target_location"):
        v = getattr(teacher, name)
        widths[name] = tuple(v.shape)
        print("  %-16s %s" % (name, tuple(v.shape)), flush=True)

    units_w = teacher.units.shape[1] if teacher.units.dim() == 3 else None
    print("\n  units 宽度 = %s（target 那边是 %d）"
          % (units_w, AHP.max_selected + 1), flush=True)
    if units_w == AHP.max_selected:
        print("  -> 确认：真实 teacher 的 units 是 %d 宽，比 target 少 1 列" % units_w, flush=True)
        print("     **这就是缺陷③ 的实证**（第 10 轮只是推测）", flush=True)
    else:
        print("  -> 意外：宽度与推断不符，需要重新评估缺陷③", flush=True)

    print("\n==> T10 完成。下一步（第 12 轮）：把真实 teacher 喂进 loss_function 复现，再修 ③",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
