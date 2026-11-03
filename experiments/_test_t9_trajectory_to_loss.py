"""T9 验收：batch acting 产出的数据能不能构成一条**可训练的 Trajectory**。

为什么这一步是"最后一道关"
--------------------------
第八轮 T8 证明了 behavior_logits 能被 `RA.log_prob` 消费、rho 精确等于 1。
但那是在**手工拼的 tensor** 上验的，没有走 `Trajectory` 这个真实数据容器。

`Learner.update_parameters()` 的实际入口是：
    trajectories = self.get_normal_trajectories()
    loss, loss_dict = loss_function(agent, update_trajectories, ...)
即 **list of Trajectory** -> `loss_function`。

所以本脚本做的事情是：用 batch_server 收 T 步，拼成一条 Trajectory，
**直接调 loss_function**，看它能不能算出 loss、梯度能不能回传。
这是"batch acting 能训练"的最终判据。

顺带发现的缺口
--------------
`Trajectory` 需要 `entity_num`，而旧版 batch_server 的 `_serve_batch` 拿到了 `en`
（arch_model.forward 的返回值）却**没有写进共享内存**。
本轮的 T9a 已补上 `a_en` + `read_entity_num()`。

字段来源对照
------------
| Trajectory 字段 | 来自 |
|---|---|
| state | srv._build_state_subset（device 上的 MsState） |
| baseline_state | 同 state 的 statistical_state（本测试用玩家自己的观测冒充） |
| memory | 每步 serve **之前**的 srv.h / srv.c |
| action | srv.read_action() 的 7 元组 |
| behavior_logits | srv.read_behavior_logits() |
| player_select_units_num | srv.read_action() 的 n |
| entity_num | srv.read_entity_num()（本轮新增） |
| masks / unit_type_entity_mask | action_type 决定哪些 head 有效；本测试按"全 1"给，再按 action_type 收敛 |
| reward / is_final | 本测试用假值（不验证 RL 效果，只验证 shape 与可算性） |
"""
import os
import sys
import glob
import contextlib

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
from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
from alphastarmini.core.rl.rl_utils import get_supervised_agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.rl.batch_server import (
    AsyncBatchInferenceServer, StateShapes, FakeRawUnits, LOGIT_FIELDS)
from alphastarmini.core.rl.rl_utils import Trajectory
from alphastarmini.core.rl.action import ArgsAction, ArgsActionLogits
from alphastarmini.core.rl.rl_loss import loss_function
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")
T = AHP.sequence_length        # 8
MAX_SEL = AHP.max_selected     # 12
MAX_ENT = AHP.max_entities     # 128


@contextlib.contextmanager
def deterministic_sampling():
    """固定采样路径（argmax），避免自回归随机性影响可复现性。"""
    orig = torch.multinomial

    def _argmax(input, num_samples=1, replacement=False):
        idx = input.argmax(dim=-1, keepdim=True)
        if int(num_samples) > 1:
            idx = idx.expand(-1, int(num_samples))
        return idx

    torch.multinomial = _argmax
    try:
        yield
    finally:
        torch.multinomial = orig


def load_feats(n):
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    return torch.load(f0)[0][:n].float()


def make_server(agent, n=1):
    shapes = StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])
    srv = AsyncBatchInferenceServer(agent, n, DEV, shapes,
                                    max_units=128, max_selected=MAX_SEL,
                                    max_batch=n, return_logits=True)
    srv.agent = agent
    return srv


# action_type -> 该动作类型下哪些 head 有意义。
# 这里不追求与游戏规则完全一致（那是 env 的事），只保证 units/target 的 mask 不会把
# 有效位置误屏蔽。所以按"给定 action_type 时至少有 units 可用"来给 mask。
def make_action(a):
    """把 read_action 的 7 元组转成 ArgsAction，**形状严格对齐真实 actor 的产物**。

    _probe_logits.py 实测的单样本形状（B=1）：
        action_type (1,1)   delay (1,1)   queue (1,1)
        units (1,12,1)      target_unit (1,1,1)   target_location (1,2)

    ⚠️ 第一版我把 units 写成 (1,1,n)（n=有效单位数，各步不同），
    target_unit 写成 (1,1)、target_location 写成 (1,1,2) —— 全部错。
    rl_unroll 里 `torch.cat([...], dim=0)` 立刻报
    `Sizes of tensors must match except in dimension 2. Got 2 and 1`。
    教训：**形状要抄实测值，不要按语义自己推**。
    """
    units = torch.zeros(1, MAX_SEL, 1, dtype=torch.int64, device=DEV)
    n = min(int(a[6]), MAX_SEL)
    if n > 0:
        units[0, :n, 0] = a[3][:n].to(DEV)
    return ArgsAction(
        action_type=torch.tensor([[a[0]]], dtype=torch.int64, device=DEV),
        delay=torch.tensor([[a[1]]], dtype=torch.int64, device=DEV),
        queue=torch.tensor([[a[2]]], dtype=torch.int64, device=DEV),
        units=units,
        target_unit=torch.tensor([[[a[4]]]], dtype=torch.int64, device=DEV),
        target_location=a[5].reshape(1, 2).to(DEV))


def zero_logits(shapes):
    """构造一个全零的 ArgsActionLogits，用于 teacher_logits 占位。

    ⚠️ 必须是 ArgsActionLogits，不能是 tensor：
       human_policy_kl_loss 里直接 `getattr(trajectories.teacher_logits, field)`，
       给 tensor 会报 `AttributeError: 'Tensor' object has no attribute 'action_type'`。
    真实训练里 teacher_logits 来自 SL 模型的前向（get_supervised_agent），
    本测试没有 SL 模型，用零 logits 只为走通 shape（KL 项会是常数，不影响本次验收目标）。
    """
    # ⚠️ units 要 **max_selected + 1**（13）宽，而不是 forward 输出的 12 宽。
    #    因为 target_logits 来自 rl_unroll -> mimic_forward，它的 units 就是 13 宽
    #    （比 forward 多一列 padding）；KL 要比较两个同宽分布，teacher 必须对齐。
    #    第 9 轮我把 `+1` 当成既有缺陷改掉了，实测报
    #    `invalid for input of size 106496`（= 64*13*128）才发现 target 本来就是 13。
    d = {}
    for k, v in shapes.items():
        if k == "units":
            d[k] = torch.zeros(1, MAX_SEL + 1, MAX_ENT, device=DEV)
        else:
            d[k] = torch.zeros(1, *v, device=DEV)
    return ArgsActionLogits(**d)


# baseline_state 的 8 项索引（来自 arch/agent.py:167 get_baseline_state_from_multi_source_state）
#   statistical_state 的 16 项里挑这 7 个，再追加 cumulative_score（来自 obs['score_cumulative']）
BASELINE_INDEX = [0, 3, 7, 9, 10, 11, 12]
CUMULATIVE_SCORE_DIM = 13


def make_baseline_state(st):
    """构造 winloss_baseline 需要的 8 项。

    ⚠️ 第一版我直接把完整的 statistical_state（16 项）当 baseline_state 传进去，
    baseline.py:83 的 `[a,b,c,d,e,f,g,h] = various_observations` 立刻报
    `ValueError: too many values to unpack (expected 8)`。
    真实路径是 arch/agent.py:138 的 get_baseline_state_from_multi_source_state()，
    它只挑 7 项 + 从 obs['score_cumulative'] 追加 1 项。

    本测试没有真实 pysc2 obs，所以 cumulative_score 用 0 填充
    （它只进 baseline 网络，不影响 units/action 的 log_prob，故不干扰本次验收）。
    """
    bs = [st.statistical_state[i] for i in BASELINE_INDEX]
    bs.append(torch.zeros(1, CUMULATIVE_SCORE_DIM, device=DEV))
    return bs


def build_masks(action_types):
    """返回 masks 与 unit_type_entity_mask，**类型必须是 python list / numpy**。

    真实来源（rl_vs_inner_bot_mp.py:360 与 rl_utils.py:160）：
      masks               = U.get_mask(action, action_spec)  -> [1,1,0,0,0,0] 这样的 6 元 list（每步一个）
      unit_type_entity_mask = RU.get_unit_type_mask(action, obs) -> 1D numpy（长度 max_entities）

    ⚠️ 第一版我用 torch.tensor 给这两个字段，namedtuple_zip 之后会 np.array(...)，
       tensor 变成 object 数组 -> `TypeError: can't convert np.ndarray of type numpy.object_`。
       所以这里保持 list / numpy 的原始类型。

    head 顺序（ACTION_FIELDS）：0 action_type, 1 delay, 2 queue, 3 units, 4 target_unit, 5 target_location
    本测试 6 个全给 1，确保所有 head 的 loss 项都参与计算（不验证 mask 的策略语义）。
    """
    masks = [[1, 1, 1, 1, 1, 1] for _ in range(len(action_types))]
    utem = [np.ones(MAX_ENT, dtype=np.float32) for _ in range(len(action_types))]
    return masks, utem


def main():
    torch.manual_seed(23)
    feats = load_feats(T)
    # ⚠️ 这里必须用 AlphaStarAgent 而不是 Agent：
    #    loss_function 第一步就是 agent.rl_unroll(...)，而 rl_unroll 定义在
    #    core/rl/alphastar_agent.py 的 AlphaStarAgent 上；core/arch/agent.py 的 Agent
    #    只是网络封装，没有这个方法。（第一版用 Agent 直接报 AttributeError。）
    as_agent = AlphaStarAgent(name="t9")
    as_agent.agent_nn.to(DEV)
    # ⚠️ 必须 train()，不能 eval()：cuDNN 的 RNN backward 只允许在 training mode 下调用，
    #    否则报 `cudnn RNN backward can only be called in training mode`。
    #    注意 learner.py:218 把 `agent.agent_nn.model.train()` **注释掉了**，
    #    真实训练走到 backward 会被同一个错误拦住（见本轮日志 §4 第 3 条既有缺陷）。
    as_agent.agent_nn.model.train()
    # RL 训练时 action_type head 会额外加 mask，这里打开以贴近真实训练
    as_agent.set_rl_training(True)
    agent = as_agent.agent_nn      # server 用同一个网络实例

    srv = make_server(agent, n=1)
    ok = True

    # 可选：绕过 KL 蒸馏分支。
    # 原因：rl_loss.py:113 的 get_kl_or_entropy 对 units 按 max_selected+1（13）reshape，
    # 而 units 的 logits 实际宽度是 max_selected（12）——
    # get_logprob_and_rhos 那条路会先调 change_units_and_logits() 补一列 padding，
    # 而 KL 这条路**没有补**，于是必然 `shape '[64, 13, 128]' is invalid for input of size 98304`。
    # 这是既有代码的缺陷（与 batch acting 无关）。跳过它，才能验证 VTrace/UPGO/entropy
    # 这几部分的数据链路与反向传播是否通。
    skip_kl = os.environ.get("MAS_SKIP_KL", "0") == "1"
    if skip_kl:
        import alphastarmini.core.rl.rl_loss as _RL
        _z = torch.zeros((), device=DEV, requires_grad=True)
        _RL.human_policy_kl_loss = lambda *a, **k: (_z, {"action_type": 0.0})
        _RL.human_policy_kl_loss_action = lambda *a, **k: _z
        print("⚠️ MAS_SKIP_KL=1：已绕过 human_policy_kl_loss（其 units 维度不一致，见注释）",
              flush=True)

    with deterministic_sampling():
        # ---------- 收 T 步 ----------
        print("=== 收集 %d 步（server 批前向 + logits + entity_num）===" % T, flush=True)
        # teacher_logits：默认用**真实 SL 模型**产出（第 12 轮）。
        # 这是缺陷③ 的验证前提——零占位是 13 宽（手动对齐），真实 forward 是 12 宽。
        use_sl = os.environ.get("MAS_TEACHER", "sl") == "sl"
        if use_sl:
            sl_agent = get_supervised_agent(race=Race.protoss, device=DEV, model_type="sl")
            sl_agent.agent_nn.to(DEV)
            sl_agent.agent_nn.model.eval()
            print("  teacher 来源：真实 SL 模型（units 应为 12 宽，验证缺陷③ 的修复）", flush=True)
        else:
            sl_agent = None
            print("  teacher 来源：零占位（13 宽，仅用于对照）", flush=True)
        th = torch.zeros(1, 1, 128, device=DEV)
        tc = torch.zeros(1, 1, 128, device=DEV)

        states, memories, actions, logits_list, suns, ens = [], [], [], [], [], []
        teacher_list = []
        for t in range(T):
            st = Feature.feature2state(feats[t:t + 1])
            ut = np.zeros(8, dtype=np.int64)
            srv.write_state(0, st, ut)

            # memory 必须是**进入这一步时**的 hidden（actor.py 里 memory=player_memory 的语义）
            h_before = (srv.h[:, 0:1, :].clone().to(DEV), srv.c[:, 0:1, :].clone().to(DEV))

            srv._serve_batch([0])
            a = srv.read_action(0)
            bl = srv.read_behavior_logits(0)
            en = srv.read_entity_num(0)

            st_dev = srv._build_state_subset(torch.tensor([0]), 1)
            states.append(st_dev)

            if use_sl:
                _ol = [{"raw_units": FakeRawUnits(np.zeros(64, dtype=np.int64))}]
                with torch.no_grad():
                    _tl, _, _nh, _, _ = sl_agent.agent_nn.batch_action_logits_by_state(
                        st_dev, hidden_state=(th, tc), obs_list=_ol)
                teacher_list.append(_tl)
                th, tc = _nh[0], _nh[1]
            memories.append(h_before)
            actions.append(make_action(a))
            logits_list.append(bl)
            suns.append(torch.tensor([a[6]], dtype=torch.int64, device=DEV))
            ens.append(torch.tensor([en], dtype=torch.int64, device=DEV))

        print("  entity_num: %s" % [int(e[0]) for e in ens], flush=True)
        print("  select_units_num: %s" % [int(s[0]) for s in suns], flush=True)
        print("  action_type: %s" % [int(a.action_type.reshape(-1)[0]) for a in actions], flush=True)
        if all(int(e[0]) == 0 for e in ens):
            print("  ⚠️ 全部 entity_num = 0，可能有问题（真实状态应有单位）", flush=True)
            ok = False

        # ---------- 组 Trajectory ----------
        masks, utem = build_masks([int(a.action_type.reshape(-1)[0]) for a in actions])
        traj = Trajectory(
            state=states,
            baseline_state=[make_baseline_state(s) for s in states],
            # 与 rl_vs_inner_bot_mp.py:472/474 完全一致的 None（缺陷① 已在 rl_utils 里修好，
            # namedtuple_zip 现在会把 None 字段填成 [None]*seq_len，不再整条断掉）。
            baseline_state_op=None,
            memory=memories,
            z=None,
            # is_final / reward 必须是 **python 标量 list**，不能是 tensor。
            # 真实来源：is_final = home_next_obs.last()（bool）、reward = home_next_obs.reward（float）。
            # 若给 tensor，namedtuple_zip 会按第 0 维切片来 zip，语义完全错位；
            # 随后 np.array(...) 得到 object 数组 ->
            # `TypeError: can't convert np.ndarray of type numpy.object_`。
            is_final=[False] * T,
            masks=masks,
            unit_type_entity_mask=utem,
            action=actions,
            behavior_logits=logits_list,
            # 真实 SL 的 teacher 是 12 宽（缺陷③ 要修的就是这个）；零占位走 13 宽对照
            teacher_logits=(teacher_list if use_sl else [zero_logits(srv.logit_shapes)] * T),
            reward=[0.0] * T,
            player_select_units_num=suns,
            entity_num=ens,
            build_order=None,          # 真实值来自 calculate_build_order；本测试走 no_replay_learn
            z_build_order=None,
            unit_counts=None,
            z_unit_counts=None,
            game_loop=[100] * T,       # 真实值是 int（秒/22.4）
            last_list=[[0, 0, 0]] * T, # [last_delay, last_action_type, last_repeat_queued]
        )
        print("\n=== Trajectory 组装完成 ===", flush=True)
        print("  len(state)=%d  len(action)=%d  len(memory)=%d"
              % (len(traj.state), len(traj.action), len(traj.memory)), flush=True)

        # ---------- 调 loss_function ----------
        print("\n=== 调 loss_function（use_opponent_state=False, no_replay_learn=True）===",
              flush=True)
        try:
            # ⚠️ 必须传 AHP.batch_size 条轨迹，不是 1 条。
            #    rl_loss.py:215 transpose_target_logits 里写死了
            #        logits.view(AHP.sequence_length, AHP.batch_size, ...)
            #    只传 1 条时 rl_unroll 产出 [8,1,564]（4512 个），而 view(8,8,564) 要 36096 个 ->
            #    `shape '[8, 8, 564]' is invalid for input of size 4512`。
            #    这是**既有训练代码的硬约束**（loss 只按整 batch 计算），不是 batch acting 的问题。
            #    本测试复制同一份轨迹 8 次，只为验证 shape 链路；真实训练里是 8 条不同轨迹。
            n_traj = AHP.batch_size
            print("  传 %d 条轨迹（AHP.batch_size）" % n_traj, flush=True)
            loss, loss_dict = loss_function(
                as_agent, [traj] * n_traj,
                use_opponent_state=False,
                no_replay_learn=True,
                only_update_baseline=False,
                learner_baseline_weight=1)
            print("  PASS：loss = %.6f（finite=%s）" % (float(loss), bool(torch.isfinite(loss))),
                  flush=True)
            for k, v in loss_dict.items():
                print("    %-40s %s" % (k, v), flush=True)

            # ---------- 梯度能否回传 ----------
            print("\n=== 反向传播 ===", flush=True)
            loss.backward()
            no_grad, with_grad = [], []
            for name, prm in as_agent.agent_nn.model.named_parameters():
                if prm.grad is None:
                    no_grad.append(name)
                else:
                    with_grad.append((name, float(prm.grad.abs().max())))
            print("  有梯度的参数：%d 个；无梯度：%d 个" % (len(with_grad), len(no_grad)), flush=True)
            for name, g in with_grad[:8]:
                print("    %-52s |grad|max = %.3e" % (name, g), flush=True)

            # ---------- 第 16 轮：按模块分组统计无梯度参数 ----------
            # 目的：区分「设计如此」（FIELDS_WEIGHT 里该 head 权重为 0）与「真 bug」。
            # 只看总数（364 有 / 386 无）没有信息量，必须落到具体模块。
            if os.environ.get("MAS_GRAD_AUDIT", "0") == "1":
                import collections
                groups = collections.OrderedDict()
                for name, prm in as_agent.agent_nn.model.named_parameters():
                    mod = name.split(".")[0]
                    if name.startswith("core") or name.startswith("lstm"):
                        mod = "core"
                    g = groups.setdefault(mod, [0, 0, 0])   # [总数, 有梯度, 无梯度]
                    g[0] += 1
                    if prm.grad is None:
                        g[2] += 1
                    else:
                        g[1] += 1
                print("\n  === 按模块分组（第 16 轮新增）===", flush=True)
                print("  %-26s %6s %8s %8s" % ("module", "total", "with_grad", "no_grad"), flush=True)
                for mod, (tot, wg, ng) in groups.items():
                    flag = "   <-- 全部无梯度" if wg == 0 else ""
                    print("  %-26s %6d %8d %8d%s" % (mod, tot, wg, ng, flag), flush=True)
                print("\n  无梯度参数明细（前 24 个）：", flush=True)
                for nm in no_grad[:24]:
                    print("    %s" % nm, flush=True)
                if len(no_grad) > 24:
                    print("    ... 其余 %d 个" % (len(no_grad) - 24), flush=True)
            if not with_grad:
                print("  FAIL：所有参数都没有梯度，loss 没有连到网络", flush=True)
                ok = False
            elif any(g == 0.0 for _, g in with_grad):
                print("  ⚠️ 有参数梯度为 0（可能被 mask 掉了，先记录）", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("\n  FAIL：loss_function 抛 %s" % type(e).__name__, flush=True)
            ok = False

    print("\n==> T9 结论：%s" % (
        "batch acting 的数据可构成可训练 Trajectory，loss + backward 全通"
        if ok else "未通过，见上面输出"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
