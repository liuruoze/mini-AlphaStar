"""T3a 诊断：把 preprocess 与 forward 的真实开销逐段拆开。

背景（T2 实测）：真实 env 下
  preprocess_state_all  42.6~69.6 ms   （bench 报 9.6 ms）
  center forward (N=8)  166.5 ms       （bench 报 30.9 ms）

两个怀疑：
  A. bench 的 9.6 ms 是「同一个 obs 反复调用」，缓存命中；真实每步 obs 不同 -> 分配+计算都变慢
  B. forward 变慢来自 obs_list 非 None（get_batch_unit_type_mask 是纯 Python 逐样本循环）
     以及三段共享内存 -> CUDA 非 pinned

本脚本在单进程里（无多 actor 争抢）分别量化：
  [1] preprocess 三步分解 + np.asarray(raw_units) 的时间
  [2] forward: obs_list=None vs 真实 obs_list，N=1/4/8
  [3] 三段 H2D vs 一次性 feature 向量 H2D
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

import numpy as np
import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent as SC2Agent, Race, Bot, Difficulty
from pysc2.lib import actions as A
from pysc2.lib.features import FeatureUnit

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.rl.state import MsState
from alphastarmini.core.rl.batch_server import FakeRawUnits
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])
STEPS = 12


def make_env(idx=0):
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    return SC2Env(map_name="Simple64",
                  players=[SC2Agent(Race.protoss, "p%d" % idx),
                           Bot(Race.random, Difficulty(1))],
                  step_mul=8, agent_interface_format=[aif], random_seed=idx + 4242)


def main():
    dev = torch.device("cuda:0")
    agent = Agent()
    agent.to(dev)
    net = agent.model.eval()

    env = make_env(0)
    ts = env.reset()[0]
    obs = ts.observation
    print("%-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s" %
          ("step", "raw_ut", "entity", "spatial", "scalar", "torch", "total", "n_units"), flush=True)

    acc = {"ut": 0.0, "ent": 0.0, "spa": 0.0, "sca": 0.0, "tch": 0.0, "tot": 0.0}
    last_state = None
    for s in range(STEPS):
        if ts.last():
            ts = env.reset()[0]
            obs = ts.observation

        t0 = time.time()
        ut = np.asarray(obs.raw_units)[:, FeatureUnit.unit_type]
        t_ut = time.time() - t0

        t0 = time.time()
        be, ep = Agent.preprocess_state_entity_numpy(obs, return_entity_pos=True)
        t_ent = time.time() - t0

        t0 = time.time()
        map_data = Agent.preprocess_state_spatial_numpy(obs, entity_pos_list=ep)
        t_spa = time.time() - t0

        t0 = time.time()
        scalar_list = Agent.preprocess_state_scalar_numpy(obs, build_order=[], last_list=[0, 0, 0])
        t_sca = time.time() - t0

        t0 = time.time()
        be_t = torch.tensor(be)
        sl_t = [torch.tensor(x) for x in scalar_list]
        mp_t = torch.tensor(map_data)
        st = MsState(entity_state=be_t, statistical_state=sl_t, map_state=mp_t)
        t_tch = time.time() - t0
        last_state = st

        t_tot = t_ut + t_ent + t_spa + t_sca + t_tch
        for k, v in zip(["ut", "ent", "spa", "sca", "tch"], [t_ut, t_ent, t_spa, t_sca, t_tch]):
            acc[k] += v
        acc["tot"] += t_tot

        if s % 3 == 0 or s == STEPS - 1:
            print("%-6d %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8d" %
                  (s, t_ut * 1000, t_ent * 1000, t_spa * 1000, t_sca * 1000,
                   t_tch * 1000, t_tot * 1000, ut.shape[0]), flush=True)

        ts = env.step([NO_OP])[0]
        obs = ts.observation

    print("\n--- 平均（%d 步，单进程无争抢）---" % STEPS)
    for k, label in [("ut", "np.asarray(raw_units)"), ("ent", "entity"), ("spa", "spatial"),
                     ("sca", "scalar"), ("tch", "torch.tensor×3"), ("tot", "preprocess 合计")]:
        print("  %-22s %6.2f ms" % (label, acc[k] / STEPS * 1000))

    # ---------- forward 对比 ----------
    print("\n--- forward 对比（同一份 state 复制 N 份）---", flush=True)
    ut = np.asarray(obs.raw_units)[:, FeatureUnit.unit_type]

    def dup_state(n):
        # 注意：statistical_state 原始有 float64 项，模型要 float32，必须显式转
        return MsState(entity_state=last_state.entity_state.float().repeat(n, 1, 1),
                       statistical_state=[x.float().repeat(n, *([1] * (x.dim() - 1)))
                                          for x in last_state.statistical_state],
                       map_state=last_state.map_state.float().repeat(n, 1, 1, 1))

    action_type_ref = None
    for n in [1, 4, 8]:
        st_cpu = dup_state(n)
        st = MsState(entity_state=st_cpu.entity_state.to(dev),
                     statistical_state=[x.to(dev) for x in st_cpu.statistical_state],
                     map_state=st_cpu.map_state.to(dev))
        h = torch.zeros(1, n, 128, device=dev)
        c = torch.zeros(1, n, 128, device=dev)

        # 先跑一次拿 action_type，供 mask 使用
        with torch.no_grad():
            al, acts, nh, sun, en = agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
            ats = acts.action_type.reshape(-1).cpu().numpy()

        # (a) obs_list=None
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            with torch.no_grad():
                agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
        torch.cuda.synchronize(); t_none = (time.time() - t0) / 3

        # (b) 真实 obs_list
        fake_obs = [{"raw_units": FakeRawUnits(ut)} for _ in range(n)]
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            with torch.no_grad():
                agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=fake_obs)
        torch.cuda.synchronize(); t_obs = (time.time() - t0) / 3

        # (c) 只测 mask 本身
        from alphastarmini.lib import utils as L
        t0 = time.time()
        for _ in range(3):
            L.get_batch_unit_type_mask(torch.from_numpy(ats), fake_obs)
        t_mask = (time.time() - t0) / 3

        print("  N=%-2d  obs_list=None %6.1f ms | obs_list 真实 %6.1f ms | 差 %5.1f ms | 其中 mask %5.1f ms"
              % (n, t_none * 1000, t_obs * 1000, (t_obs - t_none) * 1000, t_mask * 1000), flush=True)

    # ---------- CPU 调度 vs GPU 执行 ----------
    print("\n--- CPU 提交时间 vs GPU 执行时间（return_logits=True）---", flush=True)
    for n in [1, 8]:
        st_cpu = dup_state(n)
        st = MsState(entity_state=st_cpu.entity_state.to(dev),
                     statistical_state=[x.to(dev) for x in st_cpu.statistical_state],
                     map_state=st_cpu.map_state.to(dev))
        h = torch.zeros(1, n, 128, device=dev)
        c = torch.zeros(1, n, 128, device=dev)
        for _ in range(2):
            with torch.no_grad():
                agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            with torch.no_grad():
                agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
        t_submit = (time.time() - t0) / 3          # 只算 CPU 提交（未同步）
        torch.cuda.synchronize()
        t_total = (time.time() - t0) / 3           # 含 GPU 完成
        print("  N=%-2d  CPU 提交(异步) %6.1f ms | 含 GPU 完成 %6.1f ms | GPU 净 %5.1f ms | CPU 占比 %.0f%%"
              % (n, t_submit * 1000, t_total * 1000, (t_total - t_submit) * 1000,
                 t_submit / t_total * 100), flush=True)

    # ---------- H2D 对比 ----------
    print("\n--- H2D 传输（N=8）---", flush=True)
    st_cpu = dup_state(8)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        _ = st_cpu.entity_state.to(dev)
    torch.cuda.synchronize(); t_ent = (time.time() - t0) / 3

    feat = torch.randn(8, 343892)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        _ = feat.to(dev)
    torch.cuda.synchronize(); t_feat = (time.time() - t0) / 3
    print("  三段中的 entity 段 %.2f ms (%.1f MB) | 一次性 feature 向量 %.2f ms (%.1f MB)"
          % (t_ent * 1000, st_cpu.entity_state.numel() * 4 / 1024 / 1024,
             t_feat * 1000, feat.numel() * 4 / 1024 / 1024), flush=True)

    env.close()


if __name__ == "__main__":
    main()
