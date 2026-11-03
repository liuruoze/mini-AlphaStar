"""MiniMax 补充实测：actor 线程侧的 CPU 开销。

问题：batch acting 之后，actor 线程不再自己做前向，但仍要做
  (a) preprocess_state_all  —— 纯 Python/numpy，GIL 敏感
  (b) env.step             —— pysc2 两次 gRPC 往返
如果 (a) 在多线程下被 GIL 串行化，它会成为新瓶颈，batch 推理的收益会被吃掉。

测三件事：
  [1] 单线程 preprocess_state_all 耗时
  [2] K 线程并发下的每轮墙钟（暴露 GIL 串行化）
  [3] env.step 单次耗时（no_op + 真实 action 各测一次）
"""
import os
import sys
import time
import threading

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent as SC2Agent, Race, Bot, Difficulty
from pysc2.lib import actions as A

import param as P
P.on_server = True

from alphastarmini.core.arch.agent import Agent as AASAgent
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])
res_lock = threading.Lock()


def make_env(idx):
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    return SC2Env(map_name="Simple64",
                   players=[SC2Agent(Race.protoss, "p%d" % idx),
                            Bot(Race.random, Difficulty(1))],
                   step_mul=8, agent_interface_format=[aif], random_seed=idx)


def worker(idx, n_pre, n_step, res):
    try:
        env = make_env(idx)
        obs = env.reset()[0].observation

        # --- preprocess 单次耗时（串行测量）---
        for _ in range(3):
            AASAgent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
        t = time.time()
        for _ in range(n_pre):
            AASAgent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
        pre_serial = (time.time() - t) / n_pre

        # --- env.step 耗时 ---
        ts = []
        for _ in range(n_step):
            t = time.time()
            env.step([NO_OP])
            ts.append(time.time() - t)
        step_avg = sum(ts) / len(ts)

        # --- 并发下的 preprocess 墙钟（每线程各自跑 n_pre 次，取总墙钟）---
        t0 = time.time()
        for _ in range(n_pre):
            AASAgent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
        wall_pre = (time.time() - t0) / n_pre

        env.close()
        with res_lock:
            res.append(dict(idx=idx, pre_serial=pre_serial, pre_wall=wall_pre, step=step_avg))
    except Exception as e:
        with res_lock:
            res.append(dict(idx=idx, err=str(e)[:160]))


def run(K, n_pre=20, n_step=15):
    res = []
    ths = [threading.Thread(target=worker, args=(i, n_pre, n_step, res)) for i in range(K)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    total_wall = time.time() - t0

    ok = [r for r in res if "err" not in r]
    bad = [r for r in res if "err" in r]
    if not ok:
        print("K=%d 全部失败: %s" % (K, bad[:1]), flush=True)
        return

    pre_serial = min(r["pre_serial"] for r in ok)
    pre_wall = max(r["pre_wall"] for r in ok)
    step_avg = sum(r["step"] for r in ok) / len(ok)
    # 并发扩展性：K 个线程同跑一轮 preprocess，理想墙钟 ≈ 单次耗时；实际 = pre_wall
    scaling = pre_serial / pre_wall if pre_wall > 0 else 0

    print("K=%-3d 成功%2d/%d | preprocess 单线程 %6.2f ms | 并发下每轮墙钟 %6.2f ms "
          "| GIL 扩展性 %.2fx (理想 %.2f) | env.step %5.2f ms | 总墙钟 %.1fs"
          % (K, len(ok), K, pre_serial * 1000, pre_wall * 1000, scaling, K,
             step_avg * 1000, total_wall), flush=True)
    for b in bad[:2]:
        print("    失败: %s" % b["err"], flush=True)


if __name__ == "__main__":
    for K in (1, 8, 32):
        run(K)
