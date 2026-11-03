"""Hy4 补充实测 [1]：多进程（而非多线程）下 SC2 env 的扩展性。

MiniMax 测出「同进程 8 线程」时 env.step 从 17.4ms 退化到 80ms，但没拆清归因：
是 GIL（进程级隔离可解），还是 CPU/内存带宽争抢（进程级隔离也救不了）。

本脚本用 **N 个独立进程** 各跑 1 个真实 SC2 实例，测稳态 env.step 耗时。
若多进程下仍退化 -> 瓶颈是机器资源，天花板比预期低；
若多进程下稳定在 ~17ms -> GIL 是唯一元凶，进程级隔离成立。
"""
import os
import sys
import time
import multiprocessing as mp

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

from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])
N_STEP = 24


def make_env(idx):
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    return SC2Env(map_name="Simple64",
                  players=[SC2Agent(Race.protoss, "p%d" % idx),
                           Bot(Race.random, Difficulty(1))],
                  step_mul=8, agent_interface_format=[aif], random_seed=idx + 777)


def worker(idx, start_ev, res_q):
    try:
        env = make_env(idx)
        obs = env.reset()[0].observation
        start_ev.wait()                       # 所有实例启动完再一起开跑
        for _ in range(4):                    # 预热
            obs = env.step([NO_OP])[0].observation
        t = time.time()
        for _ in range(N_STEP):
            obs = env.step([NO_OP])[0].observation
        per = (time.time() - t) / N_STEP * 1000
        res_q.put((idx, per))
        env.close()
    except Exception as e:
        res_q.put((idx, -1.0))
        print("worker %d failed: %r" % (idx, e), flush=True)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    start_ev = mp.Event()
    res_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(i, start_ev, res_q)) for i in range(N)]
    for p in procs:
        p.start()

    t_launch = time.time()
    start_ev.set()
    wall_start = time.time()

    got = []
    for _ in range(N):
        got.append(res_q.get(timeout=300))
    wall = time.time() - wall_start

    ok = [v for _, v in got if v > 0]
    for p in procs:
        p.join(timeout=10)

    if not ok:
        print("N=%-3d  ALL_FAILED" % N, flush=True)
        return
    avg = sum(ok) / len(ok)
    # 系统级吞吐：N 个实例并行，每实例 1000/avg 步/秒
    print("N=%-3d  env.step 平均 %6.2f ms  (min %.2f / max %.2f)  |  墙钟 %.1fs  |  "
          "系统合计 %6.0f step/s  |  单实例 %.1f step/s"
          % (N, avg, min(ok), max(ok), wall, N * 1000.0 / avg, 1000.0 / avg), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
