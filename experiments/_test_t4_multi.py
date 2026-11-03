"""T4：多推理进程（多卡）全量验证。

复用 T3b 的 AsyncBatchInferenceServer，但启动 P 个 group，每个 group：
  - 独占一张卡 cuda:g
  - 内部 fork N 个 actor
  - 自己跑动态批主循环

fork 顺序（关键）：
  主进程 fork group（主进程不碰 CUDA）
  -> group fork actor（group 此时也不碰 CUDA）
  -> group 初始化 CUDA
  -> group 跑 server
这样 actor 子进程里没有 CUDA context，不会踩父进程。

用法: python _test_t4_multi.py <P> <N> <steps>
  P=4 N=24 steps=25  -> 96 个 SC2 实例
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import time
import multiprocessing as mp

import numpy as np
import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

SHAPES = StateShapes(
    entity_shape=(128, 1856), map_shape=(24, 64, 64),
    scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                   (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])


def actor_main(srv, idx, steps, seed, gid, result_q):
    try:
        from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent as SC2Agent, Race, Bot, Difficulty
        from pysc2.lib import actions as A
        from pysc2.lib.features import FeatureUnit
        from alphastarmini.core.arch.agent import Agent
        from alphastarmini.core.rl.action import ArgsAction
        from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP

        torch.set_num_threads(1)
        NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])

        aif = AgentInterfaceFormat(**AAIFP._asdict())
        env = SC2Env(map_name="Simple64",
                     players=[SC2Agent(Race.protoss, "g%dp%d" % (gid, idx)),
                              Bot(Race.random, Difficulty(1))],
                     step_mul=8, agent_interface_format=[aif], random_seed=seed)
        ts = env.reset()[0]
        obs = ts.observation
        action_spec = env.action_spec()
        if isinstance(action_spec, (list, tuple)) and not hasattr(action_spec, "functions"):
            action_spec = action_spec[0]

        done = 0
        fallback = 0
        for t in range(steps):
            if t > 0 and ts.last():
                ts = env.reset()[0]
                obs = ts.observation

            st = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
            ut = np.asarray(obs.raw_units)[:, FeatureUnit.unit_type]
            srv.write_state(idx, st, ut)

            a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n = srv.submit_and_wait(idx)

            action = ArgsAction(action_type=torch.tensor([a_type]), delay=torch.tensor([a_delay]),
                                queue=torch.tensor([a_queue]), units=a_units.reshape(1, -1),
                                target_unit=torch.tensor([a_tunit]),
                                target_location=a_tloc.reshape(1, -1))
            try:
                fc = Agent.action_to_func_call(action, torch.tensor([n]), action_spec)
                ts = env.step([fc])[0]
            except Exception:
                fallback += 1
                ts = env.step([NO_OP])[0]
            obs = ts.observation
            done += 1

        srv.mark_exited(idx)
        env.close()
        result_q.put((gid, idx, done, fallback, ""))
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            srv.mark_exited(idx)
        except Exception:
            pass
        result_q.put((gid, idx, 0, -1, repr(e)))


def run_group(gid, N, steps, result_q):
    """一个 group：独占 cuda:gid，管 N 个 actor。"""
    try:
        srv = AsyncBatchInferenceServer(None, N, torch.device("cuda:%d" % gid), SHAPES,
                                        max_units=128, max_selected=AHP.max_selected,
                                        max_batch=N)
        srv.spawn_actors(actor_main, [(steps, 5000 + gid * 100 + i, gid, result_q) for i in range(N)])

        from alphastarmini.core.arch.agent import Agent
        agent = Agent(); agent.to(torch.device("cuda:%d" % gid)); agent.model.eval()
        srv.agent = agent

        t0 = time.time()
        rounds = srv.run(idle_timeout=45.0)
        wall = time.time() - t0
        # 必须等 actor 子进程把各自结果 put 完再退，否则 daemon 子进程会被直接杀掉，
        # 主进程就收不全 actor 结果（T4 首测只收到 4/16）
        srv.join(timeout=20)
        result_q.put(("GROUP", gid, rounds, wall, srv.async_summary()))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_q.put(("GROUP", gid, -1, -1.0, repr(e)))


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    print("启动 %d 个推理进程 × %d actor = %d 个 SC2 实例, 每 actor %d 步"
          % (P, N, P * N, steps), flush=True)

    result_q = mp.Queue()
    t_start = time.time()
    groups = [mp.Process(target=run_group, args=(g, N, steps, result_q)) for g in range(P)]
    for p in groups:
        p.start()

    actors = []
    group_stats = []
    expected = P * N
    while len(actors) < expected or len(group_stats) < P:
        try:
            item = result_q.get(timeout=120)
        except Exception:
            break
        if item[0] == "GROUP":
            group_stats.append(item)
        else:
            actors.append(item)

    total_wall = time.time() - t_start
    ok = [a for a in actors if a[2] > 0]
    total_steps = sum(a[2] for a in ok)
    fallbacks = sum(a[3] for a in ok if a[3] > 0)

    print("\n==== T4 多卡结果 ====", flush=True)
    for gs in sorted(group_stats, key=lambda x: x[1]):
        print("  group %s: %s  -> %.0f step/s" % (gs[1], gs[4], gs[2] * 0 + 0) if False else "  group %s: %s" % (gs[1], gs[4]), flush=True)
    # 吞吐按各 group 的最长墙钟计（它们并行）
    walls = [g[3] for g in group_stats if g[3] > 0]
    par_wall = max(walls) if walls else total_wall
    print("actor 完成 %d/%d | 总步数 %d | fallback %d" % (len(ok), expected, total_steps, fallbacks), flush=True)
    print("**系统吞吐 %.0f step/s**（按并行墙钟 %.2f s 计）| 启动总耗时 %.1f s"
          % (total_steps / par_wall, par_wall, total_wall), flush=True)
    for a in actors:
        if a[4]:
            print("  actor g%d[%d] 失败: %s" % (a[0], a[1], a[4]), flush=True)
    for p in groups:
        p.join(timeout=15)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
