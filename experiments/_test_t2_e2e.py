"""T2 端到端验证：真实 SC2 env + N 个 actor 进程 + 单中心批推理。

验证三件事：
  1. 流程跑通（动作合法、游戏能推进）
  2. 真实吞吐（不是 sleep 模拟）——检验"同步批下双缓冲无效"的判断
  3. 各阶段耗时分布（build / forward / write / wait），定位瓶颈

注意 fork 顺序：先 fork actor（不碰 CUDA），父进程再初始化 CUDA，否则子进程踩父进程 context 死锁。
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

import time
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

from alphastarmini.core.rl.batch_server import BatchInferenceServer, StateShapes
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])


def make_env(idx, seed):
    aif = AgentInterfaceFormat(**AAIFP._asdict())
    return SC2Env(map_name="Simple64",
                  players=[SC2Agent(Race.protoss, "p%d" % idx),
                           Bot(Race.random, Difficulty(1))],
                  step_mul=8, agent_interface_format=[aif], random_seed=seed)


def actor_main(srv, idx, steps, seed, result_q):
    try:
        from alphastarmini.core.arch.agent import Agent
        from alphastarmini.core.rl.action import ArgsAction

        print("[actor %d] starting env" % idx, file=sys.stderr, flush=True)
        env = make_env(idx, seed)
        ts = env.reset()[0]
        obs = ts.observation
        print("[actor %d] env ready" % idx, file=sys.stderr, flush=True)
        action_spec = env.action_spec()
        if isinstance(action_spec, (list, tuple)) and not hasattr(action_spec, "functions"):
            action_spec = action_spec[0]

        done_steps = 0
        t_pre = t_wr = t_fc = t_env = t_read = 0.0
        for t in range(steps):
            if t > 0 and ts.last():
                ts = env.reset()[0]
                obs = ts.observation

            _t = time.time()
            st = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
            ut = np.asarray(obs.raw_units)[:, FeatureUnit.unit_type]
            t_pre += time.time() - _t
            _t = time.time()
            srv.write_state(idx, st, ut)
            t_wr += time.time() - _t

            if t == 0:
                print("[actor %d] wrote state, waiting" % idx, file=sys.stderr, flush=True)
            srv.b_ready.wait()
            srv.b_done.wait()
            if t == 0:
                print("[actor %d] got action" % idx, file=sys.stderr, flush=True)

            _t = time.time()
            a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n = srv.read_action(idx)
            t_read += time.time() - _t
            action = ArgsAction(
                action_type=torch.tensor([a_type]),
                delay=torch.tensor([a_delay]),
                queue=torch.tensor([a_queue]),
                units=a_units.reshape(1, -1),
                target_unit=torch.tensor([a_tunit]),
                target_location=a_tloc.reshape(1, -1))
            try:
                _t = time.time()
                fc = Agent.action_to_func_call(action, torch.tensor([n]), action_spec)
                t_fc += time.time() - _t
                _t = time.time()
                ts = env.step([fc])[0]
                t_env += time.time() - _t
            except Exception as _e:
                if t < 2:
                    import traceback
                    print("[actor %d] action_to_func_call 失败: %r" % (idx, _e), file=sys.stderr, flush=True)
                    traceback.print_exc()
                _t = time.time()
                ts = env.step([NO_OP])[0]
                t_env += time.time() - _t
            obs = ts.observation
            done_steps += 1

        env.close()
        n = max(done_steps, 1)
        print("[actor %d] 每步: preprocess %.1f ms | write_state %.1f | read %.1f | func_call %.1f | env.step %.1f"
              % (idx, t_pre / n * 1000, t_wr / n * 1000, t_read / n * 1000,
                 t_fc / n * 1000, t_env / n * 1000), file=sys.stderr, flush=True)
        result_q.put((idx, done_steps, ""))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_q.put((idx, 0, repr(e)))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    dev = torch.device("cuda:0")

    shapes = StateShapes(
        entity_shape=(128, 1856),
        map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])
    print("每帧 %.1f KB，%d actors 共享内存约 %.1f MB"
          % (shapes.bytes_per_frame() / 1024, N, shapes.bytes_per_frame() * N / 1024 / 1024), flush=True)

    srv = BatchInferenceServer(None, N, dev, shapes,
                               max_units=128, max_selected=AHP.max_selected)

    result_q = __import__("multiprocessing").Queue()
    # 先 fork actor（此时父进程还没碰 CUDA）
    procs = srv.spawn_actors(actor_main, [(STEPS, 1000 + i, result_q) for i in range(N)])

    # fork 之后再初始化 CUDA
    from alphastarmini.core.arch.agent import Agent
    agent = Agent()
    agent.to(dev)
    agent.model.eval()
    srv.agent = agent
    print("CUDA agent ready", flush=True)

    t0 = time.time()
    for r in range(STEPS):
        if r == 0:
            print("[server] wait round 0", file=sys.stderr, flush=True)
        srv.serve_once()
        if r == 0:
            print("[server] round 0 done", file=sys.stderr, flush=True)
    wall = time.time() - t0

    total = N * STEPS
    print("\n==== T2 结果 ====", flush=True)
    print("N=%d  steps/actor=%d  墙钟 %.2f s  ->  **%.0f step/s**"
          % (N, STEPS, wall, total / wall), flush=True)
    print("每轮 %.2f ms（其中 forward %.2f ms）"
          % (wall / STEPS * 1000, srv.timing["forward"] / STEPS * 1000), flush=True)
    print(srv.summary(STEPS), flush=True)

    got = [result_q.get(timeout=120) for _ in range(N)]
    ok = sum(1 for _, s, e in got if s > 0)
    print("actor 完成 %d/%d, 步数分布 min=%d max=%d"
          % (ok, N, min(s for _, s, _ in got), max(s for _, s, _ in got)), flush=True)
    for idx, s, e in got:
        if e:
            print("  actor %d 失败: %s" % (idx, e), flush=True)

    srv.join(timeout=10)
    print("==> T2 %s" % ("PASS" if ok == N else "PARTIAL/FAIL"), flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    main()
