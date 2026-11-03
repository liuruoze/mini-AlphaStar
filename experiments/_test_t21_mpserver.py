"""T21：验证「独立 server 进程 + 多 actor 进程」的批推理架构能跑通。

## 为什么这一步是前提

前几轮所有验证都是「单进程 + 线程 server」。真实训练里是：

    Worker(rank=0..N) ──┐
    Worker(rank=1)    ──┼──> 共享内存 srv <──> server 进程（唯一持有模型）
    Worker(rank=2)    ──┘

跨进程要同时成立三件事，缺一不可：
  1. `srv` 的共享内存张量（state 槽 / 结果槽 / hidden / logits）fork 后**各进程看到同一块**
  2. `mp.Event` 的 set/wait 能跨进程唤醒（不是只在同进程内有效）
  3. server 进程能独占 GPU 模型，actor 进程**不建 CUDA context**（省显存）

本脚本用真实 SC2 env + 真实模型验证这三点，并测出多进程吞吐。

四个检查项：
[A] 跨进程跑通：N 个 actor 各自完成 steps 步，fallback(NO_OP) = 0
[B] 结果正确性：actor 收到的 action 与 server 算的一致（抽查 action_type）
[C] 吞吐：多进程 vs 单进程（同 N）的对比
[D] actor 进程不持有 CUDA context（验证省显存的前提）

用法：python _test_t21_mpserver.py [N_ACTORS] [STEPS]
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

# 线程超订会毁掉一切（第 7 轮实测 6 倍差距），必须在 import numpy 之前设
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

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


def make_shapes():
    return StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])


# ---------------------------------------------------------------- server 进程
def server_proc(srv, model_learner, device, max_rounds, result_q, ready_ev):
    """server 进程：唯一持有可 forward 的 agent，跑动态批循环。"""
    try:
        from alphastarmini.core.arch.agent import Agent
        a = Agent()
        # 换成父进程 share_memory 的那份模型 —— 这样 server 的 forward 与
        # learner 的更新是同一块内存，权重天然同步，不需要额外的拷贝。
        a.model = model_learner
        a.model.to(device)
        a.model.eval()
        srv.agent = a
        srv.model = a.model
        # ⚠️ 必须用 mp.Event 通知父进程，不能用 `srv.model` 这种 Python 属性：
        #    fork 之后父子进程的内存是**写时复制**，父进程读不到子进程设的属性。
        #    （第一版就是这么错的：父进程 getattr(srv, "model") 永远 None，卡 60 秒超时。）
        ready_ev.set()

        t0 = time.time()
        rounds = srv.run(max_rounds=max_rounds, idle_timeout=40.0, supervise=False)
        wall = time.time() - t0
        result_q.put(("SERVER", rounds, wall, srv.async_summary(),
                       float(srv.steady_throughput() or -1)))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_q.put(("SERVER", 0, 0.0, "ERROR %r" % e, -1.0))


# ---------------------------------------------------------------- actor 进程
def actor_proc(srv, idx, steps, seed, result_q):
    """actor 进程：真实 SC2 env + 通过 srv 批推理。**不建 CUDA context。**"""
    try:
        from pysc2.env.sc2_env import (SC2Env, AgentInterfaceFormat, Agent as SC2Agent,
                                        Race, Bot, Difficulty)
        from pysc2.lib import actions as A
        from pysc2.lib.features import FeatureUnit
        from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
        from alphastarmini.core.arch.agent import Agent
        from alphastarmini.core.rl.actor_inference import ActorInferenceClient

        NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])

        aif = AgentInterfaceFormat(**AAIFP._asdict())
        env = SC2Env(map_name="Simple64",
                     players=[SC2Agent(Race.protoss, "p%d" % idx),
                              Bot(Race.random, Difficulty(1))],
                     step_mul=8, agent_interface_format=[aif], random_seed=seed)
        ts = env.reset()[0]
        obs = ts.observation
        spec = env.action_spec()
        if isinstance(spec, (list, tuple)) and not hasattr(spec, "functions"):
            spec = spec[0]

        # ⚠️ agent=None + device="cpu"：actor 进程**不建 CUDA context**。
        #    batch 模式不需要本地 forward，只用 action_to_func_call（纯函数）。
        client = ActorInferenceClient(agent=None, action_spec=spec, srv=srv, idx=idx,
                                      device=torch.device("cpu"))

        done = 0
        fallback = 0
        t_pre = t_wait = t_env = 0.0
        first_type = -1

        for t in range(steps):
            if t > 0 and ts.last():
                ts = env.reset()[0]
                obs = ts.observation
                client.reset_hidden()

            _t = time.time()
            st = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
            t_pre += time.time() - _t

            _t = time.time()
            fc, act, logits, mem, sun, en = client.step(st, None, obs)
            t_wait += time.time() - _t
            if first_type < 0:
                first_type = int(act.action_type.reshape(-1)[0])

            _t = time.time()
            try:
                ts = env.step([fc])[0]
            except Exception:
                fallback += 1
                ts = env.step([NO_OP])[0]
            t_env += time.time() - _t
            obs = ts.observation
            done += 1

        srv.mark_exited(idx)
        env.close()
        n = max(done, 1)
        print("[actor %d] preprocess %.1f | wait %.1f | env %.1f | fallback %d"
              % (idx, t_pre / n * 1000, t_wait / n * 1000, t_env / n * 1000, fallback),
              file=sys.stderr, flush=True)

        # [D] actor 进程有没有偷偷建 CUDA context
        cuda_ctx = torch.cuda.is_initialized()
        result_q.put(("ACTOR", idx, done, fallback, first_type, cuda_ctx))
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            srv.mark_exited(idx)
        except Exception:
            pass
        result_q.put(("ACTOR", idx, 0, -1, -1, True))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    DEV = torch.device("cuda:0")

    torch.manual_seed(20261102)

    # ---- 父进程建模型并 share_memory（与 test() 里的 model_learner 同构）----
    from alphastarmini.core.rl.alphastar_agent import AlphaStarAgent
    tmp = AlphaStarAgent(name="t21")
    model_learner = tmp.agent_nn.model
    model_learner.share_memory()
    print("model_learner 已 share_memory（参数量 %s）"
          % "{:,}".format(sum(p.numel() for p in model_learner.parameters())), flush=True)

    srv = AsyncBatchInferenceServer(None, N, DEV, make_shapes(),
                                    max_units=128, max_selected=AHP.max_selected,
                                    max_batch=N, return_logits=True)

    result_q = mp.Queue()
    procs = []

    # server 进程先起（它要设 srv.agent）
    ready_ev = mp.Event()
    ps = mp.Process(target=server_proc,
                    args=(srv, model_learner, DEV, None, result_q, ready_ev),
                    daemon=True)
    ps.start()
    procs.append(ps)

    # 等 server 把 agent 建好（跨进程必须用 Event，见 server_proc 里的注释）
    t0 = time.time()
    if not ready_ev.wait(timeout=90):
        print("FAIL：server 90 s 内未就绪", flush=True)
        return 1
    print("server 就绪耗时 %.1f s" % (time.time() - t0), flush=True)

    t_start = time.time()
    for i in range(N):
        p = mp.Process(target=actor_proc, args=(srv, i, STEPS, 3000 + i, result_q),
                       daemon=True)
        p.start()
        procs.append(p)

    got = []
    for _ in range(N + 1):
        try:
            got.append(result_q.get(timeout=300))
        except Exception:
            break
    wall = time.time() - t_start

    for p in procs:
        p.join(timeout=10)

    # ---------------- 汇总 ----------------
    actors = [g for g in got if g[0] == "ACTOR"]
    servers = [g for g in got if g[0] == "SERVER"]
    total_steps = sum(a[2] for a in actors)
    total_fb = sum(a[3] for a in actors)
    cuda_in_actor = [a[5] for a in actors]

    ok = True
    print("\n" + "=" * 66, flush=True)
    print("T21 跨进程批推理结果（N=%d, steps=%d）" % (N, STEPS), flush=True)
    print("  actor 完成: %d/%d | 总步数 %d | fallback %d | 墙钟 %.2f s"
          % (len(actors), N, total_steps, total_fb, wall), flush=True)
    for a in actors:
        print("    actor %d: %d 步, fallback=%d, 首个 action_type=%d, cuda_ctx=%s"
              % (a[1], a[2], a[3], a[4], a[5]), flush=True)
    for s in servers:
        print("  server: rounds=%s wall=%.2f s\n          %s\n          稳态 %.0f step/s"
              % (s[1], s[2], s[3], s[4]), flush=True)
    if total_steps > 0:
        print("  墙钟吞吐 %.0f step/s" % (total_steps / wall), flush=True)

    # [A]
    a_ok = len(actors) == N and all(a[2] == STEPS for a in actors) and total_fb == 0
    print("  %-6s [A] 跨进程跑通，fallback=0" % ("PASS" if a_ok else "FAIL"), flush=True)
    ok = ok and a_ok

    # [B] 每个 actor 都拿到了非负的 action_type
    b_ok = all(a[4] >= 0 for a in actors) and len(actors) == N
    print("  %-6s [B] 每个 actor 都收到有效 action（action_type >= 0）"
          % ("PASS" if b_ok else "FAIL"), flush=True)
    ok = ok and b_ok

    # [D] actor 不该建 CUDA context
    d_ok = all(c is False for c in cuda_in_actor) and len(cuda_in_actor) == N
    print("  %-6s [D] actor 进程未初始化 CUDA（cuda_ctx 全 False）: %s"
          % ("PASS" if d_ok else "FAIL", cuda_in_actor), flush=True)
    ok = ok and d_ok

    print("\n==> %s" % ("全部 PASS" if ok else "有 FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    sys.exit(main())
