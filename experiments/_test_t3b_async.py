"""T3b 验证：动态批（异步调度）vs 同步批。

同步批（T2 实测）：N=4 15 step/s、N=8 19 step/s，actor 利用率仅 24~39%。
本测试用 AsyncBatchInferenceServer，actor 谁先到谁先进批，不再互相等。
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

from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes
from alphastarmini.lib.hyper_parameters import AlphaStar_Agent_Interface_Format_Params as AAIFP
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

NO_OP = A.FunctionCall(A.RAW_FUNCTIONS.no_op.id, [])

# 是否让 server 回传 behavior_logits（接 RL 训练时必须开）。
# 开启后每轮多拷 25.2 KB/actor 的 logits -> 用来测这个开销。
RETURN_LOGITS = os.environ.get("MAS_RETURN_LOGITS", "0") == "1"


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

        env = make_env(idx, seed)
        ts = env.reset()[0]
        obs = ts.observation
        action_spec = env.action_spec()
        if isinstance(action_spec, (list, tuple)) and not hasattr(action_spec, "functions"):
            action_spec = action_spec[0]

        done = 0
        fallback = 0
        held = []
        t_pre = t_wait = t_env = 0.0
        for t in range(steps):
            if t > 0 and ts.last():
                ts = env.reset()[0]
                obs = ts.observation

            _t = time.time()
            st = Agent.preprocess_state_all(obs, build_order=[], last_list=[0, 0, 0])
            ut = np.asarray(obs.raw_units)[:, FeatureUnit.unit_type]
            srv.write_state(idx, st, ut)
            t_pre += time.time() - _t

            _t = time.time()
            if RETURN_LOGITS:
                (a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n), bl = srv.submit_and_get(idx)
                # 模拟真实使用：轨迹里必须存下来（否则只测了传输不测持有）
                held.append(bl.action_type[0, :4].clone())
                if len(held) > 16:
                    del held[:8]
            else:
                a_type, a_delay, a_queue, a_units, a_tunit, a_tloc, n = srv.submit_and_wait(idx)
            t_wait += time.time() - _t

            action = ArgsAction(action_type=torch.tensor([a_type]), delay=torch.tensor([a_delay]),
                                queue=torch.tensor([a_queue]), units=a_units.reshape(1, -1),
                                target_unit=torch.tensor([a_tunit]),
                                target_location=a_tloc.reshape(1, -1))
            _t = time.time()
            try:
                fc = Agent.action_to_func_call(action, torch.tensor([n]), action_spec)
                ts = env.step([fc])[0]
            except Exception as e:
                fallback += 1
                if fallback <= 2:
                    import traceback
                    print("[actor %d] func_call 失败: %r" % (idx, e), file=sys.stderr, flush=True)
                    traceback.print_exc()
                ts = env.step([NO_OP])[0]
            t_env += time.time() - _t
            obs = ts.observation
            done += 1

        srv.mark_exited(idx)
        env.close()
        nn = max(done, 1)
        print("[actor %d] preprocess %.1f | wait(含排队+前向) %.1f | env+fc %.1f | fallback %d"
              % (idx, t_pre / nn * 1000, t_wait / nn * 1000, t_env / nn * 1000, fallback),
              file=sys.stderr, flush=True)
        result_q.put((idx, done, fallback, ""))
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            srv.mark_exited(idx)
        except Exception:
            pass
        result_q.put((idx, 0, -1, repr(e)))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    dev = torch.device("cuda:0")

    shapes = StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])

    import os as _os
    srv = AsyncBatchInferenceServer(None, N, dev, shapes,
                                    max_units=128, max_selected=AHP.max_selected,
                                    max_batch=N,
                                    poll_s=float(_os.environ.get("MAS_POLL_S", "0.0005")),
                                    return_logits=RETURN_LOGITS)
    import multiprocessing as mp
    result_q = mp.Queue()
    procs = srv.spawn_actors(actor_main, [(STEPS, 2000 + i, result_q) for i in range(N)])

    from alphastarmini.core.arch.agent import Agent
    agent = Agent(); agent.to(dev); agent.model.eval()
    srv.agent = agent
    print("CUDA agent ready, N=%d steps=%d return_logits=%s" % (N, STEPS, RETURN_LOGITS), flush=True)

    t0 = time.time()
    rounds = srv.run()
    wall = time.time() - t0

    got = [result_q.get(timeout=180) for _ in range(N)]
    total = sum(s for _, s, _, _ in got)
    fallbacks = sum(f for _, _, f, _ in got)
    print("\n==== T3b 动态批结果 ====", flush=True)
    print("N=%d 总步数=%d 墙钟 %.2f s -> **%.0f step/s**" % (N, total, wall, total / wall), flush=True)
    print(srv.async_summary(), flush=True)
    print("fallback(NO_OP) 次数 = %d %s" % (fallbacks, "(必须为 0)" if fallbacks else "OK"), flush=True)
    for idx, s, f, e in got:
        if e:
            print("  actor %d 失败: %s" % (idx, e), flush=True)
    srv.join(timeout=10)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    main()
