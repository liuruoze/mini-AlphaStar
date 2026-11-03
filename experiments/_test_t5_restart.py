"""T5 验收：actor 崩溃重启。

为什么需要
----------
第四轮列的盲区 4：actor 进程被 SC2 崩溃 / OOM / 外部 kill 干掉时，来不及 `mark_exited`，
`req[i]` 永久停在 SUBMITTED/IDLE。之前的 `idle_timeout` 只是**放弃**（整轮强制退出），
长跑训练里一个 actor 挂掉就损失吞吐、轨迹出现空洞 —— 必须能自愈。

本测试的真实性关键：**必须真的 kill -9 一个 actor 子进程**，
而不是模拟一个异常（异常会被 actor 自己的 try/except 吃掉，测不出监督逻辑）。

三个验收点
----------
[A] kill 一个 actor 后，supervisor 能检测到并重启它（restart_count 增加）
[B] 系统总步数继续增长（不因一个 actor 死亡而停摆）
[C] 重启后的 actor 的 hidden 被清零（否则会接下一条不相关轨迹的 LSTM 状态，静默错误）

为了控制变量，actor 用一个极轻量的假 actor（不做 SC2），只反复 submit 一个固定 state。
这样几十秒就能跑完，而监督逻辑与真实场景完全相同。
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import time
import numpy as np
import torch
import torch.multiprocessing as mp

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.rl.batch_server import AsyncBatchInferenceServer, StateShapes
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

DEV = torch.device("cuda:0")


def load_one_state():
    import glob
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    st = Feature.feature2state(feats[0:1].float())
    return st


def fake_actor(srv, idx, steps, die_after, result_q, startup_h, startup_cnt):
    """轻量 actor：反复 submit 同一 state，不做 env.step。

    die_after > 0 时在这么多步之后 os._exit(9) —— 模拟 SC2 崩溃（被 SIGKILL，
    来不及执行任何清理代码，包括 mark_exited）。
    """
    try:
        st = load_one_state()
        ut = np.zeros(8, dtype=np.int64)
        # 重启后的第一件事：读自己的 hidden 槽位。
        # server 的 _restart_dead 在 spawn 前会把 h[:, i, :] 与 c 清零，
        # 而此刻 server 还没 serve 过这个新 actor，所以这里读到的必须是 0。
        # 若不清零，新 actor 会继承上一条轨迹的 LSTM 状态 -> 与观测不匹配的静默错误。
        h_norm = float(srv.h[:, idx, :].abs().max())
        j = int(startup_cnt[idx])
        if j < 10:
            startup_h[idx * 10 + j] = h_norm
            startup_cnt[idx] = j + 1
        print("[actor %d] 启动第 %d 次，|h|max = %.6f" % (idx, j + 1, h_norm),
              file=sys.stderr, flush=True)
        n = 0
        for t in range(steps):
            if die_after and t >= die_after and idx == 1:
                # 真的硬退出：不走 except，不调 mark_exited
                os._exit(9)
            srv.write_state(idx, st, ut)
            srv.submit_and_wait(idx)
            n += 1
        srv.mark_exited(idx)
        result_q.put((idx, n, ""))
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            srv.mark_exited(idx)
        except Exception:
            pass
        result_q.put((idx, 0, repr(e)))


def main():
    torch.manual_seed(11)
    N = 6
    STEPS = 200

    shapes = StateShapes(
        entity_shape=(128, 1856), map_shape=(24, 64, 64),
        scalar_shapes=[(10,), (5,), (5,), (320,), (320,), (64,), (564,), (259,), (7,),
                       (259,), (13,), (320,), (20, 259), (128,), (564,), (2,)])

    agent = Agent()
    agent.to(DEV)
    agent.model.eval()

    srv = AsyncBatchInferenceServer(None, N, DEV, shapes,
                                    max_units=128, max_selected=AHP.max_selected,
                                    max_batch=N, return_logits=False)
    srv.agent = agent

    result_q = mp.Queue()
    startup_h = mp.Array('d', N * 10)
    startup_cnt = mp.Array('i', N)
    # 只让 actor 1 在第 5 步硬退出一次
    procs = srv.spawn_actors(
        fake_actor, [(STEPS, 5, result_q, startup_h, startup_cnt) for _ in range(N)])

    print("启动 %d 个 actor，其中 actor 1 会在第 5 步 os._exit(9)" % N, flush=True)

    # 关键：actor 1 重启后会拿到同样的 args（STEPS, 5），会再死一次，
    # 所以这里用 max_restarts 控制，验证「重启上限」这条路径也能走通。
    startup_norms = []
    t0 = time.time()
    rounds = srv.run(max_rounds=3000, idle_timeout=30.0, supervise=True, max_restarts=3)
    wall = time.time() - t0

    got = []
    while True:
        try:
            got.append(result_q.get(timeout=3))
        except Exception:
            break

    print("\n==== T5 结果 ====", flush=True)
    print("墙钟 %.2f s，rounds=%d" % (wall, rounds), flush=True)
    print(srv.async_summary(), flush=True)
    rc = srv.restart_count()
    print("actor 重启次数 = %d" % rc, flush=True)
    print("收到 %d/%d 个 actor 的完成结果" % (len(got), N), flush=True)

    ok = True
    # [A] 至少重启过一次
    if rc >= 1:
        print("[A] PASS：supervisor 检测到 actor 死亡并重启了 %d 次" % rc, flush=True)
    else:
        print("[A] FAIL：没有任何重启，监督逻辑没生效", flush=True)
        ok = False

    # [B] 总步数应该远超「一个 actor 死掉就停摆」的量
    steps = int(srv.stats[1])
    print("[B] 总步数 = %d（若停摆会接近 0）%s"
          % (steps, "PASS" if steps > 500 else "FAIL"), flush=True)
    if steps <= 500:
        ok = False

    # [C] 重启后 hidden 必须归零。证据：每个 actor 启动时读自己的 hidden 槽位并回传。
    rec = {}
    for i in range(N):
        vals = [float(startup_h[i * 10 + j]) for j in range(int(startup_cnt[i]))]
        rec[i] = vals
    print("[C] 各 actor 每次启动时的 |h|max：%s" % rec, flush=True)
    zero_all = all(v == 0.0 for vals in rec.values() for v in vals)
    if zero_all:
        print("[C] PASS：全部 %d 次启动（含重启）看到的 hidden 都是 0"
              % sum(len(v) for v in rec.values()), flush=True)
    else:
        print("[C] FAIL：有启动看到非零 hidden，新 actor 会继承旧轨迹的 LSTM 状态", flush=True)
        ok = False

    print("\n==> T5 崩溃重启：%s" % ("PASS" if ok else "FAIL"), flush=True)

    for p in srv._procs:
        if p is not None and p.is_alive():
            p.terminate()
    return 0 if ok else 1


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    sys.exit(main())
