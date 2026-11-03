"""Kimi 补充实测 [1]：多推理进程扩展性 —— 裁决 Hy4 的 4 卡外推。

Hy4 的核心论断：批推理有 27.26 ms 的**每进程串行**固定开销，
所以 P 个推理进程能把它并行掉，4 卡 ~1300 step/s（纯外推，未实测）。

本脚本固定**总 actor 数 = 48**，只改推理进程数 P：
  P=1 -> 每进程 48 actor（Hy4 已实测 796，作对照）
  P=2 -> 每进程 24 actor
  P=4 -> 每进程 12 actor
每个推理进程绑一张卡（cuda:0..P-1），组内用双缓冲流水。

若总吞吐随 P 近似线性增长 -> 固定开销确实可并行，Hy4 成立；
若增长乏力 -> 固定开销里有不可并行的部分（GPU 串行化/内存带宽），外推要下修。

用法: python _bench_scale.py <P> [total_actors] [cpu_ms]
"""
import os
import sys
import time
import glob
import multiprocessing as mp

sys.path.insert(0, os.getcwd())
import torch

N_ITER = 24


def actor_worker(idx, in_slot, b_ready, b_done, n_iter, start_ev, cpu_ms):
    torch.set_num_threads(1)
    src = in_slot[0][idx].clone()
    start_ev.wait()
    cur = 0
    for _ in range(n_iter):
        in_slot[cur][idx].copy_(src)
        b_ready.wait()
        if cpu_ms > 0:
            time.sleep(cpu_ms / 1000.0)
        b_done.wait()
        cur ^= 1


def run_group(g, M, n_iter, cpu_ms, res_q):
    try:
        f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
        base = torch.load(f0)[0][:1].float()
        D = base.shape[1]
        in_slot = torch.zeros(2, M, D).share_memory_()
        in_slot[:] = base
        b_ready = mp.Barrier(M + 1)
        b_done = mp.Barrier(M + 1)
        start_ev = mp.Event()

        procs = [mp.Process(target=actor_worker,
                            args=(i, in_slot, b_ready, b_done, n_iter, start_ev, cpu_ms))
                 for i in range(M)]
        for p in procs:
            p.start()

        # fork 之后才初始化 CUDA
        import param as P
        P.on_server = True
        from alphastarmini.core.arch.arch_model import ArchModel
        from alphastarmini.core.sl.feature import Feature

        device = torch.device("cuda:%d" % g)
        net = ArchModel().to(device).eval()
        for cur in range(2):
            with torch.no_grad():
                net.forward(Feature.feature2state(in_slot[cur].to(device)),
                            batch_size=M, sequence_length=1)
        torch.cuda.synchronize()
        start_ev.set()

        cur = 0
        t_gpu = 0.0
        t = time.time()
        for _ in range(n_iter):
            b_ready.wait()
            t0 = time.time()
            st = Feature.feature2state(in_slot[cur].to(device))
            with torch.no_grad():
                net.forward(st, batch_size=M, sequence_length=1)
            torch.cuda.synchronize()
            t_gpu += time.time() - t0
            b_done.wait()
            cur ^= 1
        wall = time.time() - t

        for p in procs:
            p.join(timeout=10)
        res_q.put((g, M, wall / n_iter * 1000, t_gpu / n_iter * 1000, M * n_iter / wall))
    except Exception as e:
        import traceback
        traceback.print_exc()
        res_q.put((g, M, -1.0, -1.0, 0.0))


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    cpu_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 27.0
    M = total // P

    res_q = mp.Queue()
    groups = [mp.Process(target=run_group, args=(g, M, N_ITER, cpu_ms, res_q)) for g in range(P)]
    t0 = time.time()
    for p in groups:
        p.start()

    got = [res_q.get(timeout=300) for _ in range(P)]
    for p in groups:
        p.join(timeout=20)

    tot = sum(r[4] for r in got)
    for g, m, ms, gpu, thr in sorted(got):
        print("  group cuda:%d  m=%-3d  %6.2f ms/round (GPU %6.2f)  -> %6.0f step/s"
              % (g, m, ms, gpu, thr), flush=True)
    print("P=%d  总 actor=%d  每进程 m=%d  |  **系统合计 %.0f step/s**  |  单组均值 %.0f  |  耗时 %.1fs"
          % (P, M * P, M, tot, tot / P, time.time() - t0), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
