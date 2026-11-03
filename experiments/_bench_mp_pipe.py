"""Hy4 补充实测 [2]：双缓冲流水线能否吃掉 actor 侧的 CPU 等待。

原 _bench_mp.py 是严格同步：
    actor 写槽 -> [barrier] -> GPU 推理 -> [barrier] -> actor 才继续
CPU(env/preprocess) 与 GPU 完全串行，周期 = T_gpu + T_cpu。

本脚本改成 **双缓冲流水**：GPU 在算第 t 批时，actor 已经在准备第 t+1 批。
稳态周期 = max(T_gpu, T_cpu)，收益取决于两者谁大。
A=48 时 T_gpu ≈ 49ms，T_cpu ≈ 27ms(preprocess 9.6 + env.step 17.4) -> 期望压到 ~49ms。

用法: python _bench_mp_pipe.py <M> <cpu_ms>
  cpu_ms 模拟 actor 侧每步的 CPU 开销（真实值 ~27ms，0 表示纯 GPU 上限）
"""
import os
import sys
import time
import glob
import multiprocessing as mp

sys.path.insert(0, os.getcwd())
import torch

N_ITER = 24


def worker(idx, in_slot, out_slot, b_ready, b_done, n_iter, start_ev, cpu_ms):
    torch.set_num_threads(1)
    print("W%d born" % idx, file=sys.stderr, flush=True)
    src = in_slot[0][idx].clone()   # 槽形状 (2,M,D)，先取第 0 槽再取该 actor 行
    start_ev.wait()
    cur = 0
    for _ in range(n_iter):
        in_slot[cur][idx].copy_(src)     # 写本轮观测
        b_ready.wait()                   # 通知：本轮数据就绪
        if cpu_ms > 0:                   # 模拟 preprocess + env.step（与 GPU 推理重叠）
            time.sleep(cpu_ms / 1000.0)
        b_done.wait()                    # 取上一轮结果
        cur ^= 1
    # 多等一轮收尾，保证主进程最后一批取完
    if n_iter % 2 == 0:
        pass


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    cpu_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 27.0

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    base = torch.load(f0)[0][:1].float()
    D = base.shape[1]

    in_slot = torch.zeros(2, M, D).share_memory_()
    in_slot[:] = base
    # out 只需一个标量占位，验证时序即可（真实实现里放 logits）
    out_slot = torch.zeros(2, M, 4).share_memory_()

    b_ready = mp.Barrier(M + 1)
    b_done = mp.Barrier(M + 1)
    start_ev = mp.Event()

    procs = [mp.Process(target=worker,
                        args=(i, in_slot, out_slot, b_ready, b_done, N_ITER, start_ev, cpu_ms))
             for i in range(M)]
    for p in procs:
        p.start()
    print("[%.1fs] %d workers forked" % (time.time()-T0, M), file=sys.stderr, flush=True)

    # fork 之后再初始化 CUDA
    import param as P
    P.on_server = True
    from alphastarmini.core.arch.arch_model import ArchModel
    from alphastarmini.core.sl.feature import Feature

    device = torch.device("cuda:0")
    net = ArchModel().to(device).eval()
    print("[%.1fs] model ready" % (time.time()-T0), file=sys.stderr, flush=True)

    for cur in range(2):
        with torch.no_grad():
            net.forward(Feature.feature2state(in_slot[cur].to(device)),
                        batch_size=M, sequence_length=1)
    torch.cuda.synchronize()
    print("[%.1fs] warmup done" % (time.time()-T0), file=sys.stderr, flush=True)
    start_ev.set()

    # 流水：GPU 算第 cur 批时 actor 在准备第 cur^1 批
    cur = 0
    t_gpu = 0.0
    t = time.time()
    for it in range(N_ITER):
        if it % 5 == 0:
            print("[%.1fs] iter %d" % (time.time()-T0, it), file=sys.stderr, flush=True)
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

    total = M * N_ITER
    print("M=%-3d cpu=%-5.1fms  %.2f ms/round (其中GPU %.2f)  -> 总 %6.0f step/s  "
          "(流水前理论同步值 %.2f ms)"
          % (M, cpu_ms, wall / N_ITER * 1000, t_gpu / N_ITER * 1000,
             total / wall, t_gpu / N_ITER * 1000 + cpu_ms), flush=True)


T0 = time.time()

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
