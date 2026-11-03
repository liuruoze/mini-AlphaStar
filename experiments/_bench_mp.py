"""端到端验证：M 个 actor 进程 + 共享内存 + 中心批量推理。

关键：fork 必须在 CUDA 初始化之前，否则子进程会踩父进程的 CUDA context 死锁。
"""
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.getcwd())
import torch


def worker(idx, slot, barrier, n_iter, start_ev):
    torch.set_num_threads(1)
    dst = slot[idx]
    src = dst.clone()
    start_ev.wait()           # 等主进程预热完再开始，保证 barrier 配对次数一致
    for _ in range(n_iter):
        dst.copy_(src)        # 模拟 env 产出新 obs，写进共享槽（零拷贝给推理端）
        barrier.wait()        # 我准备好了
        barrier.wait()        # 推理完成，放行


def main():
    D = 343892
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    N_ITER = 20

    import glob
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    base = torch.load(f0)[0][:1].float()      # 真实一帧，不能用 randn（会触发 embedding 越界）
    D = base.shape[1]
    slot = torch.zeros(M, D).share_memory_()
    slot[:] = base
    barrier = mp.Barrier(M + 1)

    start_ev = mp.Event()
    procs = [mp.Process(target=worker, args=(i, slot, barrier, N_ITER, start_ev)) for i in range(M)]
    for p in procs:
        p.start()

    # ---- fork 完成后才初始化 CUDA ----
    import param as P
    P.on_server = True
    from alphastarmini.core.arch.arch_model import ArchModel
    from alphastarmini.core.sl.feature import Feature

    device = torch.device("cuda:0")
    net = ArchModel().to(device).eval()

    # 预热（worker 还阻塞在 start_ev 上，不影响 barrier 计数）
    for _ in range(2):
        with torch.no_grad():
            net.forward(Feature.feature2state(slot.to(device)), batch_size=M, sequence_length=1)
    torch.cuda.synchronize()
    start_ev.set()

    t = time.time()
    for _ in range(N_ITER):
        barrier.wait()
        st = Feature.feature2state(slot.to(device))
        with torch.no_grad():
            net.forward(st, batch_size=M, sequence_length=1)
        barrier.wait()
    wall = time.time() - t

    for p in procs:
        p.join(timeout=10)

    total = M * N_ITER
    print("M=%-3d  %.2f ms/round  ->  总 %5.0f step/s  (单 actor %.1f step/s)"
          % (M, wall / N_ITER * 1000, total / wall, N_ITER / wall), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
