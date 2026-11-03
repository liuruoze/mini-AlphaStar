"""旁路基准：batch acting 到底可不可行（不改任何源码）。

回答四个问题：
  [1] 现状 bs=1 串行的吞吐
  [2] 拼批 + 一次 H2D 的吞吐（现实上限）
  [3] 纯 GPU 计算吞吐（理论上限，不含传输）
  [4] M 个 actor 进程 + 共享内存 + 中心批推理的端到端吞吐（IPC 是否成为瓶颈）
"""
import os
import sys
import time
import glob

sys.path.insert(0, os.getcwd())
import torch
import torch.multiprocessing as mp

import param as P
P.on_server = True

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl.feature import Feature
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

device = torch.device("cuda:0")
BARRIER = None


def _batch_forward(net, feats_cpu, B, reps=5, warmup=2):
    """把 B 份 feature 一次搬上卡 + 重组 + 前向，返回每批墙钟秒数。"""
    with torch.no_grad():
        for _ in range(warmup):
            st = Feature.feature2state(feats_cpu.to(device))
            net.forward(st, batch_size=B, sequence_length=1)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(reps):
            st = Feature.feature2state(feats_cpu.to(device))
            net.forward(st, batch_size=B, sequence_length=1)
        torch.cuda.synchronize()
        return (time.time() - t) / reps


def _pure_gpu_forward(net, state, B, reps=5):
    """state 已常驻 GPU，只测算力。"""
    with torch.no_grad():
        net.forward(state, batch_size=B, sequence_length=1)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(reps):
            net.forward(state, batch_size=B, sequence_length=1)
        torch.cuda.synchronize()
        return (time.time() - t) / reps


def _worker(idx, slot, n_iter, half):
    """模拟一个 actor：每轮把自己那份 obs 写进共享槽，等中心推理完。"""
    dst = slot[idx]
    src = slot[idx].clone()          # 本地快照，模拟「刚采到的新 obs」
    for _ in range(n_iter):
        dst.copy_(src)               # 1.4MB memcpy，走共享内存
        BARRIER.wait(timeout=180)    # 我准备好了
        BARRIER.wait(timeout=180)    # 推理完成，可以取 action


def main():
    net = ArchModel().to(device).eval()
    fs = sorted(glob.glob("data/replay_data_v3/*.pt"))
    print("dataset files: %d" % len(fs), flush=True)
    feats = torch.load(fs[0])[0].float()
    D = feats.shape[1]
    print("one file feats: %s   D=%d  (%.2f MB/frame float32)"
          % (tuple(feats.shape), D, D * 4 / 1024 / 1024), flush=True)
    print("max_entities=%d  embedding_size=%d  map_channels=%d  minimap_size=%d"
          % (AHP.max_entities, AHP.embedding_size, AHP.map_channels, AHP.minimap_size), flush=True)

    base = feats[:1]                 # (1, D) CPU，当作「一个 actor 的 obs」

    # ---------- [1] 现状 ----------
    st1 = Feature.feature2state(base.to(device))
    t1 = _pure_gpu_forward(net, st1, 1, reps=20)
    print("\n[1] bs=1 纯 GPU（现状 A3C）: %.2f ms/step -> %.1f step/s"
          % (t1 * 1000, 1 / t1), flush=True)

    # ---------- [2] 拼批 + H2D ----------
    print("\n[2] 拼批 + 一次 H2D（现实路径）", flush=True)
    for B in (8, 16, 32, 48, 64, 96):
        fb = base.repeat(B, 1)
        tb = _batch_forward(net, fb, B, reps=5)
        print("    bs=%-3d %.2f ms/batch -> %5.0f step/s" % (B, tb * 1000, B / tb), flush=True)

    # ---------- [3] 纯 GPU 上限 ----------
    print("\n[3] 纯 GPU 上限（state 常驻，只测算力）", flush=True)
    for B in (1, 8, 32, 64):
        st = Feature.feature2state(base.repeat(B, 1).to(device))
        tb = _pure_gpu_forward(net, st, B, reps=5)
        print("    bs=%-3d %.2f ms/batch -> %5.0f step/s" % (B, tb * 1000, B / tb), flush=True)

    # ---------- [4] 多进程 + 共享内存端到端 ----------
    global BARRIER
    print("\n[4] M 个 actor 进程 + 共享内存 + 中心批推理", flush=True)
    N_ITER = 40
    for M in (16, 48, 96):
        slot = torch.zeros(M, D).share_memory_()
        slot[:] = base
        BARRIER = mp.Barrier(M + 1)
        procs = [mp.Process(target=_worker, args=(i, slot, N_ITER, M)) for i in range(M)]
        for p in procs:
            p.start()
        t = time.time()
        for _ in range(N_ITER):
            BARRIER.wait()                                   # 等所有 actor 就绪
            st = Feature.feature2state(slot.to(device))       # 一次 H2D
            with torch.no_grad():
                net.forward(st, batch_size=M, sequence_length=1)
            BARRIER.wait()                                   # 放行
        wall = time.time() - t
        for p in procs:
            p.join()
        total = M * N_ITER
        print("    M=%-3d %.1f ms/round -> 总 %5.0f step/s (单 actor %.1f step/s)"
              % (M, wall / N_ITER * 1000, total / wall, N_ITER / wall), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
