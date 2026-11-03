"""T7：forward CPU 开销优化尝试。

profiler 结论（_check_t7.py）：
  cudnn_convolution  28.0 ms self CPU（180 次，175 us/次）  <- 最大头
  multinomial        15.0 ms（85 次，176 us/次）
  aten::to            0.4 ms（占 1%）  <- Hy4 说的"27ms 固定开销"被证伪

cudnn 175 us/次对小尺寸输入异常高，怀疑是算法选择问题。
本脚本对比四种配置的 forward 时间：
  [A] 默认
  [B] cudnn.benchmark = True          （让 cuDNN 自动选最快算法）
  [C] cudnn.benchmark + TF32 关闭     （本卡是 4090，默认可能开 TF32）
  [D] torch.set_num_threads 变化       （确认线程数影响）
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import torch

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from alphastarmini.core.arch.agent import Agent
from alphastarmini.core.sl.feature import Feature


def bench(agent, st, h, c, reps=8, warm=3):
    with torch.no_grad():
        for _ in range(warm):
            agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
        torch.cuda.synchronize()
        return (time.time() - t0) / reps


def main():
    dev = torch.device("cuda:0")
    agent = Agent()
    agent.to(dev)
    agent.model.eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    for N in [1, 8]:
        idx = torch.linspace(0, feats.shape[0] - 1, N).long()
        st = Feature.feature2state(feats[idx].float().to(dev))
        h = torch.zeros(1, N, 128, device=dev)
        c = torch.zeros(1, N, 128, device=dev)

        print("\n=== N=%d ===" % N, flush=True)

        torch.backends.cudnn.benchmark = False
        t_a = bench(agent, st, h, c)
        print("  [A] 默认                       %.1f ms" % (t_a * 1000), flush=True)

        torch.backends.cudnn.benchmark = True
        bench(agent, st, h, c, reps=2, warm=3)      # 让 cuDNN 完成算法搜索
        t_b = bench(agent, st, h, c)
        print("  [B] cudnn.benchmark=True       %.1f ms  (%+.0f%%)"
              % (t_b * 1000, (t_b / t_a - 1) * 100), flush=True)

        if hasattr(torch.backends.cuda, "matmul"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                bench(agent, st, h, c, reps=2)
                t_c = bench(agent, st, h, c)
                print("  [C] benchmark + 关 TF32       %.1f ms  (%+.0f%%)"
                      % (t_c * 1000, (t_c / t_a - 1) * 100), flush=True)
            except Exception as e:
                print("  [C] 跳过: %r" % (e,), flush=True)

    # 线程数影响（主进程内）
    print("\n=== 线程数影响（N=8）===", flush=True)
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    st = Feature.feature2state(feats[idx].float().to(dev))
    h = torch.zeros(1, N, 128, device=dev)
    c = torch.zeros(1, N, 128, device=dev)
    for nt in [1, 2, 4, 8]:
        torch.set_num_threads(nt)
        t = bench(agent, st, h, c, reps=5)
        print("  torch.set_num_threads(%d) -> %.1f ms" % (nt, t * 1000), flush=True)


if __name__ == "__main__":
    main()
