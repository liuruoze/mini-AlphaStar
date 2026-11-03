"""T3a 关键验证：这台机器上，推理到底该用 GPU 还是 CPU？

现有认知被 profile 推翻：
  forward 的 CPU 提交时间 35.4 ms (N=1) / 40.6 ms (N=8)，**GPU 净耗时 0.0 ms**。

也就是说 GPU 算完还在等 CPU 提交下一个算子。那么：
  如果 CPU 直接跑 forward 更快（省掉 H2D 与 CUDA 调度开销），
  整个 batch acting 的架构前提（必须上 GPU 批推理）就要重估。

本脚本对比三种路径的 forward 时间：
  [A] GPU + return_logits=True（生产路径）
  [B] GPU + return_logits=False（四份计划 bench 用的口径）
  [C] CPU + return_logits=True
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("DISPLAY", ":1024")

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


def timeit(agent, feat, n, device, return_logits, reps=5):
    st = Feature.feature2state(feat.repeat(n, 1).to(device))
    h = torch.zeros(1, n, 128, device=device)
    c = torch.zeros(1, n, 128, device=device)
    kw = dict(hidden_state=(h, c))
    if return_logits:
        kw["obs_list"] = None
    fn = (lambda: agent.batch_action_logits_by_state(st, **kw)) if return_logits \
        else (lambda: agent.model.forward(st, batch_size=n, sequence_length=1, hidden_state=(h, c)))
    with torch.no_grad():
        for _ in range(2):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / reps


def main():
    torch.set_num_threads(8)
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    feat = feats[:1].float()

    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")

    ag = Agent(); ag.to(gpu); ag.model.eval()
    ac = Agent(); ac.to(cpu); ac.model.eval()

    print("%-4s %-22s %-22s %-22s" % ("N", "GPU logits=True", "GPU logits=False", "CPU logits=True"))
    for n in [1, 8]:
        ta = timeit(ag, feat, n, gpu, True)
        tb = timeit(ag, feat, n, gpu, False)
        tc = timeit(ac, feat, n, cpu, True)
        print("%-4d %-22s %-22s %-22s" %
              (n, "%.1f ms" % (ta * 1000), "%.1f ms" % (tb * 1000), "%.1f ms" % (tc * 1000)), flush=True)

    print("\n每 step 边际（N=8 除以 8）:")
    print("  见上表；若 CPU 明显快于 GPU，则说明当前瓶颈是 CUDA 算子调度而非算力", flush=True)


if __name__ == "__main__":
    main()
