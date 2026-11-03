"""在 th_2_1 (torch 2.1.1+cu118) 上试 torch.compile —— 验证「新 torch 路线」的真实收益。

背景：forward 29.7 ms = 算子 ~15 ms + Python ~15 ms。
消除 Python 开销的正解是 torch.compile（reduce-overhead 模式还会用 CUDA Graph）。
torch 2.1 已有 compile 与 cudagraphs，本轮正好在训练机上有一个可用的新 torch 环境
（th_2_1 修好后，wheel 是 manylinux_2_17，glibc 2.27 兼容）。

本脚本：
  [1] 验证 mini-AS 能否在 torch 2.1.1 + numpy 1.26 + PySC2 3.0 下 import 并 forward
  [2] 测 eager forward 的 CPU 提交时间与总时间
  [3] torch.compile(net.model, mode="reduce-overhead")，测收益与数值一致性
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
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| numpy", __import__("numpy").__version__, flush=True)

import param as P
P.on_server = True

from absl import flags
try:
    flags.FLAGS(["x"])
except Exception:
    flags.FLAGS.mark_as_parsed()

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl.feature import Feature

DEV = torch.device("cuda:0")


def timeit(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    net = ArchModel().to(DEV).eval()
    print("[1] mini-AS 在 torch 2.1.1 下 import + forward 成功", flush=True)

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    st = Feature.feature2state(feats[idx].float().to(DEV))
    h = torch.zeros(1, N, 128, device=DEV)
    c = torch.zeros(1, N, 128, device=DEV)

    def fwd():
        with torch.no_grad():
            return net.forward(st, batch_size=N, sequence_length=1,
                               hidden_state=(h, c), return_logits=True, obs_list=None)

    t_eager = timeit(fwd)
    print("[2] eager forward: %.2f ms" % (t_eager * 1000), flush=True)
    ref = fwd()[0].action_type.detach().clone()

    # ---- [3] torch.compile ----
    for mode in ["default", "reduce-overhead"]:
        print("\n[3] torch.compile(mode=%s) ..." % mode, flush=True)
        t0 = time.time()
        try:
            cm = torch.compile(net, mode=mode)
            with torch.no_grad():
                out = cm.forward(st, batch_size=N, sequence_length=1,
                                 hidden_state=(h, c), return_logits=True, obs_list=None)
            print("    编译耗时 %.1f s" % (time.time() - t0), flush=True)
            d = (out[0].action_type.detach() - ref).abs().max().item()
            t_c = timeit(lambda: cm.forward(st, batch_size=N, sequence_length=1,
                                            hidden_state=(h, c), return_logits=True, obs_list=None))
            print("    forward %.2f ms -> %.2f ms  (%+.0f%%) | 数值 max|d| = %.3e"
                  % (t_eager * 1000, t_c * 1000, (t_c / t_eager - 1) * 100, d), flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc().splitlines()
            print("    失败: %s" % (tb[-3] if len(tb) >= 3 else repr(e))[:220], flush=True)
            for l in tb[-8:-3]:
                print("      " + l[:150], flush=True)


if __name__ == "__main__":
    main()
