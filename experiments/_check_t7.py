"""检查项 + T7 第一步：验证 RL 训练路径，并 profile forward 的 CPU 算子分布。

检查项（上一轮改动没覆盖的部分）：
  T4 的所有测试都是 is_rl_training=False，因此 action_type_head 里
  `if self.is_rl_training and ...` 那段（我改的向量化）从未被执行过。
  这里用 set_rl_training(True) 跑一次，确认：
    [1] 不抛 device mismatch
    [2] 输出 logits 形状正确、mask 位置被正确置 True（不全是 -1e9）

T7：profile forward 的 CPU 时间，定位 29~41 ms 花在哪些算子上。
"""
import os
import sys
import glob

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


def main():
    print("torch %s | CUDAGraph=%s | compile=%s"
          % (torch.__version__, hasattr(torch.cuda, "CUDAGraph"), hasattr(torch, "compile")), flush=True)

    dev = torch.device("cuda:0")
    agent = Agent()
    agent.to(dev)
    net = agent.model.eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    st = Feature.feature2state(feats[idx].float().to(dev))
    h = torch.zeros(1, N, 128, device=dev)
    c = torch.zeros(1, N, 128, device=dev)

    # ---------- 检查项：RL 训练路径 ----------
    print("\n--- 检查 is_rl_training=True 路径 ---", flush=True)
    for flag_val in [False, True]:
        agent.set_rl_training(flag_val)
        try:
            with torch.no_grad():
                al, acts, nh, sun, en = agent.batch_action_logits_by_state(
                    st, hidden_state=(h, c), obs_list=None)
            atl = al.action_type
            masked = int((atl <= -1e8).sum().item())
            total = atl.numel()
            print("  is_rl_training=%-5s -> OK | logits %s | 被 mask 的位置 %d/%d (%.1f%%)"
                  % (flag_val, tuple(atl.shape), masked, total, masked / total * 100), flush=True)
        except Exception as e:
            import traceback
            print("  is_rl_training=%-5s -> FAIL: %r" % (flag_val, e), flush=True)
            traceback.print_exc()
    agent.set_rl_training(False)

    # ---------- T7: profiler ----------
    print("\n--- forward CPU profile（N=8，取前 18 个算子）---", flush=True)
    for _ in range(3):
        with torch.no_grad():
            agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
    torch.cuda.synchronize()

    from torch.autograd.profiler import profile
    with torch.no_grad():
        with profile(use_cuda=False) as prof:
            for _ in range(5):
                agent.batch_action_logits_by_state(st, hidden_state=(h, c), obs_list=None)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=18), flush=True)

    # 统计 aten::to 的总次数与耗时
    tot_to = 0.0
    cnt_to = 0
    tot_all = 0.0
    for evt in prof.key_averages():
        if evt.key.startswith("aten::to"):
            tot_to += evt.self_cpu_time_total
            cnt_to += evt.count
        tot_all += evt.self_cpu_time_total
    print("\naten::to 家族: %d 次, 自身 CPU %.1f ms (5 次 forward), 占全部算子自身 CPU 的 %.0f%%"
          % (cnt_to, tot_to / 1000, tot_to / max(tot_all, 1e-9) * 100), flush=True)
    print("每次 forward 平均: aten::to %.1f ms, 全部算子 %.1f ms"
          % (tot_to / 1000 / 5, tot_all / 1000 / 5), flush=True)


if __name__ == "__main__":
    main()
