"""步骤 1：用 torch.jit.trace 降低 forward 的 CPU 调度开销。

背景（profiler，_check_t7.py）：
  forward 每次 29.5 ms，其中 cudnn_convolution 28 ms（180 次调用 / 5 次 forward = 36 次/forward）
  GPU 净耗时 0.0 ms —— CPU 全在等算子调度。
  torch 1.7 无 CUDAGraph、无 torch.compile，所以先试 torch.jit.trace。

本脚本：
  [1] 分模块计时：entity / scalar / spatial / core / heads 各占多少
  [2] trace 空间编码器，对比 eager vs traced 的耗时与数值
  [3] trace「encoders + core」整段（用 wrapper 复刻 arch_model.forward 的前半段），
      对比耗时与数值（数值必须与原 forward 一致，否则不采用）
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

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl.feature import Feature

DEV = torch.device("cuda:0")


def sync():
    torch.cuda.synchronize()


def timeit(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.time()
    for _ in range(reps):
        fn()
    sync()
    return (time.time() - t0) / reps


def main():
    torch.manual_seed(0)
    net = ArchModel().to(DEV).eval()

    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 8
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    st = Feature.feature2state(feats[idx].float().to(DEV))
    h = torch.zeros(1, N, 128, device=DEV)
    c = torch.zeros(1, N, 128, device=DEV)

    # ---------- [1] 分模块计时 ----------
    print("=== [1] 分模块耗时（N=8，eval 模式）===", flush=True)
    with torch.no_grad():
        ent = net.entity_encoder(st.entity_state, return_unit_types=True)
        ee, emb, enums, utypes = ent
        t_ent = timeit(lambda: net.entity_encoder(st.entity_state, return_unit_types=True))
        t_sca = timeit(lambda: net.scalar_encoder(st.statistical_state))
        t_spa = timeit(lambda: net.spatial_encoder(st.map_state, ee))
        # core 要三个 2D [batch_seq x embedding]，单独计时需要 flatten，这里跳过
        t_core = 0.0
        t_all = timeit(lambda: net.forward(st, batch_size=N, sequence_length=1,
                                           hidden_state=(h, c), return_logits=True, obs_list=None))
    print("  entity_encoder %6.2f ms" % (t_ent * 1000), flush=True)
    print("  scalar_encoder %6.2f ms" % (t_sca * 1000), flush=True)
    print("  spatial_encoder %6.2f ms" % (t_spa * 1000), flush=True)
    print("  core(LSTM)     %6.2f ms" % (t_core * 1000), flush=True)
    heads = t_all - t_ent - t_sca - t_spa - t_core
    print("  heads(其余)     %6.2f ms  [= 总 %.2f - 上面四项]" % (heads * 1000, t_all * 1000), flush=True)
    print("  总 forward     %6.2f ms" % (t_all * 1000), flush=True)

    # ---------- [2] trace spatial_encoder ----------
    print("\n=== [2] trace spatial_encoder ===", flush=True)
    with torch.no_grad():
        ref_spa, ref_emb = net.spatial_encoder(st.map_state, ee)
        try:
            with torch.no_grad():
                traced_spa = torch.jit.trace(net.spatial_encoder, (st.map_state, ee),
                                            check_trace=False)
                traced_spa = torch.jit.freeze(traced_spa)
            out_t, emb_t = traced_spa(st.map_state, ee)
            d = (out_t - ref_spa).abs().max().item()
            de = (emb_t - ref_emb).abs().max().item()
            print("  数值: map_skip max|d|=%.3e, embedded_spatial max|d|=%.3e" % (d, de), flush=True)
            t_eager = timeit(lambda: net.spatial_encoder(st.map_state, ee))
            t_traced = timeit(lambda: traced_spa(st.map_state, ee))
            print("  eager %.2f ms -> traced %.2f ms  (%+.0f%%)"
                  % (t_eager * 1000, t_traced * 1000, (t_traced / t_eager - 1) * 100), flush=True)
        except Exception as e:
            import traceback
            print("  trace 失败: %r" % (e,), flush=True)
            traceback.print_exc()

    # ---------- [3] trace encoders + core ----------
    print("\n=== [3] trace 「encoders + core」整段 ===", flush=True)

    def enc_part(state, hh, cc):
        entity_embeddings, embedded_entity, entity_nums, unit_types_one = \
            net.entity_encoder(state.entity_state, return_unit_types=True)
        map_skip, embedded_spatial = net.spatial_encoder(state.map_state, entity_embeddings)
        embedded_scalar, scalar_context = net.scalar_encoder(state.statistical_state)
        # core 期望三段都是 2D [batch_seq x embedding]，与 arch_model.forward 一致
        lstm_output, hidden_state = net.core(embedded_scalar,
                                             embedded_entity.reshape(embedded_entity.shape[0], -1),
                                             embedded_spatial.reshape(embedded_spatial.shape[0], -1),
                                             N, 1, (hh, cc))
        return (entity_embeddings, embedded_entity, entity_nums, unit_types_one,
                map_skip, embedded_spatial, embedded_scalar, scalar_context,
                lstm_output, hidden_state)

    with torch.no_grad():
        ref = enc_part(st, h, c)
        try:
            with torch.no_grad():
                traced = torch.jit.trace(enc_part, (st, h, c), check_trace=False)
                traced = torch.jit.freeze(traced)
            out = traced(st, h, c)
            worst = 0.0
            for r, o in zip(ref, out):
                if isinstance(r, torch.Tensor) and r.numel() > 0:
                    worst = max(worst, (r - o).abs().max().item())
            print("  数值: 所有张量输出 max|d| = %.3e" % worst, flush=True)
            t_eager = timeit(lambda: enc_part(st, h, c))
            t_traced = timeit(lambda: traced(st, h, c))
            print("  eager %.2f ms -> traced %.2f ms  (%+.0f%%)"
                  % (t_eager * 1000, t_traced * 1000, (t_traced / t_eager - 1) * 100), flush=True)
        except Exception as e:
            import traceback
            print("  trace 失败: %r" % (e,), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
