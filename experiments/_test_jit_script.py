"""步骤 1 续：torch.jit.script 路线（trace 因自回归采样不可行，script 可以）。

关键数据修正：profiler 表里的 "CPU total" 是 **5 次 forward 的累计**，
所以每次 forward：cudnn_convolution 5.8 ms、conv1d 2.4 ms、conv2d 4.2 ms ——
**所有算子加起来才 ~15 ms，而 forward 总耗时 29.7 ms**。
剩下 ~15 ms 是 **Python 层开销**（模块调用、参数搬运、字典查找等），profiler 的 self CPU 抓不到。

→ 所以要提速就得消 Python 开销。trace 会把自回归采样固化成常量（不可行），
   但 **script 保留控制流**，采样被编译成 aten 算子，理论上可行。

本脚本对每个模块试 script，报告：可行？耗时变化？数值差异？
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


def timeit(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def try_script(name, mod, args, ref_fn):
    """尝试 script 一个模块，返回 (成功?, 耗时比, 最大数值偏差, 错误信息)"""
    try:
        with torch.no_grad():
            sm = torch.jit.script(mod)
            sm = torch.jit.freeze(sm)
    except Exception as e:
        msg = str(e).strip().splitlines()
        return False, None, None, (msg[0][:120] if msg else repr(e)[:120])
    try:
        with torch.no_grad():
            ref = ref_fn()
            got = sm(*args)
            ref_l = ref if isinstance(ref, (list, tuple)) else (ref,)
            got_l = got if isinstance(got, (list, tuple)) else (got,)
            worst = 0.0
            for r, g in zip(ref_l, got_l):
                if isinstance(r, torch.Tensor) and r.numel() and isinstance(g, torch.Tensor):
                    worst = max(worst, (r - g).abs().max().item())
        t_e = timeit(lambda: ref_fn())
        t_s = timeit(lambda: sm(*args))
        return True, (t_s / t_e if t_e else None), worst, ""
    except Exception as e:
        return False, None, None, "script 后运行失败: " + repr(e)[:120]


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

    with torch.no_grad():
        ee, emb, enums, utypes = net.entity_encoder(st.entity_state, return_unit_types=True)
        esc, sctx = net.scalar_encoder(st.statistical_state)
        mskip, esp = net.spatial_encoder(st.map_state, ee)

    print("=== 逐模块 script 可行性与收益（N=8）===", flush=True)
    rows = []

    ok, ratio, worst, err = try_script(
        "entity_encoder", net.entity_encoder, (st.entity_state,),
        lambda: net.entity_encoder(st.entity_state, return_unit_types=True))
    rows.append(("entity_encoder", ok, ratio, worst, err))

    ok, ratio, worst, err = try_script(
        "scalar_encoder", net.scalar_encoder, (st.statistical_state,),
        lambda: net.scalar_encoder(st.statistical_state))
    rows.append(("scalar_encoder", ok, ratio, worst, err))

    ok, ratio, worst, err = try_script(
        "spatial_encoder", net.spatial_encoder, (st.map_state, ee),
        lambda: net.spatial_encoder(st.map_state, ee))
    rows.append(("spatial_encoder", ok, ratio, worst, err))

    # action_type_head: forward(lstm_output, scalar_context, action_type_mask, action_type)
    lo = torch.randn(N, net.lstm_dim if hasattr(net, "lstm_dim") else 128, device=DEV)
    avail = st.statistical_state[6]
    ok, ratio, worst, err = try_script(
        "action_type_head", net.action_type_head, (lo, sctx, avail, None),
        lambda: net.action_type_head(lo, sctx, avail, None))
    rows.append(("action_type_head", ok, ratio, worst, err))

    for name, ok, ratio, worst, err in rows:
        if ok:
            print("  %-18s script OK  耗时 %s  数值 max|d|=%.3e"
                  % (name, ("%.0f%%" % (ratio * 100)) if ratio else "?", worst), flush=True)
        else:
            print("  %-18s script 失败: %s" % (name, err), flush=True)

    # ---------- 整体 forward ----------
    print("\n=== 整体 ArchModel.forward 的 script ===", flush=True)
    try:
        with torch.no_grad():
            sm = torch.jit.script(net)
        print("  script 成功", flush=True)
    except Exception as e:
        msg = str(e).strip().splitlines()
        print("  script 失败: %s" % (msg[0][:200] if msg else repr(e)[:200]), flush=True)
        # 打印更多上下文帮助定位
        import traceback
        tb = traceback.format_exc().splitlines()
        for l in tb[-12:]:
            print("    " + l[:160], flush=True)


if __name__ == "__main__":
    main()
