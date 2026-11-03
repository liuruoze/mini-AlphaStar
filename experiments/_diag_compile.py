"""P0-2：定位 torch.compile 在 torch 2.1.1 上失败的根因。

上一轮（space-bunny）现象：compile 报错，但错误被 dynamo 的 fallback 提示截断，
只看到 "import torch._dynamo ... Set TORCH_LOGS=..."，真正根因没显示。
而单独 `import torch._dynamo` 是 OK 的。

本脚本分三级定位：
  [1] compile 最小模块（spatial_encoder，纯 conv，最可能成功）
  [2] compile action_type_head（有 multinomial 采样）
  [3] compile 整个 ArchModel
每级打印完整 traceback（最后 25 行）与 dynamo 的 explanation。
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
print("torch", torch.__version__, flush=True)

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


def show_err(tag, e):
    import traceback
    tb = traceback.format_exc().splitlines()
    print("--- %s 失败 ---" % tag, flush=True)
    for l in tb[-25:]:
        print("  " + l[:170], flush=True)


def main():
    net = ArchModel().to(DEV).eval()
    f0 = sorted(glob.glob("data/replay_data_v3/*.pt"))[0]
    feats = torch.load(f0)[0]
    N = 4
    idx = torch.linspace(0, feats.shape[0] - 1, N).long()
    st = Feature.feature2state(feats[idx].float().to(DEV))
    h = torch.zeros(1, N, 128, device=DEV)
    c = torch.zeros(1, N, 128, device=DEV)

    with torch.no_grad():
        ee, emb, enums, utypes = net.entity_encoder(st.entity_state, return_unit_types=True)

    # ---- [1] 最小模块 ----
    print("\n[1] compile spatial_encoder", flush=True)
    try:
        cm = torch.compile(net.spatial_encoder, mode="default")
        with torch.no_grad():
            o = cm(st.map_state, ee)
        print("    OK: map_skip %s" % (tuple(o[0].shape),), flush=True)
    except Exception as e:
        show_err("spatial_encoder", e)

    # ---- [2] 带采样的大头 ----
    print("\n[2] compile action_type_head", flush=True)
    lo = torch.randn(N, 128, device=DEV)
    avail = st.statistical_state[6]
    try:
        cm2 = torch.compile(net.action_type_head, mode="default")
        with torch.no_grad():
            o2 = cm2(lo, net.scalar_encoder(st.statistical_state)[1], avail, None)
        print("    OK: logits %s" % (tuple(o2[0].shape),), flush=True)
    except Exception as e:
        show_err("action_type_head", e)

    # ---- [3] 整个网络 ----
    print("\n[3] compile ArchModel", flush=True)
    try:
        cm3 = torch.compile(net, mode="default")
        with torch.no_grad():
            o3 = cm3.forward(st, batch_size=N, sequence_length=1,
                             hidden_state=(h, c), return_logits=True, obs_list=None)
        print("    OK: logits %s" % (tuple(o3[0].action_type.shape),), flush=True)
    except Exception as e:
        show_err("ArchModel", e)


if __name__ == "__main__":
    main()
