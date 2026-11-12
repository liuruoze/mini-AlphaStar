"""把 HierNet-SC2 lv10-0.94 的 5 个子网 policy 分支导出成 numpy 权重。

背景：v2 夜间计划里，专家必须以 **raw 动作** 落地（pysc2 3.0 的
`reverse_raw_action` 只认 action_raw，HierNet 的 *_screen 动作会变 no_op），
所以要在 th_1_7(py3.8/torch) 里做 HierNet 推理、在 venv_hier(py3.7/tf1.15)
之外跑。跨环境只能靠 .npz。

网络结构（algo/ppo.py + lib/layer.py，activation=relu，units=64）：
    h1 = relu(x @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)
    logits = h2 @ W4 + b4
    probs = softmax(logits)          # act_probs
输入维度实测：controller 20 / base 41 / tech 29 / pop 32 / battle 41
输出维度实测：2 / 2 / 4 / 3 / 2

用法（在 ~/HierNet-SC2-main 下，用 ~/venv_hier/bin/python 跑）：
    python hier_export_npz.py --out ~/mini-AlphaStar/experiments/hier_lv10.npz
门禁：100 条随机输入，numpy 前向 vs TF `get_action_prob` 的 max|Δ| < 1e-6，
      argmax 100% 一致。
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from absl import flags  # noqa: E402

FLAGS = flags.FLAGS


def _patch_replica_device_setter():
    import tensorflow as tf
    orig = tf.train.replica_device_setter

    def _patched(ps_device="/job:ps/task:0", worker_device="/job:worker/task:0",
                 cluster=None, **kwargs):
        if cluster is None:
            return "/cpu:0"
        return orig(ps_device=ps_device, worker_device=worker_device,
                    cluster=cluster, **kwargs)

    tf.train.replica_device_setter = _patched


def fw_numpy(x, p):
    h1 = np.maximum(x @ p["W1"] + p["b1"], 0.0)
    h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0.0)
    logits = h2 @ p["W4"] + p["b4"]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./model/lv10-0.94/")
    ap.add_argument("--out", default="./hier_lv10.npz")
    ap.add_argument("--n-check", type=int, default=100)
    a = ap.parse_args()

    FLAGS(["hier_export_npz.py"])

    import numpy as np
    globals()["np"] = np
    import tensorflow as tf
    _patch_replica_device_setter()

    import lib.config as C
    C._LOAD_MODEL_PATH = a.model_dir
    from new_network import HierNetwork

    sess = tf.Session()
    net = HierNetwork(sess=sess, summary_writer=None, rl_training=False,
                      cluster=None, index=0, device="/cpu:0")
    sess.run(tf.global_variables_initializer())
    for fn in ("restore_controller", "restore_base", "restore_tech",
               "restore_pop", "restore_battle"):
        getattr(net, fn)()
    print("[export] 5 sub-policies restored", flush=True)

    subs = {
        "controller": net.controller,
        "base": net.base_net,
        "tech": net.tech_net,
        "pop": net.pop_net,
        "battle": net.battle_net,
    }
    var_by_name = {v.name: v for v in tf.global_variables()}

    flat, meta = {}, {}
    for tag, pol in subs.items():
        # pol.obs.name = "<scope>/obs:0"，变量在 <scope>/policy_net/ 之下
        prefix = pol.obs.name.split("/obs:")[0]
        p = {}
        for layer, key in (("DenseLayer1", "1"), ("DenseLayer2", "2"),
                           ("DenseLayer4", "4")):
            for what, dst in (("weights", "W" + key), ("bias", "b" + key)):
                name = "%s/policy_net/%s/%s:0" % (prefix, layer, what)
                if name not in var_by_name:
                    raise KeyError("missing TF variable: %s" % name)
                p[dst] = sess.run(var_by_name[name])
        in_dim, out_dim = int(p["W1"].shape[0]), int(p["W4"].shape[1])
        meta[tag] = {"scope": prefix, "in_dim": in_dim, "out_dim": out_dim}

        # 与 TF 原前向逐条比对
        rng = np.random.RandomState(20261110)
        xs = rng.randn(a.n_check, in_dim).astype(np.float32) * 3.0
        tf_probs = pol.get_action_prob(xs)
        np_probs = fw_numpy(xs, p)
        max_abs = float(np.max(np.abs(tf_probs - np_probs)))
        agree = float(np.mean(np.argmax(tf_probs, 1) == np.argmax(np_probs, 1)))
        meta[tag]["max_abs_diff"] = max_abs
        meta[tag]["argmax_agree"] = agree
        print("[export] %-10s in=%2d out=%d max|Δ|=%.3e argmax=%.3f"
              % (tag, in_dim, out_dim, max_abs, agree), flush=True)
        if max_abs >= 1e-6 or agree < 1.0:
            raise AssertionError("numpy/TF mismatch on %s" % tag)

        for k, v in p.items():
            flat["%s/%s" % (tag, k)] = v

    np.savez_compressed(a.out, **flat)
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("[export] PASS -> %s (%d arrays)" % (a.out, len(flat)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
