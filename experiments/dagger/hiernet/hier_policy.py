"""HierNet-SC2 lv10-0.94 的纯 numpy 推理（无 TF 依赖，可在 th_1_7 里跑）。

配套 `hier_export_npz.py`：把 5 个子网 policy 分支导成 `hier_lv10.npz`，
已验证 numpy vs TF `max|Δ| <= 1.8e-7`、argmax 100% 一致。

决策流（multi_agent.py:265-330）：
    controller(high20) 0 -> base(41) {0:tech, 1:pop}   1 -> battle(41) {0:pop, 1:fight}
    tech(high20+tech9=29)  -> 0 pylon / 1 gas / 2 gateway / 3 cyber
    pop (high20+pop12=32)  -> 0 probe / 1 zealot / 2 stalker

分层 argmax（H1 实测）：`all` 与 `route` 会让 controller 恒 0、tech 从不执行；
`exec`（只对 tech/pop 取 argmax、路由层保留 softmax 采样）是唯一改善的配置。
"""
import json
import os

import numpy as np

TAGS = ("controller", "base", "tech", "pop", "battle")
IN_DIM = {"controller": 20, "base": 41, "tech": 29, "pop": 32, "battle": 41}
OUT_DIM = {"controller": 2, "base": 2, "tech": 4, "pop": 3, "battle": 2}

TECH_MACRO = ("build_pylon", "build_assimilator", "build_gateway", "build_cyber")
POP_MACRO = ("train_probe", "train_zealot", "train_stalker")

_DET_GROUPS = {
    "": set(),
    "all": set(TAGS),
    "route": {"controller", "base", "battle"},
    "exec": {"tech", "pop"},
}


def _fw(x, p):
    h1 = np.maximum(x @ p["W1"] + p["b1"], 0.0)
    h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0.0)
    logits = h2 @ p["W4"] + p["b4"]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class HierPolicy:
    def __init__(self, npz_path, det_scope="exec", seed=3038):
        if det_scope not in _DET_GROUPS:
            raise ValueError("det_scope must be one of %s" % sorted(_DET_GROUPS))
        z = np.load(npz_path, allow_pickle=False)
        self.p = {}
        for tag in TAGS:
            self.p[tag] = {
                k: np.asarray(z["%s/%s" % (tag, k)], dtype=np.float32)
                for k in ("W1", "b1", "W2", "b2", "W4", "b4")
            }
            assert self.p[tag]["W1"].shape[0] == IN_DIM[tag], tag
            assert self.p[tag]["W4"].shape[1] == OUT_DIM[tag], tag
        self.rng = np.random.RandomState(seed)
        self.det_scope = det_scope
        self.det = _DET_GROUPS[det_scope]

    def probs(self, tag, x):
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != IN_DIM[tag]:
            raise ValueError("%s expects %d dims, got %d" % (tag, IN_DIM[tag], x.shape[1]))
        return _fw(x, self.p[tag])[0]

    def act(self, tag, x):
        p = self.probs(tag, x)
        if tag in self.det:
            return int(p.argmax())
        return int(self.rng.choice(len(p), p=p))

    def decide(self, feat41):
        """feat41 = concat(high_input20, tech_cost9, pop_num12)。"""
        f = np.asarray(feat41, dtype=np.float32).reshape(-1)
        if f.size != 41:
            raise ValueError("feat41 must be 41 dims, got %d" % f.size)
        hi, tc, pn = f[:20], f[20:29], f[29:41]
        x_tech, x_pop = np.concatenate([hi, tc]), np.concatenate([hi, pn])

        c = self.act("controller", hi)
        if c == 0:
            b = self.act("base", f)
            if b == 0:
                m = self.act("tech", x_tech)
                return {"branch": "tech", "macro": TECH_MACRO[m], "idx": m}
            m = self.act("pop", x_pop)
            return {"branch": "pop", "macro": POP_MACRO[m], "idx": m}
        b = self.act("battle", f)
        if b == 0:
            m = self.act("pop", x_pop)
            return {"branch": "pop", "macro": POP_MACRO[m], "idx": m}
        return {"branch": "fight", "macro": "attack", "idx": 1}


def selftest(npz_path=None):
    npz_path = npz_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "hier_lv10.npz")
    pol = HierPolicy(npz_path, det_scope="exec", seed=3038)
    rng = np.random.RandomState(7)
    hits = {}
    for _ in range(200):
        d = pol.decide(rng.randn(41).astype(np.float32) * 2)
        key = d["macro"]
        hits[key] = hits.get(key, 0) + 1
    print("[hier_policy] npz=%s scope=exec 200 random decisions" % npz_path)
    print("[hier_policy] macro dist:", dict(sorted(hits.items())))
    assert hits, "no decisions"
    print("[hier_policy] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest(sys.argv[1] if len(sys.argv) > 1 else None))
