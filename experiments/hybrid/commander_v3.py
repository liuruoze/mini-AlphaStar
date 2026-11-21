#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v3 指挥官：把"五条规则"逐条问一次，拼成命令。

与 policy.RuleCommander / commander_tree.TreeCommander 同接口：
    propose(state_or_obs) -> (MacroCommand | None, key, reason)

设计依据（2026-11-21 评测）：
  · 同一个 32B，一次要五位 → 0.237（坍缩成恒定向量）；拆成 5 次单问 → 0.925–0.963；
  · 局面用 JSON 渲染优于散文（0.963 vs 0.925）；
  · 批量一次算 5 条（0.55 s）比串行（1.26 s）快，但边界局面答案会变（0.912 vs 0.925），
    所以默认**串行**，`--batch` 可选。

一条 tick 只能发一条 MacroCommand，五条规则可能同时为真，因此按优先级取一条：
    补水晶 > 补兵营 > 补农民 > 量产 > 进攻
（前两条不满足会直接卡死后续；进攻放最后，等军队堆满再打。）

用法：
  # 离线自测：拿冻结题集的前 N 题，比对五位与标准答案
  python experiments/hybrid/commander_v3.py --selftest --n 20 \
      --base-url http://127.0.0.1:8769/v1

  # 实盘一局（注意 MAS_HYBRID=1 是总开关，漏了会静默退回纯 SL）
  DISPLAY=:1024 MAS_HYBRID=1 MAS_HYBRID_MODE=llm_v3 \
  LLM_BASE_URL=http://127.0.0.1:8769/v1 EVAL_MAX_LOOPS=21000 EVAL_OUT=/tmp/v3 \
      python run_rl_eval.py

坑（2026-11-21 实测）：
  · EVAL_MAX_LOOPS 的单位是**游戏步**（22.4 步 = 1 游戏秒），设 1500 只够 67 秒；
  · 盖楼类指令（build:Pylon/Gateway）在 arbiter.mask() 里是**常放行**，
    不等于强制执行——动作仍由 SL 策略选。arbiter.effective_kinds() 自己只把 train 算作"真正生效"。
    所以本指挥官目前**只能约束出兵**，补水晶/补兵营只是"允许"，不保证发生。
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from experiments.hybrid import llm_client
    from experiments.hybrid.commander import MacroCommand
    from experiments.hybrid.obs_to_text import NAMES as UNIT_NAMES
except Exception:
    sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "hybrid"))
    import llm_client
    from commander import MacroCommand
    from obs_to_text import NAMES as UNIT_NAMES

# 五条规则（与 eval_set_v3 / doctrine.py 同源）
RULES = [
    ("pylon", "补水晶：没有水晶，或者人口余量（上限−已用）≤ 9 且人口上限未满 200。"),
    ("probe", "补农民：农民 ≤ 14 且人口余量 ≥ 1。"),
    ("gateway", "补兵营：兵营 < 3 且至少有一个水晶（没有水晶就没有能量场，兵营放不下）。"),
    ("zealot", "量产狂热者：兵 < 24 且矿 ≥ 100 且农民 ≥ 10 且兵营 ≥ 1 且人口余量 ≥ 2。"),
    ("attack", "全面进攻：兵 ≥ 8。"),
]

SYS = ("你是星际争霸2神族的战术指挥官，本局打狂热者快攻。现在只判断一件事：\n\n%s\n\n"
       "结合下面的局面，回答“现在是否应该执行这件事”。只输出一个字：是 或 否，不要解释。")

# 名字 → 命令工厂。注意：**不能用列表 + zip 配 RULES**——两者顺序一旦不同，
# 就会出现"兵营的答案执行成补农民"（2026-11-21 实测踩过）。
CMD_FACTORY = {
    "pylon": lambda: MacroCommand("build", "Pylon", count=1, deadline_s=30, priority=4),
    "gateway": lambda: MacroCommand("build", "Gateway", count=1, deadline_s=45, priority=4),
    "probe": lambda: MacroCommand("train", "Probe", count=2, deadline_s=20, priority=3),
    "zealot": lambda: MacroCommand("train", "Zealot", count=4, deadline_s=30, priority=4),
    "attack": lambda: MacroCommand("attack", "enemy_main", count=1, deadline_s=30, priority=2),
}

# 执行优先级：教条顺序为 水晶→农民→兵营→进攻→量产，
# 2026-11-21 用户裁定把**进攻提到兵营之前**（否则兵营卡在 2 个时部队永远不推）。
# 结果：水晶 → 农民 → 进攻 → 兵营 → 量产。
PRIORITY = ("pylon", "probe", "attack", "gateway", "zealot")


def state_from_obs(obs):
    """pysc2 观测 → 规则需要的七个量（duck-typed，缺字段降级）。"""
    pl = getattr(obs, "player", None)
    st = {
        "t_s": float(getattr(obs, "game_loop", [0])[0]) / 22.4,
        "used": int(getattr(pl, "food_used", 0) or 0),
        "cap": int(getattr(pl, "food_cap", 0) or 0),
        "minerals": int(getattr(pl, "minerals", 0) or 0),
        "workers": 0, "army": 0, "gw": 0, "pylons": 0,
    }
    try:
        cnt = {}
        for u in obs.raw_units:
            if int(getattr(u, "alliance", 0)) == 1:      # 1 = self
                nm = UNIT_NAMES.get(int(u.unit_type), str(int(u.unit_type)))
                cnt[nm] = cnt.get(nm, 0) + 1
        st["workers"] = cnt.get("Probe", 0)
        st["gw"] = cnt.get("Gateway", 0)
        st["pylons"] = cnt.get("Pylon", 0)
        st["army"] = sum(v for k, v in cnt.items()
                         if k not in ("Probe", "Pylon", "Gateway", "Nexus", "Assimilator",
                                      "Forge", "CyberneticsCore", "TwilightCouncil",
                                      "RoboticsFacility", "Stargate", "TemplarArchives",
                                      "FleetBeacon"))
    except Exception:
        pass
    return st


def render_state(st):
    """JSON 渲染。**不带**"输出（5 位 0/1）"尾巴：实测带着它时模型会按整题回答，
    出现"42 个农民还答该补农民"这种错（离线 --clean 也比带尾巴高 2.5pp）。"""
    return ("局面数据（JSON）：\n" + json.dumps({
        "时间秒": int(st["t_s"]), "人口已用": st["used"], "人口上限": st["cap"],
        "人口余量": st["cap"] - st["used"], "矿": st["minerals"],
        "农民": st["workers"], "狂热者": st["army"],
        "兵营": st["gw"], "水晶": st["pylons"]}, ensure_ascii=False)
        + "\n\n输出（5 位 0/1）：")


def pick_command(bits):
    """五位（按 RULES 顺序）→ 命令 + 命中名字。抽出来是为了能单测，不用起 LLM。"""
    answer = {name: bool(on) for (name, _), on in zip(RULES, bits)}
    for name in PRIORITY:
        if answer.get(name):
            return CMD_FACTORY[name](), name
    return None, "none"


def selftest_mapping():
    """回归：2026-11-21 的"兵营答案执行成补农民"就是顺序错位。"""
    cases = [
        ([0, 0, 1, 0, 0], "gateway"),   # 只有兵营该造
        ([0, 1, 0, 0, 0], "probe"),     # 只有农民
        ([1, 0, 1, 1, 0], "pylon"),     # 全真时按优先级取水晶
        ([0, 0, 0, 1, 1], "attack"),    # 进攻优先于量产
        ([0, 1, 1, 0, 0], "probe"),     # 补农民优先于补兵营
        ([0, 0, 1, 0, 1], "attack"),    # 2026-11-21 裁定：进攻提到兵营之前
        ([0, 0, 0, 0, 0], "none"),
    ]
    bad = []
    for bits, want in cases:
        _, got = pick_command(bits)
        if got != want:
            bad.append((bits, want, got))
    if bad:
        raise AssertionError("位→命令映射自测失败：%s" % bad)
    return len(cases)


class V3Commander:
    """五条规则逐条问；一条 tick 只执行优先级最高的一条。"""

    def __init__(self, llm, use_batch=False, prompt_suffix=False, max_new_tokens=8):
        self.llm = llm
        self.use_batch = use_batch
        self.prompt_suffix = prompt_suffix
        self.max_new_tokens = max_new_tokens
        self.stats = {"calls": 0, "unparsed": 0, "decisions": 0}
        self.last_bits = ""
        self.last_cmd = None

    # ---- 单次决策 ----
    def _ask_all(self, st):
        text = render_state(st)
        if not self.prompt_suffix:
            text = text.split("\n\n输出")[0]
        msgs = [[{"role": "system", "content": SYS % rule},
                 {"role": "user", "content": text}] for _, rule in RULES]
        if self.use_batch:
            outs = self.llm.batch(msgs, max_new_tokens=self.max_new_tokens)
        else:
            outs = [self.llm.chat(m, max_new_tokens=self.max_new_tokens, temperature=0.0)
                    for m in msgs]
        vals = []
        for o in outs:
            v = llm_client.normalize_bool(o)
            if v is None:
                self.stats["unparsed"] += 1
            vals.append(bool(v))
        self.stats["calls"] += len(msgs)
        return vals

    def propose(self, state_or_obs):
        if isinstance(state_or_obs, dict) and "cap" in state_or_obs:
            st = state_or_obs
        else:
            st = state_from_obs(state_or_obs)

        if os.environ.get("MAS_V3_DRYRUN") == "1":
            # 诊断用：只读观测、不调 LLM、不发命令，用于隔离"读 obs"本身是否影响对局
            self.last_state = st
            return None, "00000", "dryrun"
        bits = self._ask_all(st)
        self.stats["decisions"] += 1
        self.last_bits = "".join("1" if b else "0" for b in bits)

        # 把看到的局面一并带出，便于核对"指挥官眼里的世界"是否与游戏一致
        self.last_state = st
        tag = "v3:%s|w%d a%d g%d py%d cap%d/%d min%d" % (
            "none", st["workers"], st["army"], st["gw"], st["pylons"],
            st["used"], st["cap"], st["minerals"])
        cmd, name = pick_command(bits)                              # 按名字对齐，不靠位置
        if cmd is not None:
            self.last_cmd = cmd
            return cmd, self.last_bits, tag.replace("none", name, 1)
        self.last_cmd = None
        return None, self.last_bits, tag
        self.last_cmd = None
        return None, self.last_bits, "v3:none"


def _selftest(args):
    d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "eval_set_v3.json"), encoding="utf-8"))
    items = d["items"][:args.n]
    llm = llm_client.build("openai", base_url=args.base_url, model_name=args.model)
    cmd = V3Commander(llm, use_batch=args.batch, prompt_suffix=not args.clean)

    hit = 0
    for it in items:
        st = {"t_s": it["state"]["t"], "used": it["state"]["used"], "cap": it["state"]["cap"],
              "minerals": it["state"]["minerals"], "workers": it["state"]["workers"],
              "army": it["state"]["army"], "gw": it["state"]["gw"],
              "pylons": it["state"]["pylons"]}
        prev = cmd.last_bits
        cmd.propose(st)
        ok = cmd.last_bits == it["label"]
        hit += ok
        print("%s 真值 %s 预测 %s %s｜命令 %s"
              % (it["id"], it["label"], cmd.last_bits, "OK" if ok else "XX",
                 ("%s %s×%d" % (cmd.last_cmd.kind, cmd.last_cmd.target, cmd.last_cmd.count))
                 if cmd.last_cmd else "-"))
    print("\n一致 %d/%d ｜ 调用 %d 次 ｜ 未解析 %d"
          % (hit, len(items), cmd.stats["calls"], cmd.stats["unparsed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="qwen32b")
    ap.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    ap.add_argument("--batch", action="store_true", help="5 条合并成一次请求")
    ap.add_argument("--clean", action="store_true", help="局面文本去掉'输出 5 位'后缀")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("目前只支持 --selftest")
    _selftest(a)


if __name__ == "__main__":
    main()
