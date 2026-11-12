#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用规则专家 v3 直接实盘：Protoss vs 内置 AI（默认难度 7 = VeryHard），Simple64。

与评测管线（mini-AlphaStar-off）完全解耦：不加载任何 SL 权重，
每帧 `summarize -> decide -> RAW_FUNCTIONS`，用于两件事：
  1. 统计 Build_Pylon / Build_Gateway 的**成功/失败次数**（失败 = 动作发出后
     建筑数在 N 帧内没增加，即被 SC2 拒绝或位置不可建）；
  2. 迭代脚本直到能稳赢难度 7。

用法：
  python experiments/dagger/rules_v4/rule_bot_play.py --games 5 --difficulty very_hard
  python experiments/dagger/rules_v4/rule_bot_play.py --games 1 --difficulty very_hard --verbose

注意：
- `unit_tags` 参数在本项目管线里是「raw_units 的下标」而非真实 tag
  （pysc2 features.py:1663 `find_original_tag` 用 `_raw_tags[position]` 解析），
  所以这里传下标，且必须是当前帧 raw_units 的位置。
- 建造坐标用 feature layer 的 64 域（与 label 的 world 同域）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pysc2.env import sc2_env
from pysc2.lib import actions as A
from pysc2.lib import features as F

sys.path.insert(0, _HERE)
import _rule_expert_v3 as RE
import _rule_expert_v4 as V4
import hier_expert as HX

DIFF = {
    "very_easy": sc2_env.Difficulty.very_easy,
    "easy": sc2_env.Difficulty.easy,
    "medium": sc2_env.Difficulty.medium,
    "medium_hard": sc2_env.Difficulty.medium_hard,
    "hard": sc2_env.Difficulty.hard,
    "harder": sc2_env.Difficulty.harder,
    "very_hard": sc2_env.Difficulty.very_hard,
    "cheat_vision": sc2_env.Difficulty.cheat_vision,
    "cheat_money": sc2_env.Difficulty.cheat_money,
    "cheat_insane": sc2_env.Difficulty.cheat_insane,
}

WATCH = {RE.PYLON: "pylon", RE.GATEWAY: "gateway", RE.NEXUS: "nexus",
         RE.CYBER: "cyber", RE.ASSIMILATOR: "assimilator"}
WATCH_TIMEOUT = 90          # 动作发出后多少帧内没看到建筑就算失败


def _find_units(ru, unit_type=None, alliance=1, idle_only=False):
    out = []
    for i, u in enumerate(ru):
        if alliance is not None and int(u["alliance"]) != alliance:
            continue
        if unit_type is not None and int(u["unit_type"]) != unit_type:
            continue
        if idle_only and int(u["order_length"]) != 0:
            continue
        out.append(i)
    return out


def to_func_call(act, ru, max_selected=12, tick=0, busy_tags=None):
    """把 v3 的 action dict 转成 pysc2 RAW FunctionCall；返回 (fc, err)。"""
    name = act["name"]
    hint = act.get("unit_type_hint")

    # 一起进攻：所有兵一次性全部下发。12 是标签侧的设计（max_selected），
    # raw 侧 unit_tags 上限是 512；按 12 截断会让多出来的兵永远待命。
    _cap = 512 if act["kind"] == "attack" else max_selected
    if act["kind"] == "attack":
        idxs = _find_units(ru, alliance=1)
        idxs = [i for i in idxs if int(ru[i]["unit_type"]) in RE.ARMY]
        if len(idxs) > _cap:
            k = int(tick) % len(idxs)
            idxs = idxs[k:] + idxs[:k]
    elif act["kind"] in ("train", "research"):
        # train_probe 允许排队：Nexus 在造农民、或被 learner 占着队列时也要能下命令，
        # 否则农民打光后永远补不出来（medium 实测 fc_no_unit 256–280 次、used=0）。
        idxs = _find_units(ru, unit_type=hint, alliance=1,
                           idle_only=(name != "train_probe"))
    elif act["name"] == "gather_gas" and act.get("target_tag") is not None:
        gi = int(act["target_tag"])
        gx, gy = int(ru[gi]["x"]), int(ru[gi]["y"])
        idxs = _find_units(ru, unit_type=RE.PROBE, alliance=1)
        idxs.sort(key=lambda i: (int(ru[i]["x"]) - gx) ** 2 + (int(ru[i]["y"]) - gy) ** 2)
        idxs = idxs[:3]
    elif act["name"] == "build_assimilator" and act.get("place"):
        gx, gy = int(act["place"][0]), int(act["place"][1])
        idxs = _find_units(ru, unit_type=RE.PROBE, alliance=1)
        idxs.sort(key=lambda i: (int(ru[i]["x"]) - gx) ** 2 + (int(ru[i]["y"]) - gy) ** 2)
    else:
        # 建造/采集：不强制空闲（Probe 全在采矿时也要能建），但空闲的排前面
        idxs = _find_units(ru, unit_type=hint, alliance=1)
        idxs.sort(key=lambda i: int(ru[i]["order_length"]) != 0)

    if act["kind"] == "build" and busy_tags:
        idxs = [i for i in idxs if int(ru[i]["tag"]) not in busy_tags]
    if act["kind"] == "build" and act.get("place") and len(idxs) > 1:
        # 建造只派一个 Probe（离目标最近的），多选会被 SC2 拒
        bx, by = int(act["place"][0]), int(act["place"][1])
        idxs.sort(key=lambda i: (int(ru[i]["x"]) - bx) ** 2 + (int(ru[i]["y"]) - by) ** 2)
        idxs = idxs[:1]
    if not idxs:
        return None, "no_unit"
    idxs = idxs[:_cap]

    try:
        fn = getattr(A.RAW_FUNCTIONS, act.get("raw_name") or name)
    except Exception:
        return None, "no_function"

    try:
        if act.get("target_tag") is not None:
            return fn(0, list(idxs), int(act["target_tag"])), None
        if act["kind"] in ("build", "attack"):
            if not act.get("place"):
                return None, "no_place"
            x, y = int(act["place"][0]), int(act["place"][1])
            return fn(0, list(idxs), [x, y]), None
        return fn(0, list(idxs)), None
    except Exception as e:
        return None, "build_fc:%s" % str(e)[:60]


def count_mine(ru, t):
    return sum(1 for u in ru if int(u["alliance"]) == 1 and int(u["unit_type"]) == t)


def run_one(env, feat, verbose=False, max_steps=40000, build=None, style="tech",
            hier=None, hier_interval=8, hier_max_pylon=8, difficulty_num=7,
            expert="v3"):
    _RE = V4 if expert == "v4" else RE
    if expert == "v4":
        V4.reset()      # 清跨局状态（tick 每局从 1 重计）
    obs = env.reset()[0]
    # 把 SC2 官方的 placement_grid 交给 v4 做选址（88x96，1=可建，权威数据）
    try:
        import numpy as _np
        _gi = env.game_info
        if isinstance(_gi, (list, tuple)):
            _gi = _gi[0]
        _pg = _gi.start_raw.placement_grid
        _arr = _np.frombuffer(_pg.data, dtype=_np.uint8).reshape(_pg.size.y, _pg.size.x)
        V4.PLACEMENT = (_arr > 0).astype(_np.uint8)
        print("[placement] grid=%dx%d buildable=%d"
              % (_pg.size.x, _pg.size.y, int(V4.PLACEMENT.sum())), flush=True)
    except Exception as _e:
        print("[placement] unavailable:", _e, flush=True)
        V4.PLACEMENT = None
    stats = {"actions": 0, "no_unit": 0, "err": 0,
             "build_ok": {}, "build_fail": {}, "build_err": {}, "hier_macro": {}}
    pending = []          # [{"id", "t", "key", "n_before"}]
    _pid = [0]
    issued = []           # [(x, y, frame)] 最近发出的建造点
    cooldown = {}         # action_name -> frame，避免同类建造每帧重发
    gas_sent = 0
    building_tags = {}    # probe tag -> frame，刚派去施工的 Probe
    frame = 0
    # HierNet 每 8 帧决策一次；120 帧冷却会把绝大多数 macro 吞掉（实测 216 次）。
    _CD = int(os.environ.get("MAS_BUILD_COOLDOWN", "16"))
    last_obs = obs

    while True:
        if obs.last():
            break
        frame += 1
        ru = obs.observation["raw_units"]
        tobs = obs.observation   # sc2_env 返回的已经是 transform 后的 feature NamedDict
        summ = RE.summarize(tobs)
        # 把最近 120 帧发出过的建造点并进 occupied，避免重复选同一格
        summ["occupied"] = list(summ["occupied"]) + [
            (x, y) for (x, y, ft) in issued if frame - ft < 10]
        summ["gas_sent"] = gas_sent
        if hier is not None and frame % hier_interval == 0:
            try:
                _gl = int(np.asarray(tobs["game_loop"]).reshape(-1)[0])
            except Exception:
                _gl = frame * 8
            f41 = HX.build_features(summ, tobs, difficulty=difficulty_num,
                                    game_loop=_gl)
            d = hier.decide(f41)
            _act = HX.macro_to_action(d["macro"], summ, frame,
                                      clamp_pylon=hier_max_pylon)
            if _act is not None:
                ok, act, why = True, _act, "hier:" + d["macro"]
                stats["hier_macro"][d["macro"]] = \
                    stats["hier_macro"].get(d["macro"], 0) + 1
            else:
                ok, act, why = _RE.decide(summ, tick=frame,
                                          build=build or RE.DEFAULT_BUILD, style=style)
        else:
            ok, act, why = _RE.decide(summ, tick=frame,
                                      build=build or RE.DEFAULT_BUILD, style=style)

        # ---- 结算 pending：建筑数是否增加 ----
        for p in sorted(pending, key=lambda x: x["id"]):
            key = p["key"]
            t = {"pylon": RE.PYLON, "gateway": RE.GATEWAY, "nexus": RE.NEXUS,
                 "cyber": RE.CYBER, "assimilator": RE.ASSIMILATOR}[key]
            if count_mine(ru, t) > p["n_before"]:
                stats["build_ok"][key] = stats["build_ok"].get(key, 0) + 1
                pending.remove(p)
                break          # 一次只结算一个，避免同 n_before 的多个 pending 重复计数
            if frame - p["t"] > WATCH_TIMEOUT:
                stats["build_fail"][key] = stats["build_fail"].get(key, 0) + 1
                pending.remove(p)

        fc = None
        if ok and act:
            if act["kind"] == "build" and frame - cooldown.get(act["name"], -10 ** 9) < _CD:
                fc, err = None, "cooldown"
            else:
                busy_tags = {t for t, ft in building_tags.items() if frame - ft < 400}
                fc, err = to_func_call(act, ru, tick=frame, busy_tags=busy_tags)
                if fc is not None and act["kind"] == "build":
                    cooldown[act["name"]] = frame
                    try:
                        _tags = fc.arguments[1]      # FunctionCall(arguments=[queued, unit_tags, target])
                        if len(_tags) > 0:
                            # 这里存的是 raw_units 下标，但 busy_tags 过滤用的是 u["tag"]
                            # （pysc2 的 unit_tags 参数收下标、内部再 find_original_tag）。
                            # 早先直接存下标 ⇒ 两边永不相等 ⇒ busy_tags 恒失效，
                            # 同一个 Probe 被反复改派，前一条建造令永远建不成。
                            _i0 = int(_tags[0])
                            building_tags[int(ru[_i0]["tag"])] = frame
                    except Exception:
                        pass
                if fc is not None and act["name"] == "gather_gas":
                    gas_sent += 3
            stats["actions"] += 1
            if fc is None:
                if err == "no_unit":
                    stats["no_unit"] += 1
                else:
                    stats["err"] += 1
                if verbose:
                    print("[skip] %-22s %s" % (act["name"], err), flush=True)
            elif act["kind"] == "build":
                if verbose and act["name"] == "build_gateway":
                    try:
                        _tags = fc.arguments[1]
                        _i = int(_tags[0])
                        _px, _py = int(ru[_i]["x"]), int(ru[_i]["y"])
                        _tx, _ty = int(act["place"][0]), int(act["place"][1])
                        print("[gw] f=%d place=(%d,%d) probe#%d xy=(%d,%d) d=%.1f "
                              "busy=%d mins=%.0f"
                              % (frame, _tx, _ty, _i, _px, _py,
                                 ((_px - _tx) ** 2 + (_py - _ty) ** 2) ** 0.5,
                                 len(busy_tags), summ["minerals"]), flush=True)
                    except Exception as _e:
                        print("[gw] dbg-err %s" % _e, flush=True)
                key = WATCH.get(act.get("unit_type_hint"), None)
                key = {"build_pylon": "pylon", "build_gateway": "gateway",
                       "build_nexus": "nexus", "build_cyber": "cyber",
                       "build_assimilator": "assimilator"}.get(act["name"])
                if act.get("place"):
                    issued.append((int(act["place"][0]), int(act["place"][1]), frame))
                if key and any(p["key"] == key for p in pending):
                    pass          # 同类还有未结算的，先不重复下单（否则统计会重复计数）
                elif key:
                    t = {"pylon": RE.PYLON, "gateway": RE.GATEWAY, "nexus": RE.NEXUS,
                         "cyber": RE.CYBER, "assimilator": RE.ASSIMILATOR}[key]
                    _pid[0] += 1
                    pending.append({"id": _pid[0], "t": frame, "key": key,
                                    "n_before": count_mine(ru, t)})
                if verbose:
                    print("[build] %-22s %-10s at %s  why=%s"
                          % (act["name"], key, act.get("place"), why), flush=True)

        if verbose and frame % 500 == 0:
            print("[t=%5d] fu=%2d/%2d min=%4.0f gas=%3.0f probe=%2d pylon=%d gw=%d cyber=%d "
                  "army=%d(z%d s%d a%d t%d)  -> %s"
                  % (frame, summ["food_used"], summ["food_cap"], summ["minerals"],
                     summ["vespene"], summ["n_probe"], summ["n_pylon"],
                     summ["n_gateway"], summ["n_cyber"], summ["n_army"],
                     summ["n_zealot"], summ["n_stalker"],
                     summ.get("n_adept", 0), summ["n_sentry"],
                     (act["name"] if act else "no_op")), flush=True)

        if fc is None:
            fc = A.RAW_FUNCTIONS.no_op()
        try:
            obs = env.step([fc])[0]
            last_obs = obs
        except Exception as e:
            print("[step-err] %s" % str(e)[:80], flush=True)
            obs = env.step([A.RAW_FUNCTIONS.no_op()])[0]
        if frame >= max_steps:
            break

    # 收尾：把未结算的 pending 记为失败
    for p in pending:
        stats["build_fail"][p["key"]] = stats["build_fail"].get(p["key"], 0) + 1

    o = last_obs
    return {
        "outcome": int(o.reward),
        "steps": int(frame),
        "killed": float(o.observation["score_cumulative"][5]) + float(o.observation["score_cumulative"][6]),
        "collected": float(o.observation["score_cumulative"][7]),
        "used": float(o.observation["score_cumulative"][11]),
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--difficulty", default="very_hard")
    ap.add_argument("--map", default="Simple64")
    ap.add_argument("--build", default="")
    ap.add_argument("--style", default="tech")
    ap.add_argument("--step_mul", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--expert", default="v3", choices=["v3", "v4", "hier"])
    ap.add_argument("--hier-npz", default="")
    ap.add_argument("--hier-det-scope", default="exec")
    ap.add_argument("--hier-seed", type=int, default=3038)
    ap.add_argument("--sc2-seed", type=int, default=0,
                    help=">0 时传给 SC2Env(random_seed)，让每局出生点/AI 有变化")
    ap.add_argument("--hier-interval", type=int, default=8)
    ap.add_argument("--hier-max-pylon", type=int, default=8)
    ap.add_argument("--difficulty-num", type=int, default=7)
    ap.add_argument("--save-replay-dir", default="")
    a = ap.parse_args()

    hier = None
    if a.expert == "hier":
        from hier_policy import HierPolicy
        npz = a.hier_npz or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hier_lv10.npz")
        hier = HierPolicy(npz, det_scope=a.hier_det_scope, seed=a.hier_seed)
        print("[expert] HierNet npz=%s scope=%s interval=%d max_pylon=%d"
              % (npz, a.hier_det_scope, a.hier_interval, a.hier_max_pylon),
              flush=True)
    if a.save_replay_dir:
        os.makedirs(a.save_replay_dir, exist_ok=True)

    aif = sc2_env.AgentInterfaceFormat(
        feature_dimensions=F.Dimensions(screen=64, minimap=64),
        use_feature_units=True,
        use_raw_units=True,
        use_raw_actions=True,      # 必须开：transform_action 走 raw 分支才会用 _raw_tags 解析索引
        max_raw_actions=512,
        max_selected_units=512,    # 与项目 action spec 一致（hyper_parameters.py:318）
        raw_resolution=64,
    )
    results = []
    for g in range(a.games):
        t0 = time.time()
        with sc2_env.SC2Env(
            map_name=a.map,
            players=[sc2_env.Agent(sc2_env.Race.protoss),
                     sc2_env.Bot(sc2_env.Race.terran, DIFF[a.difficulty])],
            agent_interface_format=aif,
            step_mul=a.step_mul,
            game_steps_per_episode=a.max_steps,
            random_seed=(a.sc2_seed if a.sc2_seed > 0 else None),
            save_replay_episodes=(1 if a.save_replay_dir else 0),
            replay_dir=(a.save_replay_dir or None),
            visualize=False,
        ) as env:
            gi = env.game_info
            if isinstance(gi, (list, tuple)):
                gi = gi[0]
            feat = F.features_from_game_info(
                game_info=gi, agent_interface_format=aif)
            r = run_one(env, feat, verbose=a.verbose, max_steps=a.max_steps,
                        build=(a.build or None), style=a.style,
                        hier=hier, hier_interval=a.hier_interval,
                        hier_max_pylon=a.hier_max_pylon,
                        difficulty_num=a.difficulty_num, expert=a.expert)
        r["game"] = g + 1
        r["difficulty"] = a.difficulty
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        if a.expert == "hier":
            print("         hier_macro=%s" % (r["stats"].get("hier_macro", {}),), flush=True)
        print("[game %d] outcome=%d steps=%d collected=%.0f killed=%.0f used=%.0f  %.1fs"
              % (r["game"], r["outcome"], r["steps"], r["collected"],
                 r["killed"], r["used"], r["wall_s"]), flush=True)
        st = r["stats"]
        print("         build_ok=%s build_fail=%s no_unit=%d err=%d"
              % (st["build_ok"], st["build_fail"], st["no_unit"], st["err"]), flush=True)

    wins = sum(1 for r in results if r["outcome"] > 0)
    agg = {}
    for r in results:
        for k, v in r["stats"]["build_ok"].items():
            agg.setdefault(k, {"ok": 0, "fail": 0})["ok"] += v
        for k, v in r["stats"]["build_fail"].items():
            agg.setdefault(k, {"ok": 0, "fail": 0})["fail"] += v
    out = {"difficulty": a.difficulty, "map": a.map, "games": a.games,
           "wins": wins, "results": results, "build_summary": agg}
    print("\n=== 汇总 ===")
    print("胜负: %d/%d" % (wins, a.games))
    for k in ("pylon", "gateway", "nexus", "cyber", "assimilator"):
        if k in agg:
            o, f = agg[k]["ok"], agg[k]["fail"]
            print("  %-12s ok=%3d fail=%3d  失败率=%.1f%%" % (k, o, f, 100.0 * f / max(1, o + f)))
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("\nsaved:", a.out)


if __name__ == "__main__":
    # pysc2 用 absl flags（sc2_run_config/sc2_version 等），必须先解析一次，
    # 否则 run_configs.get() 会抛 UnparsedFlagAccessError。只传程序名，
    # 让我们自己的 argparse 参数不被 absl 吃掉。
    from absl import flags as _absl_flags
    _absl_flags.FLAGS(sys.argv[:1])
    main()
