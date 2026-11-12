"""跑 HierNet-SC2 的 lv10-0.94 权重（Simple64，SC2 3.16.1）。

约束（源码 + 作者确认）：
  * 必须用自带的 `lib/my_sc2_env.SC2Env`（TimeStep 有 raw_observation）；
  * `C._FPS = 2.8` 且 `step_mul = 8` ⇒ controller 8s / base,battle 1s；
  * `high_input[0] = C.difficulty` 是输入特征；出生点固定左上。

HY v2 新增（夜间计划 PLAN_NIGHT_HY_20261110.md）：
  * --deterministic：推理走 argmax。源码 algo/ppo.py:37 的 act() 默认
    stochastic=True，而 multi_agent.py 五个调用点全用默认 ⇒ 一直按 softmax 采样。
    注意 stochastic=False 分支只返回 2 值而调用点解包 3 值，故这里包装成 3 元组。
  * --max-pylon N：只钳执行层（tech_action==0 且已建 Pylon>=N 时改 no_op）。
    tech macro 索引：0=pylon 1=gas 2=gateway 3=cyber（multi_agent.py:205）。
  * --enemy：bot_race（默认 T；本夜不换，避免 alliance 混杂）。
  * --save-replay-dir：lib/my_sc2_env.py:329 save_replay()。
  * 每局记录机制指标（Pylon/Gateway/Zealot 峰值、气消耗代理、第一波时间），
    用于 H1 判据，不只报胜率。
"""
import argparse
import json
import os
import sys
import time
import types

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from absl import flags  # noqa: E402

FLAGS = flags.FLAGS

# high_input 里我们要记录的字段（索引与既有 --trace 打印一致）
_IDX = (1, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17)
_FIELDS = ("t", "min", "gas", "cap", "used", "army", "probe", "zeal",
           "stalk", "pylon", "gas_b", "gw", "cyber")


def _patch_replica_device_setter():
    """单机跑：cluster=None 时把 replica_device_setter 退化成 /cpu:0。"""
    import tensorflow as tf

    orig = tf.train.replica_device_setter

    def _patched(ps_device="/job:ps/task:0", worker_device="/job:worker/task:0",
                 cluster=None, **kwargs):
        if cluster is None:
            return "/cpu:0"
        return orig(ps_device=ps_device, worker_device=worker_device,
                    cluster=cluster, **kwargs)

    tf.train.replica_device_setter = _patched


def _make_deterministic_act():
    """返回可绑到 Policy_net 实例上的确定性 act（保持 3 元组返回值）。"""
    import numpy as np

    def _act_det(self, obs, stochastic=True, verbose=True):
        probs, det, v = self.sess.run(
            [self.act_probs, self.act_deterministic, self.v_preds],
            feed_dict={self.obs: obs.reshape([1, -1])})
        return (probs,
                int(np.asarray(det).reshape(-1)[0]),
                float(np.asarray(v).reshape(-1)[0]))

    return _act_det


def _summarize(rows):
    """从每帧 high_input 记录算机制指标。"""
    if not rows:
        return {"n_frames": 0}
    col = {k: [r[i] for r in rows] for i, k in enumerate(_FIELDS)}
    first = next((r[0] for r in rows if r[5] >= 5), None)   # army>=5 的第一波
    gas_peak = max(col["gas"])
    gas_final = col["gas"][-1]
    return {
        "n_frames": len(rows),
        "t_final": col["t"][-1],
        "pylon_peak": max(col["pylon"]),
        "gateway_peak": max(col["gw"]),
        "cyber_peak": max(col["cyber"]),
        "zealot_peak": max(col["zeal"]),
        "stalker_peak": max(col["stalk"]),
        "army_peak": max(col["army"]),
        "cap_peak": max(col["cap"]),
        "used_peak": max(col["used"]),
        "gas_peak": gas_peak,
        "gas_final": gas_final,
        "gas_spent_proxy": max(0.0, gas_peak - gas_final),
        "first_wave_t": first,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--difficulty", type=int, default=7, help="对手难度（env 的 difficulty）")
    ap.add_argument("--feat-difficulty", type=int, default=-1,
                    help="写进 high_input[0] 的难度特征；-1 表示与 --difficulty 相同")
    ap.add_argument("--max-steps", type=int, default=18000, help="game steps per episode")
    ap.add_argument("--step-mul", type=int, default=8)
    ap.add_argument("--fps", type=float, default=2.8,
                    help="C._FPS；与 step_mul 配对：8<->2.8, 1<->22.4")
    ap.add_argument("--out", default="")
    ap.add_argument("--model-dir", default="./model/lv10-0.94/")
    ap.add_argument("--trace", action="store_true", help="打印每次 macro 决策")
    ap.add_argument("--deterministic", action="store_true",
                    help="等价 --det-scope all：5 个子网全部走 argmax")
    ap.add_argument("--det-scope", default="",
                    choices=["", "all", "route", "exec"],
                    help="分层 argmax：all=全部 / route=controller,base,battle / "
                         "exec=tech,pop。空=全随机（源码默认）")
    ap.add_argument("--max-pylon", type=int, default=0,
                    help=">0 时把 Pylon macro 钳到已建数 < N（只改执行层）")
    ap.add_argument("--enemy", default="T", help="bot_race，默认 T（本夜不换）")
    ap.add_argument("--save-replay-dir", default="",
                    help="非空时每局保存 .SC2Replay")
    a = ap.parse_args()

    FLAGS(["hier_run.py"])          # absl flags（sc2_run_config 等）必须先解析

    import lib.config as C
    import lib.utils as _U
    C._FPS = a.fps
    feat_diff = a.difficulty if a.feat_difficulty < 0 else a.feat_difficulty
    C.difficulty = feat_diff
    C._LOAD_MODEL_PATH = a.model_dir

    import tensorflow as tf
    _patch_replica_device_setter()

    from lib import my_sc2_env as sc2_env
    from new_network import HierNetwork
    import multi_agent as MA

    det_scope = "all" if a.deterministic else a.det_scope
    print("[hier] tf=%s env_diff=%d feat_diff=%d fps=%s step_mul=%d load=%s "
          "det_scope=%s max_pylon=%d enemy=%s"
          % (tf.__version__, a.difficulty, C.difficulty, C._FPS, a.step_mul,
             C._LOAD_MODEL_PATH, det_scope or "none", a.max_pylon, a.enemy),
          flush=True)

    sess = tf.Session()
    net = HierNetwork(sess=sess, summary_writer=None, rl_training=False,
                      cluster=None, index=0, device="/cpu:0")
    sess.run(tf.global_variables_initializer())
    for fn in ("restore_controller", "restore_base", "restore_tech",
               "restore_pop", "restore_battle"):
        getattr(net, fn)()
    print("[hier] 5 sub-policies restored", flush=True)

    if det_scope:
        det = _make_deterministic_act()
        groups = {
            "all": (net.controller, net.base_net, net.tech_net,
                    net.pop_net, net.battle_net),
            # 路由层：controller / base / battle（决定进 tech 还是 pop）
            "route": (net.controller, net.base_net, net.battle_net),
            # 执行层：tech（造什么建筑）/ pop（出什么兵）
            "exec": (net.tech_net, net.pop_net),
        }
        chosen = groups[det_scope]
        for pol in chosen:
            pol.act = types.MethodType(det, pol)
        print("[hier] deterministic argmax ACTIVE scope=%s on %d sub-policies"
              % (det_scope, len(chosen)), flush=True)

    agent = MA.MultiAgent(index=0, net=net, restore_model=False, rl_training=False)

    # 每帧机制指标记录（与 --trace 合并成一层，避免重复包装）
    rec = {"rows": []}
    _orig_ctrl = agent.get_controller_input

    def _ctrl_rec():
        v = _orig_ctrl()
        try:
            rec["rows"].append(tuple(float(v[i]) for i in _IDX))
        except Exception:
            pass
        if a.trace:
            print("[trace] t=%5.0f min=%4.0f gas=%3.0f cap=%2.0f used=%2.0f army=%2.0f "
                  "probe=%2.0f zeal=%2.0f stalk=%2.0f pyl=%2.0f gas_b=%d gw=%2.0f cyber=%d"
                  % (v[1], v[2], v[3], v[6], v[7], v[10], v[11], v[12], v[13],
                     v[14], v[15], v[16], v[17]), flush=True)
            try:
                hi, tech, pop = _U.get_input(agent.obs)
                print("[trace]         supply_left=%2.0f inprog[pyl=%.0f gas=%.0f gw=%.0f cyber=%.0f] "
                      "pop[a=%.0f i=%.0f gas1=%d gas2=%d q_p=%.0f q_z=%.0f q_s=%.0f]"
                      % (tech[4], tech[5], tech[6], tech[7], tech[8],
                         pop[0], pop[1], pop[7], pop[8], pop[9], pop[10], pop[11]),
                      flush=True)
            except Exception:
                pass
        return v

    agent.get_controller_input = _ctrl_rec

    if a.trace:
        for tag, pol in (("controller", net.controller), ("base", net.base_net),
                         ("tech", net.tech_net), ("pop", net.pop_net),
                         ("battle", net.battle_net)):
            _orig_act = pol.act

            def _mk(orig, tag_):
                def _act(obs, stochastic=True, verbose=False):
                    out = orig(obs, stochastic=stochastic, verbose=verbose)
                    print("[trace] %-10s -> %s" % (tag_, out[1]), flush=True)
                    return out
                return _act

            pol.act = _mk(_orig_act, tag)

    if a.max_pylon > 0:
        _orig_tech = agent.tech_step

        def _tech_capped(act):
            if act == 0:                      # 0 = pylon（multi_agent.py:205）
                try:
                    v = agent.get_controller_input()
                    if len(v) > 14 and float(v[14]) >= a.max_pylon:
                        act = -1              # 走 else -> safe_action(NO_OP)
                except Exception:
                    pass
            return _orig_tech(act)

        agent.tech_step = _tech_capped
        print("[hier] max_pylon=%d ACTIVE (exec-layer clamp only)" % a.max_pylon,
              flush=True)

    if a.save_replay_dir:
        os.makedirs(a.save_replay_dir, exist_ok=True)

    results = []
    for g in range(1, a.games + 1):
        t0 = time.time()
        rec["rows"] = []
        env = sc2_env.SC2Env(
            map_name="Simple64", agent_race="P", bot_race=a.enemy,
            difficulty=str(a.difficulty), step_mul=a.step_mul, score_index=-1,
            game_steps_per_episode=a.max_steps,
            screen_size_px=(64, 64), minimap_size_px=(64, 64),
            visualize=False, game_version="3.16.1")
        raw_result = None
        obs_reward = None
        try:
            agent.reset()
            agent.set_env(env)
            agent.play()
            raw_result = getattr(agent, "result", None)
            reward = None
            if isinstance(raw_result, dict):
                reward = raw_result.get("reward")
            try:
                obs_reward = int(agent.obs.reward)
            except Exception:
                pass
            if reward is None:
                reward = obs_reward
            if a.save_replay_dir:
                try:
                    env.save_replay(a.save_replay_dir)
                except Exception as _e:
                    print("[game %d] save_replay failed: %r" % (g, _e), flush=True)
        except Exception as exc:  # noqa: BLE001
            print("[game %d] ERROR %r" % (g, exc), flush=True)
            reward = None
        finally:
            try:
                env.close()
            except Exception:
                pass
        dt = time.time() - t0
        metrics = _summarize(rec["rows"])
        print("[game %d] reward=%s obs_reward=%s wall=%.1fs | %s"
              % (g, reward, obs_reward, dt,
                 {k: metrics.get(k) for k in ("t_final", "pylon_peak",
                                              "gateway_peak", "zealot_peak",
                                              "army_peak", "gas_peak",
                                              "gas_spent_proxy", "first_wave_t")}),
              flush=True)
        results.append({
            "game": g, "reward": reward, "obs_reward": obs_reward,
            "raw_result": (raw_result if isinstance(raw_result, (dict, str, int, float))
                           else str(raw_result)),
            "wall_s": round(dt, 1), "metrics": metrics,
        })

    wins = sum(1 for r in results if r["reward"] == 1)
    print("[summary] %d/%d win" % (wins, a.games), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"difficulty": a.difficulty, "feat_difficulty": C.difficulty,
                       "games": a.games, "wins": wins,
                       "deterministic": a.deterministic,
                       "det_scope": det_scope,
                       "max_pylon": a.max_pylon, "enemy": a.enemy,
                       "step_mul": a.step_mul, "fps": a.fps,
                       "max_steps": a.max_steps,
                       "results": results}, fh, indent=2, default=str)
        print("[summary] saved", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
