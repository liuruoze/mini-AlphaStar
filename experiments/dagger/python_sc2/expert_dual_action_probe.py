# -*- coding: utf-8 -*-
"""P-DUAL: can one pysc2 agent send TWO FunctionCalls in the same step?

Case A : single Build_Pylon_pt           -> find a legal placement, baseline
Case B : single Train_Probe_quick        -> baseline
Case C : [build, train] in one step      -> both effective?
Case D : [train, build] in same step     -> order-independent?
Each case starts a fresh episode (very_easy terran, Simple64, step_mul=8).

Verdict: C and D both effective -> real concurrency (learner + expert in one step)
         only one of them     -> only one Action reaches the engine
         neither              -> must fall back to alternating steps
"""
import json
import os
import sys

from absl import flags as _absl_flags
try:
    _absl_flags.FLAGS(["dual_action_probe"] + sys.argv[1:])
except Exception:
    _absl_flags.FLAGS.mark_as_parsed()

from pysc2.env import sc2_env
from pysc2.lib import actions, features
from pysc2.lib.actions import FunctionCall
from pysc2.lib.actions import RAW_FUNCTIONS as R

OUT = os.path.expanduser("~/bvb/expert_probe/dual_action_probe.json")

AIF = features.AgentInterfaceFormat(
    feature_dimensions=features.Dimensions(screen=64, minimap=64),
    action_space=actions.ActionSpace.RAW,
    use_feature_units=True,
    use_raw_units=True,
    use_unit_counts=True,
    use_raw_actions=True,
)


def mk_env():
    return sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.protoss),
                 sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
        step_mul=8,
        game_steps_per_episode=3000,
        agent_interface_format=[AIF],
        visualize=False,
        replay_dir=None,
        ensure_available_actions=False,
    )


def snap(o, env, tag=""):
    eo = env._obs[0]          # ResponseObservation (private but stable)
    ob = eo.observation
    mine = [u for u in ob.raw_data.units if u.alliance == 1]
    pyl = [u for u in mine if u.unit_type == 60]
    nex = [u for u in mine if u.unit_type == 59]
    aids = []
    kinds = []
    for a in eo.actions:
        fields = [f.name for f, _ in a.ListFields()]
        kinds.append("+".join(fields) if fields else "EMPTY")
        if a.HasField("action_raw"):
            aids.append(int(a.action_raw.unit_command.ability_id))
    out_kinds = kinds
    orders = []
    for n in nex:
        orders.append([int(x.ability_id) for x in n.orders])
    return {
        "tag": tag,
        "gl": int(ob.game_loop),
        "food": "%d/%d" % (ob.player_common.food_used, ob.player_common.food_cap),
        "food_used": int(ob.player_common.food_used),
        "minerals": int(ob.player_common.minerals),
        "nexus_orders": orders,
        "n_pylon": len(pyl),
        "pylon_prog": [round(float(u.build_progress), 2) for u in pyl],
        "n_actions_recorded": len(eo.actions),
        "action_kinds": out_kinds,
        "recorded_aids": aids,
        "probes": [[int(u.tag), round(float(u.pos.x), 1), round(float(u.pos.y), 1)]
                   for u in mine if u.unit_type == 84],
        "n_errors": len(eo.action_errors),
        "errors": [[int(e.ability_id), int(e.result)] for e in eo.action_errors],
    }


def unit_index(ob, utype):
    for i, u in enumerate(ob.raw_data.units):
        if u.alliance == 1 and u.unit_type == utype:
            return i, u
    return None, None


def diag(env, fc, label):
    ob = env._obs[0].observation
    f = env._features[0]
    try:
        a = f.transform_action(ob, fc, skip_available=True)
        has = a.HasField("action_raw")
        cmd = a.action_raw.unit_command if has else None
        print("[diag] %-18s fields=%-22s aid=%-6s tags=%s" % (
            label, ",".join(x.name for x, _ in a.ListFields()) or "EMPTY",
            int(cmd.ability_id) if cmd else None,
            list(cmd.unit_tags) if cmd else None), flush=True)
        return bool(has and cmd and int(cmd.ability_id))
    except Exception as e:
        print("[diag] %-18s ERR %s: %s" % (label, type(e).__name__, str(e)[:110]), flush=True)
        return False


NOOP = None


def idle(env, n):
    """Step n times with no-op to let units walk/build."""
    global NOOP
    if NOOP is None:
        NOOP = FunctionCall(R.no_op.id, [])
    o = None
    for _ in range(n):
        o = env.step([[NOOP]])[0]
    return o


def clamp(v):
    return max(2, min(85, int(v)))


def case_a_placement():
    """Find a legal Pylon placement with single commands."""
    env = mk_env()
    o = env.reset()[0]
    ob = env._obs[0].observation
    idx_p, probe = unit_index(ob, 84)
    idx_n, nexus = unit_index(ob, 59)
    nx, ny = nexus.pos.x, nexus.pos.y
    # raw unit_command.target_world_space_pos is in WORLD coords -> no rescale
    scale = 1.0
    fx, fy = nx, ny
    log = [snap(o, env, "reset")]
    found = None
    for dx, dy in [(0, 6), (6, 0), (-6, 0), (0, -6), (6, 6), (-6, 6), (6, -6), (-6, -6), (0, 9)]:
        tx, ty = clamp(fx + dx), clamp(fy + dy)
        fc = FunctionCall(R.Build_Pylon_pt.id, [[0], [idx_p], [tx, ty]])
        ok_conv = diag(env, fc, "build@%d,%d" % (tx, ty))
        o = env.step([[fc]])[0]
        o = idle(env, 20)
        s = snap(o, env, "build@%d,%d+20" % (tx, ty))
        s["target"] = [tx, ty]
        log.append(s)
        if s["n_pylon"] > 0:
            found = [tx, ty]
            break
    env.close()
    return {"case": "A_single_build", "placement": found, "log": log,
            "nexus_pos": [nx, ny], "scale": scale}, found


def case_b_train():
    env = mk_env()
    o = env.reset()[0]
    ob = env._obs[0].observation
    idx_n, _ = unit_index(ob, 59)
    base = snap(o, env, "reset")
    fc = FunctionCall(R.Train_Probe_quick.id, [[0], [idx_n]])
    o = env.step([[fc]])[0]
    s = snap(o, env, "train")
    env.close()
    ok = s["food_used"] > base["food_used"] or any(s["nexus_orders"])
    return {"case": "B_single_train", "ok": ok, "log": [base, s]}


def case_dual(order):
    """Position-independent test: Move one probe far away + Train a probe.

    Move  -> that probe's pos must change
    Train -> food_used must +1
    If both change, one agent really can issue two unit_commands per step.
    """
    env = mk_env()
    o = env.reset()[0]
    ob = env._obs[0].observation
    idx_p, probe = unit_index(ob, 84)
    idx_n, nexus = unit_index(ob, 59)
    base = snap(o, env, "reset")
    tag_p = int(probe.tag)
    tx, ty = clamp(nexus.pos.x + 14), clamp(nexus.pos.y + 14)
    fc_move = FunctionCall(R.Move_Move_pt.id, [[0], [idx_p], [tx, ty]])
    fc_train = FunctionCall(R.Train_Probe_quick.id, [[0], [idx_n]])
    ok_m = diag(env, fc_move, "move@%d,%d" % (tx, ty))
    ok_t = diag(env, fc_train, "train")
    acts = [fc_move, fc_train] if order == "move_first" else [fc_train, fc_move]
    o = env.step([acts])[0]
    s_imm = snap(o, env, "dual_" + order + "_imm")
    o = idle(env, 20)
    s = snap(o, env, "dual_" + order + "_+20")

    def pos_of(snapd, tag):
        for t, x, y in snapd["probes"]:
            if t == tag:
                return (x, y)
        return None

    p0, p1 = pos_of(base, tag_p), pos_of(s, tag_p)
    moved = 0.0
    if p0 and p1:
        moved = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
    train_ok = any(s["nexus_orders"]) or s["food_used"] > base["food_used"]
    return {"case": "C_dual_" + order, "order": order, "target": [tx, ty],
            "diag_move_ok": ok_m, "diag_train_ok": ok_t,
            "probe_tag": tag_p, "probe_from": p0, "probe_to": p1,
            "probe_moved": round(moved, 2),
            "move_effective": bool(moved >= 2.0),
            "train_effective": bool(train_ok),
            "both": bool(moved >= 2.0 and train_ok),
            "log": [base, s_imm, s]}


def main():
    res = {}
    b = case_b_train()
    res["B"] = b
    print("[B] single train ok=%s food %s -> %s" % (b["ok"], b["log"][0]["food"], b["log"][1]["food"]), flush=True)

    for order in ("move_first", "train_first"):
        c = case_dual(order)
        res[order] = c
        for row in c["log"]:
            print("    %-22s gl=%-5d food=%-6s pylon=%d rec=%-2d kinds=%s aids=%s err=%s"
                  % (row["tag"], row["gl"], row["food"], row["n_pylon"],
                     row["n_actions_recorded"], row.get("action_kinds"),
                     row["recorded_aids"], row["errors"]), flush=True)
        print("[%s] diag(move=%s train=%s) probe %s->%s moved=%.1f | move=%s train=%s both=%s"
              % (order, c["diag_move_ok"], c["diag_train_ok"], c["probe_from"], c["probe_to"],
                 c["probe_moved"], c["move_effective"], c["train_effective"], c["both"]), flush=True)

    res["verdict_both_effective"] = bool(res["move_first"]["both"] and res["train_first"]["both"])
    res["verdict_one_effective"] = bool(res["move_first"]["both"] or res["train_first"]["both"])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("[done] -> %s" % OUT, flush=True)
    print("[VERDICT] both_effective=%s one_effective=%s"
          % (res["verdict_both_effective"], res["verdict_one_effective"]), flush=True)


if __name__ == "__main__":
    main()
