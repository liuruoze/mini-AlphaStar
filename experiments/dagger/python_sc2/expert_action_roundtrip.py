"""Round-trip check: python-sc2 Action -> pysc2 FunctionCall -> SC2 Action proto.

If reverse_raw_action + transform_action reproduce the same ability/tags/target,
the translated FunctionCall is a legally executable pysc2 action, not just a
parseable label.
"""
import json
import os
import pickle

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib import features as F
from pysc2.env import sc2_env

P = "/home/liuruoze/bvb/expert_probe"


def load(name):
    with open(os.path.join(P, name), "rb") as f:
        return pickle.load(f)


GI = sc_pb.ResponseGameInfo()
GI.ParseFromString(load("response_game_info.pkl"))
AIF = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
    raw_resolution=64, use_feature_units=True, use_raw_units=True,
    use_unit_counts=True, use_raw_actions=True, max_selected_units=12,
    show_cloaked=True, show_burrowed_shadows=True, show_placeholders=True,
)
FEAT = F.Features(agent_interface_format=AIF, map_size=GI.start_raw.map_size,
                  map_name=GI.map_name or "Simple64")

CASES = ["A_supply_blocked", "B_supply_ok", "C_tech_army"]
BOTS = ["warpgate", "cannon", "voidray"]
OUT = []

for case in CASES:
    ro = sc_pb.ResponseObservation()
    ro.ParseFromString(load("matrix_obs_%s.pkl" % case))
    prev_obs = FEAT.transform_obs(ro)
    for bot in BOTS:
        raws = load("matrix_actions_%s_%s.pkl" % (bot, case))
        for i, b in enumerate(raws):
            src = sc_pb.Action()
            src.ParseFromString(b)
            u0 = src.action_raw.unit_command
            rec = {"bot": bot, "case": case, "idx": i,
                   "ability_id": u0.ability_id,
                   "unit_tags": list(u0.unit_tags)}
            try:
                fc = FEAT.reverse_raw_action(src, prev_obs)
                back = FEAT.transform_action(prev_obs, fc, skip_available=True)
                if hasattr(back, "actions"):
                    acts = list(back.actions)
                elif isinstance(back, (list, tuple)):
                    acts = list(back)
                else:
                    acts = [back]
                u1 = acts[0].action_raw.unit_command
                rec.update({
                    "function": fc.function.name,
                    "roundtrip_ok": True,
                    "ability_match": u1.ability_id == u0.ability_id,
                    "tags_match": list(u1.unit_tags) == list(u0.unit_tags),
                    "target_pos_match": (
                        u1.HasField("target_world_space_pos") ==
                        u0.HasField("target_world_space_pos")),
                })
            except Exception as exc:
                rec.update({"function": None, "roundtrip_ok": False,
                            "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
            OUT.append(rec)

ok = sum(1 for r in OUT if r.get("roundtrip_ok"))
ab = sum(1 for r in OUT if r.get("ability_match"))
tg = sum(1 for r in OUT if r.get("tags_match"))
tp = sum(1 for r in OUT if r.get("target_pos_match"))
print("total=%d roundtrip_ok=%d ability_match=%d tags_match=%d target_pos_match=%d"
      % (len(OUT), ok, ab, tg, tp), flush=True)
errs = [r for r in OUT if not r.get("roundtrip_ok")]
print("errors=%d sample=%s" % (len(errs), errs[:2]), flush=True)
with open(os.path.join(P, "action_roundtrip.json"), "w") as f:
    json.dump({"summary": {"total": len(OUT), "roundtrip_ok": ok,
                           "ability_match": ab, "tags_match": tg,
                           "target_pos_match": tp}, "rows": OUT}, f, indent=1)
