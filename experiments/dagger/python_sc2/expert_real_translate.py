"""Translate actions produced from REAL observations into pysc2 FunctionCall."""
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

FRAMES = [0, 5, 10, 15, 20, 25, 29]
OUT = []
for it in FRAMES:
    ro = sc_pb.ResponseObservation()
    ro.ParseFromString(load("real_obs_%03d.pkl" % it))
    prev_obs = FEAT.transform_obs(ro)
    for variant, stem in [("plain", "real_actions_%03d.pkl" % it),
                          ("boost", "real_actions_boost_%03d.pkl" % it)]:
        raws = load(stem)
        items = []
        for b in raws:
            a = sc_pb.Action()
            a.ParseFromString(b)
            aid = a.action_raw.unit_command.ability_id
            try:
                fc = FEAT.reverse_raw_action(a, prev_obs)
                items.append({"ability_id": aid, "function": fc.function.name, "ok": True})
            except Exception as exc:
                items.append({"ability_id": aid, "ok": False,
                              "error": "%s: %s" % (type(exc).__name__, str(exc)[:110])})
        ok = sum(1 for x in items if x.get("ok"))
        OUT.append({"frame": it, "variant": variant,
                    "n_actions": len(raws), "ok": ok, "translated": items})
        print("[real %03d %s] %d actions, %d ok: %s"
              % (it, variant, len(raws), ok,
                 [x.get("function", x.get("error")) for x in items]), flush=True)

with open(os.path.join(P, "real_translate_result.json"), "w") as f:
    json.dump(OUT, f, indent=2)
