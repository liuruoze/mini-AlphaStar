"""P3 extension: translate every action from the 3x3 offline matrix."""
import json
import os
import pickle

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib import features as F
from pysc2.env import sc2_env

P = os.path.expanduser("~/bvb/expert_probe")


def load(name):
    with open(os.path.join(P, name), "rb") as f:
        return pickle.load(f)


GI = sc_pb.ResponseGameInfo()
GI.ParseFromString(load("response_game_info.pkl"))

AIF = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
    raw_resolution=64,
    use_feature_units=True,
    use_raw_units=True,
    use_unit_counts=True,
    use_raw_actions=True,
    max_selected_units=12,
    show_cloaked=True,
    show_burrowed_shadows=True,
    show_placeholders=True,
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
        stem = "%s_%s" % (bot, case)
        raws = load("matrix_actions_%s.pkl" % stem)
        items = []
        for b in raws:
            a = sc_pb.Action()
            a.ParseFromString(b)
            aid = a.action_raw.unit_command.ability_id
            try:
                fc = FEAT.reverse_raw_action(a, prev_obs)
                items.append({"ability_id": aid, "function": fc.function.name,
                              "function_id": int(getattr(fc.function, "value", -1)),
                              "ok": True})
            except Exception as exc:
                items.append({"ability_id": aid, "ok": False,
                              "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
        OUT.append({"bot": bot, "case": case, "n_actions": len(raws), "translated": items})

with open(os.path.join(P, "matrix_translate.json"), "w") as f:
    json.dump(OUT, f, indent=2)

for row in OUT:
    ok = sum(1 for x in row["translated"] if x.get("ok"))
    names = [x.get("function", x.get("error", "?"))[:28] for x in row["translated"]]
    print("[%s/%s] %d actions, %d translated: %s"
          % (row["bot"], row["case"], row["n_actions"], ok, names), flush=True)
