"""Quantify the loss caused by transform_replay_data.py taking only function_calls[0].

Puts all actions produced by one offline expert step into one ResponseObservation,
runs the project's getFuncCall (or an equivalent fallback) and reports how many
survive the main loop's [0] slice.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, "/home/liuruoze/mini-AlphaStar")

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
    use_feature_units=True, use_raw_units=True,
    use_unit_counts=True, use_raw_actions=True,
    max_selected_units=12,
)
FEAT = F.Features(agent_interface_format=AIF, map_size=GI.start_raw.map_size,
                  map_name=GI.map_name or "Simple64")

try:
    from alphastarmini.core.sl.transform_replay_data import getFuncCall
    SOURCE = "project getFuncCall"
except Exception as exc:
    SOURCE = "fallback (getFuncCall import failed: %s)" % str(exc)[:80]

    def getFuncCall(o, feat, obs, use_raw=True):
        out = []
        for action in o.actions:
            fc = feat.reverse_raw_action(action, obs)
            if fc.function.value in (0, 168, 1):
                continue
            out.append(fc)
        return out if out else None


CASES = [
    ("warpgate", "A_supply_blocked"),
    ("warpgate", "B_supply_ok"),
    ("warpgate", "C_tech_army"),
    ("voidray", "C_tech_army"),
]

RESULTS = []
for bot, case in CASES:
    ro = sc_pb.ResponseObservation()
    ro.ParseFromString(load("matrix_obs_%s.pkl" % case))
    raws = load("matrix_actions_%s_%s.pkl" % (bot, case))
    for b in raws:
        a = sc_pb.Action()
        a.ParseFromString(b)
        ro.actions.add().CopyFrom(a)
    obs = FEAT.transform_obs(ro)
    try:
        fcs = getFuncCall(ro, FEAT, obs)
        names = [fc.function.name for fc in (fcs or [])]
        err = None
    except Exception as exc:
        names = []
        err = "%s: %s" % (type(exc).__name__, str(exc)[:140])
    kept = names[:1]
    RESULTS.append({
        "bot": bot, "case": case,
        "actions_in_step": len(raws),
        "getFuncCall_returned": len(names),
        "all": names,
        "kept_by_main_loop": kept,
        "dropped": len(names) - len(kept),
        "error": err,
    })

print("SOURCE:", SOURCE, flush=True)
print(json.dumps(RESULTS, indent=2), flush=True)
with open(os.path.join(P, "multiaction_loss.json"), "w") as f:
    json.dump({"source": SOURCE, "results": RESULTS}, f, indent=2)
