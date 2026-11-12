"""Capture REAL non-empty ResponseObservation frames while a normal bot plays.

Unlike the earlier (action-less) probe, this bot calls the real CannonRushBot
on_step so the game advances and observations contain actual units.
"""
import os
import pickle
import sys

sys.path.insert(0, "/home/liuruoze/sc2bots")
from cannon_rush import CannonRushBot  # noqa: E402

import sc2.client as _cl  # noqa: E402

OUT = "/home/liuruoze/bvb/expert_probe"
MAX_FRAMES = 30
_LAST = {"obs": None}
_orig = _cl.Client.observation


async def _patched(self):
    r = await _orig(self)
    _LAST["obs"] = r
    return r


_cl.Client.observation = _patched


class ProbeCannonBot(CannonRushBot):
    async def on_step(self, iteration):
        response = _LAST["obs"]          # sc_pb.Response
        inner = response.observation if response is not None else None  # ResponseObservation
        if iteration < MAX_FRAMES and inner is not None:
            with open(os.path.join(OUT, "real_obs_%03d.pkl" % iteration), "wb") as f:
                pickle.dump(inner.SerializeToString(), f, protocol=2)
            obs = inner.observation
            if iteration % 10 == 0:
                print("[real] it=%d gl=%s units=%s minerals=%s food=%s/%s"
                      % (iteration, obs.game_loop, len(obs.raw_data.units),
                         obs.player_common.minerals, obs.player_common.food_used,
                         obs.player_common.food_cap), flush=True)
        await super().on_step(iteration)
