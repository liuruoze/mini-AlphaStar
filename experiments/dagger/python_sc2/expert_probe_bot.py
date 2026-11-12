"""Read-only probe: dump ResponseData / ResponseGameInfo / ResponseObservation protos.

Loaded by experiments/dagger/python_sc2/sc2bot_run.py as a BotAI class. No game actions are issued;
the only purpose is to capture the protos needed to reconstruct a python-sc2
GameState offline (for the state->actions expert experiment).
"""
import os
import pickle

from sc2.bot_ai import BotAI

OUT = os.path.expanduser("~/bvb/expert_probe")
MAX_FRAMES = 25


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "wb") as f:
        pickle.dump(obj, f, protocol=2)


import sc2.game_data as _gd
import sc2.game_info as _gi
import sc2.client as _cl

_gd_init = _gd.GameData.__init__


def _gd_patched(self, data):
    _gd_init(self, data)
    try:
        self._raw_proto = data.SerializeToString()
    except Exception:
        self._raw_proto = None


_gd.GameData.__init__ = _gd_patched

_gi_init = _gi.GameInfo.__init__


def _gi_patched(self, proto):
    _gi_init(self, proto)
    try:
        self._raw_proto = proto.SerializeToString()
    except Exception:
        self._raw_proto = None


_gi.GameInfo.__init__ = _gi_patched

_LAST = {"obs": None}
_cl_obs = _cl.Client.observation


async def _obs_patched(self):
    r = await _cl_obs(self)
    _LAST["obs"] = r
    return r


_cl.Client.observation = _obs_patched


class ExpertProbeBot(BotAI):
    def on_start(self):
        os.makedirs(OUT, exist_ok=True)
        d = getattr(self._game_data, "_raw_proto", None)
        i = getattr(self._game_info, "_raw_proto", None)
        _save("response_data.pkl", d)
        _save("response_game_info.pkl", i)
        print("[probe] on_start data_bytes=%s info_bytes=%s"
              % (len(d) if d else None, len(i) if i else None), flush=True)

    async def on_step(self, iteration):
        response = _LAST["obs"]          # sc_pb.Response
        inner = response.observation if response is not None else None
        if iteration < MAX_FRAMES and inner is not None:
            _save("obs_%03d.pkl" % iteration, inner.SerializeToString())
            if iteration % 5 == 0:
                obs = inner.observation
                print("[probe] it=%d game_loop=%s units=%s"
                      % (iteration, obs.game_loop,
                         len(obs.raw_data.units)), flush=True)
        if iteration == MAX_FRAMES:
            print("[probe] captured %d frames" % MAX_FRAMES, flush=True)
