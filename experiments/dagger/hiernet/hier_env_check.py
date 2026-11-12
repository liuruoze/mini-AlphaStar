"""P0 环境自检：TF / pysc2 1.2 / run_configs / 配置常量。"""
import os
import inspect

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import tensorflow as tf  # noqa: E402

print("[chk] tf", tf.__version__)

from pysc2.env.environment import TimeStep  # noqa: E402
print("[chk] TimeStep fields:", TimeStep._fields)

from pysc2 import run_configs  # noqa: E402
rc = run_configs.get()
print("[chk] runconfig:", type(rc).__name__)
print("[chk] start sig:", inspect.signature(rc.start))

import lib.config as C  # noqa: E402
print("[chk] FPS=%s difficulty=%s load=%s" % (C._FPS, C.difficulty, C._LOAD_MODEL_PATH))
print("[chk] sizes: high=%d tech_in=%d pop_in=%d | ctrl_out=%d base_out=%d tech_out=%d pop_out=%d battle_out=%d"
      % (C._SIZE_HIGH_NET_INPUT, C._SIZE_TECH_NET_INPUT, C._SIZE_POP_NET_INPUT,
         C._SIZE_CONTROLLER_OUT, C._SIZE_BASE_NET_OUT, C._SIZE_TECH_NET_OUT,
         C._SIZE_POP_NET_OUT, C._SIZE_BATTLE_NET_OUT))

import multi_agent as MA  # noqa: E402
print("[chk] MultiAgent methods:", [m for m in dir(MA.MultiAgent) if not m.startswith("_")][:12])
