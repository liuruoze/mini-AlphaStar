from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

__author__ = "Ruo-Ze Liu"

debug = True

# Minimap index
_M_HEIGHT = features.MINIMAP_FEATURES.height_map.index
_M_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_M_CAMERA = features.MINIMAP_FEATURES.camera.index
_M_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_M_SELECTED = features.MINIMAP_FEATURES.selected.index
