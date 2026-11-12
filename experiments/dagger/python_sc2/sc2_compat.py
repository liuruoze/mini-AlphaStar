"""SC2 3.16.1 (Base55958) 与 python-sc2 0.8.4 的兼容层。

python-sc2 0.8.4 是按 SC2 4.x 的协议写的，和 3.16.1 有三处硬差异：

  1. **地图路径**：3.16 的 `local_map` 只认相对路径，且相对的是 SC2 进程
     cwd，绝对路径直接 InvalidMapPath。改成用 `map_data` 传地图二进制。
  2. **ability id 过滤**：`GameData.__init__` 会用 `AbilityId` 枚举过滤，
     3.16 的编号不在 4.x 枚举里 -> 整张 ability 表被清空 ->
     `unit.creation_ability` 全是 None -> `can_afford(PROBE)` 直接 assert。
     改成不按枚举过滤，并同时用 `ability_id` / `remaps_to_ability_id` 建索引。
  3. **未知 id 不能构造枚举**：`AbilityId(...)` 会抛 ValueError。用 int 子类
     兜底，保留 `.value` 接口，数值比较不受影响。

导入即生效，不改 python-sc2 源码，也不改任何 bot 脚本。
"""
import os

os.environ.setdefault("SC2PATH", os.path.expanduser("~/StarCraftII"))

import sc2.controller as _ctrl  # noqa: E402
import sc2.game_data as _gd  # noqa: E402
from sc2.ids.ability_id import AbilityId as _AbilityId  # noqa: E402
from s2clientprotocol import sc2api_pb2 as _sc_pb  # noqa: E402


# ---- 1. 地图：走 map_data -------------------------------------------------
async def _create_game_patched(self, game_map, players, realtime):
    req = _sc_pb.RequestCreateGame(
        local_map=_sc_pb.LocalMap(
            map_path="Maps/Melee/%s.SC2Map" % game_map.name,
            map_data=game_map.data,
        ),
        realtime=realtime,
    )
    for player in players:
        p = req.player_setup.add()
        p.type = player.type.value
        if isinstance(player, _ctrl.Computer):
            p.race = player.race.value
            p.difficulty = player.difficulty.value
    return await self._execute(create_game=req)


_ctrl.Controller.create_game = _create_game_patched


# ---- 2/3. ability 表 ------------------------------------------------------
class _AbilityIdLike(int):
    """未知编号的兜底：保持 int 比较语义，同时提供 `.value`。"""

    @property
    def value(self):
        return int(self)


def _tolerant_ability_id(self):
    raw = int(getattr(self._proto, "remaps_to_ability_id", 0)
              or getattr(self._proto, "ability_id", 0) or 0)
    try:
        return _AbilityId(raw)
    except ValueError:
        return _AbilityIdLike(raw)


_gd.AbilityData.id = property(_tolerant_ability_id)
_gd.AbilityData.id_exists = staticmethod(lambda aid: isinstance(aid, int) and aid != 0)

_orig_gamedata_init = _gd.GameData.__init__


def _gamedata_init(self, data):
    self.abilities = {}
    for a in data.abilities:
        keys = {int(getattr(a, "ability_id", 0) or 0),
                int(getattr(a, "remaps_to_ability_id", 0) or 0)}
        keys.discard(0)
        if not keys:
            continue
        try:
            ad = _gd.AbilityData(self, a)
        except Exception:  # noqa: BLE001  编号异常的条目直接跳过
            continue
        for k in keys:
            self.abilities.setdefault(k, ad)

    self.units = {u.unit_id: _gd.UnitTypeData(self, u)
                  for u in data.units if getattr(u, "available", True)}
    self.upgrades = {u.upgrade_id: _gd.UpgradeData(self, u) for u in data.upgrades}


_gd.GameData.__init__ = _gamedata_init


# ---- 4. creation_ability / research_ability -------------------------------
# SC2 3.16 的 unit_type 不带 ability_id（4.0 之后才有），用 pysc2 动作表
# 反查出的 sc2_unit_ability.json 补上。
import json as _json  # noqa: E402

_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "sc2_unit_ability.json")
with open(_MAP_PATH, encoding="utf-8") as _fh:
    _MAP = _json.load(_fh)
_UNIT_ABILITY = {int(k): int(v) for k, v in _MAP.get("units", {}).items()}
_UPGRADE_ABILITY = {int(k): int(v) for k, v in _MAP.get("upgrades", {}).items()}


def _creation_ability(self):
    aid = int(getattr(self._proto, "ability_id", 0) or 0)
    if not aid:
        aid = _UNIT_ABILITY.get(int(self._proto.unit_id), 0)
    if not aid:
        return None
    return self._game_data.abilities.get(aid)


_gd.UnitTypeData.creation_ability = property(_creation_ability)


def _research_ability(self):
    aid = int(getattr(self._proto, "research_ability", 0) or 0)
    if not aid:
        aid = _UPGRADE_ABILITY.get(int(self._proto.upgrade_id), 0)
    if not aid:
        return None
    return self._game_data.abilities.get(aid)


_gd.UpgradeData.research_ability = property(_research_ability)


# ---- 5. is_mineral_field / is_vespene_geyser ------------------------------
# 3.16 的 unit_type 里 has_minerals / has_vespene 没有正确填充，
# python-sc2 的 `Unit.is_mineral_field` 恒为 False -> distribute_workers 崩。
_MINERALS = set(int(x) for x in _MAP.get("minerals", []))
_GEYSERS = set(int(x) for x in _MAP.get("geysers", []))

import sc2.unit as _unit  # noqa: E402

_unit.Unit.is_mineral_field = property(
    lambda self: int(self._proto.unit_type) in _MINERALS)
_unit.Unit.is_vespene_geyser = property(
    lambda self: int(self._proto.unit_type) in _GEYSERS)


# ---- 6. websocket 压缩 -----------------------------------------------------
# websockets>=12 默认协商 permessage-deflate，SC2 3.16 的解析器不支持，
# 会间歇性报 "unable to parse websocket frame" 并断开。关掉压缩、放大帧上限。
import websockets as _ws  # noqa: E402

_orig_ws_connect = _ws.connect


def _connect_no_deflate(*args, **kwargs):
    kwargs.setdefault("compression", None)
    kwargs.setdefault("max_size", None)
    return _orig_ws_connect(*args, **kwargs)


_ws.connect = _connect_no_deflate

# sc2.sc2process 里是 `import websockets` 后调用 websockets.connect，同上已被替换


# ---- 7. join 后延迟 --------------------------------------------------------
# SC2 3.16 在 ResponseJoinGame 之后立刻收到 RequestData 会报
# "unable to parse websocket frame"（读循环未就绪），加 0.5s 让服务端站稳。
import asyncio as _asyncio  # noqa: E402
import sc2.client as _client_mod  # noqa: E402

_orig_join_game = _client_mod.Client.join_game


async def _join_game_delayed(self, *args, **kwargs):
    result = await _orig_join_game(self, *args, **kwargs)
    await _asyncio.sleep(float(os.environ.get("MAS_JOIN_DELAY", "0.5")))
    return result


_client_mod.Client.join_game = _join_game_delayed


# ---- 8. 请求打点（MAS_TRACE_REQ=1 时打印每个请求类型，用于定位协议不兼容）--
import sc2.protocol as _proto_mod  # noqa: E402

_orig_execute = _proto_mod.Protocol._execute


async def _execute_traced(self, **kwargs):
    if os.environ.get("MAS_TRACE_REQ") == "1":
        print("[req] %s" % (sorted(kwargs),), flush=True)
    return await _orig_execute(self, **kwargs)


_proto_mod.Protocol._execute = _execute_traced
