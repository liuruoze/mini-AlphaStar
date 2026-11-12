#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAgger 规则专家 v3 —— 完整 Protoss 打法（中级以上水平）

版本串：rule_v3_20261109
职责：给定当前帧 observation，输出「该不该干预 + 干预成什么动作 + 选中谁 + 造在哪」。
      只做决策，不构造六段标签（标签由 _dagger_build.py 用同类动作模板生成）。

Build order 来源（重要）
-----------------------
**不是自拟**：直接采用 SC2_PiGBot（Vers-AI，MIT）的 B2GM 系列（Bronze to GM 标准打法），
表见 `experiments/replay_gen/protoss_build_orders.json`，仅裁掉 CHRONO / WORKER_SCOUT（本项目无对应 head），
supply 与顺序未改。原项目：https://github.com/Vers-AI/SC2_PiGBot

三套可选 build：
- B2GM_PVP_2-Gate_Expand   单矿 2 兵营开局（最贴近本项目 Simple64 单矿场景，默认）
- B2GM_PVT_Standard_Build  标准宏观（Robo → Immortal → 3 兵营）
- B2GM_PVZ_Standard_Build  对虫标准

与 v1/v2 的边界
---------------
- v1 (`rule_v1_20261108`)：5 个动作、固定模板位置，实测退化成 Pylon 循环，
  1,486 次被拒、used=0（J120）。
- v2 (`rule_v2_20261109`)：只交付 `LocationPicker` 动态位置接口，decide 仍用 v1。
- **v3（本文件）**：build order 状态机（建筑/科技）+ 常驻规则（补农民/产兵/补人口）+ 扩张 + 进攻。

决策优先级（每帧一个动作）
------------------------
  P0 卡人口（food_used >= food_cap - 1 且矿够）→ Build_Pylon   ← 最高，否则一切停摆
  P1 build order 的下一个建筑/科技步骤（按 supply 门槛 + 资源 + 未重复）
  P2 Nexus 空闲且 Probe 未满 → Train_Probe（上限见 ConstantWorkerProductionTill）
  P3 Gateway 空闲 → Train_Stalker（有气）/ Train_Zealot（无气）
  P4 兜底补 Pylon（build order 之外的额外人口）
  P5 部队 >= 10 → Attack_pt

数据来源（全部为观测可读字段，不猜）
----------------------------------
- 矿：`score_cumulative[collected_minerals] - [spent_minerals]` (+50 初始)
- 气：`collected_vespene - spent_vespene`
- 人口：`score_by_category[food_used]`；上限 = 15*n_nexus + 8*n_pylon_done
- 空闲：`raw_units[i][order_length] == 0`
- 建成：`build_progress >= 1.0`
- 坐标：`raw_units[i][x], [y]`（64 域，与 label 的 world 同域）

动作 ID（pysc2 RAW_FUNCTIONS，与 action_type 一致）
--------------------------------------------------
  2 Attack_pt   13 Move_pt   34 Build_Nexus_pt   35 Build_Pylon_pt
  36 Build_Assimilator_unit  37 Build_Gateway_pt  47 Build_CyberneticsCore_pt
  48 Build_ShieldBattery_pt  49 Train_Zealot_quick  50 Train_Stalker_quick
  53 Train_Sentry_quick  64 Train_Probe_quick  82 Research_WarpGate_quick
"""
from __future__ import annotations

import json
import math
import os

try:
    import numpy as np
except Exception:
    np = None
VERSION = "rule_v3_20261109"

# 扩张默认关闭：Simple64 上二矿的选址/路径一直失败（实测 build_nexus 0/7），
# 每次失败白扔 400 矿 + 一个 Probe，直接拖死单矿经济。开的话设 MAS_ALLOW_EXPAND=1。
ALLOW_EXPAND = os.environ.get("MAS_ALLOW_EXPAND", "0") == "1"

# ---- 单位 type（pysc2 units.Protoss）----
NEXUS, PYLON, ASSIMILATOR, GATEWAY = 59, 60, 61, 62
CYBER, WARPGATE, SHIELDBATTERY = 72, 133, 1910
PROBE, ZEALOT, STALKER, SENTRY, ADEPT = 84, 73, 74, 77, 311
GEYSER = 342

BUILDINGS = {NEXUS, PYLON, ASSIMILATOR, GATEWAY, CYBER, WARPGATE, SHIELDBATTERY}
ARMY = {ZEALOT, STALKER, SENTRY, ADEPT}

# ---- 动作 ----
ACT = {
    "build_nexus": 34, "build_pylon": 35, "build_assimilator": 36,
    "build_gateway": 37, "build_cyber": 47, "build_battery": 48,
    "train_zealot": 49, "train_stalker": 50, "train_sentry": 53, "train_adept": 54,
    "train_probe": 64, "research_warpgate": 82, "research_charge": 83,
    "attack": 2, "move": 13, "harvest": 358, "gather_gas": 358,
}

# pysc2 RAW_FUNCTIONS 的函数名（实盘调用必须用这个名字，不是上面的 key）
ACT_RAW = {
    "build_nexus": "Build_Nexus_pt",
    "build_pylon": "Build_Pylon_pt",
    "build_assimilator": "Build_Assimilator_unit",
    "build_gateway": "Build_Gateway_pt",
    "build_cyber": "Build_CyberneticsCore_pt",
    "build_battery": "Build_ShieldBattery_pt",
    "train_zealot": "Train_Zealot_quick",
    "train_stalker": "Train_Stalker_quick",
    "train_sentry": "Train_Sentry_quick",
    "train_adept": "Train_Adept_quick",
    "train_probe": "Train_Probe_quick",
    "research_warpgate": "Research_WarpGate_quick",
    "research_charge": "Research_Charge_quick",
    "attack": "Attack_pt",
    "move": "Move_pt",
    "harvest": "Harvest_Gather_Probe_unit",
    "gather_gas": "Harvest_Gather_Probe_unit",
}

COST = {
    "build_pylon": (100, 0), "build_gateway": (150, 0), "build_cyber": (150, 0),
    "build_nexus": (400, 0), "build_assimilator": (75, 0), "build_battery": (100, 0),
    "train_probe": (50, 0), "train_zealot": (100, 0), "train_stalker": (125, 50),
    "train_sentry": (50, 100), "train_adept": (100, 25),
    "research_warpgate": (50, 50), "research_charge": (100, 100),
}

# ---- build order（取自 SC2_PiGBot config.yml，MIT）----
_BO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protoss_build_orders.json")


def load_builds(path=_BO_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d.get("builds", {}), d.get("action_map", {})
    except Exception as e:
        print("[rule_v3] build order 载入失败：%s" % e)
        return {}, {}


BUILDS, ACTION_MAP = load_builds()
DEFAULT_BUILD = "B2GM_PVP_2-Gate_Expand"


class LocationPicker:
    """建造点选择：**地形感知为主，几何环为兜底**。

    地形方案移植自 HierNet-SC2（用户项目，`lib/utils.py`）：
      `get_available_area` = (height_map == max) & (visibility_map == max)
      —— 用「最高地形层」判定主平台，绕开本项目 `buildable` 层全 0 的问题（J117）。
    与上游的差别（两处必要修改）：
      1. 上游读 `obs["screen"]`（镜头视野内）；我们读 `feature_minimap`（全图 64×64），
         否则镜头不在基地时看不到地形。
      2. 上游读 `obs["screen"][3]`（= SCREEN_FEATURES 的 `power`）；minimap 没有 power 字段，
         改为从 raw_units 的已建 Pylon 坐标自己算圆形覆盖（半径 6.5 格）。
    几何兜底：obs 里没有 feature_minimap 时退回原来的环带选择。
    """

    GRID = 64

    def __init__(self, r_min=2, r_max=12, exclude=2, pylon_radius=6.5):
        self.r_min, self.r_max = r_min, r_max
        self.exclude = exclude
        self.pylon_radius = pylon_radius

    # ---------- 地形感知 ----------
    def _terrain_mask(self, mm):
        """主平台 & 可见 & 无单位 的 64×64 布尔掩码（对齐 HierNet 的 get_available_area）。"""
        try:
            hm = np.asarray(mm["height_map"]).astype(np.int32)
            vm = np.asarray(mm["visibility_map"]).astype(np.int32)
            ut = np.asarray(mm["unit_type"]).astype(np.int32)
        except Exception:
            return None
        if hm.shape[0] != self.GRID:
            return None
        ok = (hm == int(hm.max())) & (vm == int(vm.max())) & (ut == 0)
        return ok

    def _pylon_cover(self, pylons):
        """已建 Pylon 的圆形能量覆盖（半径 6.5 格）。"""
        cover = np.zeros((self.GRID, self.GRID), dtype=bool)
        r = int(math.ceil(self.pylon_radius))
        rr = self.pylon_radius ** 2
        for (px, py) in pylons:
            for dy in range(-r, r + 1):
                y = py + dy
                if y < 0 or y >= self.GRID:
                    continue
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy > rr:
                        continue
                    x = px + dx
                    if 0 <= x < self.GRID:
                        cover[y, x] = True
        return cover

    def pick_terrain(self, mm, anchor, occupied, tick=0, pylons=None, need_power=False,
                     r_max=None, r_min=None, require_power=False):
        """地形感知选址；返回 (x, y) 或 None。"""
        if mm is None:
            return None
        ok = self._terrain_mask(mm)
        if ok is None or not ok.any():
            return None
        # 排除已占实体（方形膨胀 exclude 格）
        occ = np.zeros((self.GRID, self.GRID), dtype=bool)
        for (x, y) in occupied:
            x0, x1 = max(0, x - self.exclude), min(self.GRID, x + self.exclude + 1)
            y0, y1 = max(0, y - self.exclude), min(self.GRID, y + self.exclude + 1)
            occ[y0:y1, x0:x1] = True
        cand = ok & ~occ
        # 距离锚点 r_min..r_max（切比雪夫），修复「40% 离基地 >10 格」（J121）
        yy, xx = np.mgrid[0:self.GRID, 0:self.GRID]
        d = np.maximum(np.abs(xx - int(anchor[0])), np.abs(yy - int(anchor[1])))
        rmax = self.r_max if r_max is None else int(r_max)
        rmin = self.r_min if r_min is None else int(r_min)
        cand &= (d >= rmin) & (d <= rmax)
        # 后续 Pylon 要往外扩（不要贴着已有覆盖），其余建筑优先靠近基地
        if pylons and (need_power or require_power):
            cover = self._pylon_cover(pylons)
            if require_power:
                # Cyber / 护盾电池等**必须**在 Pylon 能量场内，否则 SC2 直接拒
                cand &= cover
            else:
                # Pylon 自己则优先往外铺，别都挤在已有覆盖里
                cand &= ~cover
        if not cand.any():
            return None
        pts = np.argwhere(cand)                     # (y, x)
        dist = d[pts[:, 0], pts[:, 1]]
        # 近的优先；同距离用 tick 打散，保证相邻帧不同点
        key = dist * 1000 + ((pts[:, 0] * 7 + pts[:, 1] * 13 + int(tick) * 31) % 97)
        y, x = pts[int(np.argmin(key))]
        return int(x), int(y)

    # ---------- 几何兜底 ----------
    def _ring(self, anchor):
        ax, ay = int(anchor[0]), int(anchor[1])
        pts = []
        for r in range(self.r_min, self.r_max + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    x, y = ax + dx, ay + dy
                    if 0 <= x < self.GRID and 0 <= y < self.GRID:
                        pts.append((x, y))
        return pts

    def pick(self, anchor, occupied, tick=0, mm=None, pylons=None, need_power=False,
             r_min=None, require_power=False):
        # 有地形图时只用地形结果：几何环带会给出悬崖/矿后/斜坡等非法格，
        # 一旦退回它就会「每帧发一次、每帧被拒」（实测 Pylon 连发不停）。
        if mm is not None:
            loc = self.pick_terrain(mm, anchor, occupied, tick, pylons, need_power,
                                    r_min=r_min, require_power=require_power)
            if loc is None and need_power:
                # 向外扩的要求找不到空地时，允许贴着已有 Pylon 放；
                # 但 r_min / require_power 是硬约束，必须保留（J144 修复）。
                loc = self.pick_terrain(mm, anchor, occupied, tick, pylons, False,
                                        r_min=r_min, require_power=require_power)
            if loc is None:
                # 只放宽半径，绝不丢掉尺寸/供电约束
                loc = self.pick_terrain(mm, anchor, occupied, tick, pylons, False,
                                        r_max=self.r_max + 6, r_min=r_min,
                                        require_power=require_power)
            return loc
        occ = {(int(x), int(y)) for x, y in occupied}
        cand = []
        for x, y in self._ring(anchor):
            if any(abs(x - ox) <= self.exclude and abs(y - oy) <= self.exclude
                   for ox, oy in occ):
                continue
            cand.append((x, y))
        if not cand:
            return None
        return cand[(int(tick) * 7 + 3) % len(cand)]


# 最小中心距：Nexus 占 5x5(±2)，3x3 建筑中心必须 >=4 才不压 Nexus；
# Pylon 2x2(±1) 可以贴到 3。之前统一用 2，导致 Gateway/Cyber 一直被拒。
BUILD_RMIN = {
    "build_pylon": 3, "build_gateway": 5, "build_cyber": 5,
    "build_assimilator": 4, "build_nexus": 5, "build_battery": 4,
}

_PICKER = LocationPicker()


# Protoss 的 Gateway / CyberneticsCore / ShieldBattery **都必须**在 Pylon
# 能量场内。J141 只列了 Cyber 与 Battery，漏掉 Gateway ⇒ 实测 Gateway 动作
# 被放到无电区域后全部被 SC2 拒绝（J144 修复）。
NEEDS_POWER = {"build_gateway", "build_cyber", "build_battery"}


def _pick(summ, tick, need_power=False, name=None):
    """统一选址：地形优先，几何兜底。name 用来取该建筑的最小中心距/供电要求。"""
    return _PICKER.pick(summ["anchor"] or (19, 23), summ["occupied"], tick,
                        mm=summ.get("mm"), pylons=summ.get("pylons"),
                        need_power=need_power,
                        r_min=BUILD_RMIN.get(name or "", None),
                        require_power=(name in NEEDS_POWER))


# ---------------------------------------------------------------- summarize

def _score(obs, key):
    """读 score_cumulative 的某一项。**只有 2D（多帧）才取最后一行**，
    1D 直接索引——之前无条件 v[-1] 会让矿恒等于 50。"""
    try:
        arr = obs.get("score_cumulative")
        if arr is None:
            return 0.0
        if getattr(arr, "ndim", 1) > 1:
            arr = arr[-1]
        return float(arr[key])
    except Exception:
        return 0.0


def _food_used(obs):
    try:
        arr = obs.get("score_by_category")
        if arr is None:
            return 0
        if getattr(arr, "ndim", 1) > 1:
            arr = arr[-1]
        return int(arr[0])
    except Exception:
        return 0


# FeatureUnit 字段下标（真实环境是一行 NamedNumpyArray，支持名字索引；
# 自测用普通 ndarray，只认整数，故做一层兼容）
FU = {
    "unit_type": 0, "alliance": 1, "health": 2, "build_progress": 6,
    "x": 12, "y": 13, "is_selected": 17, "is_powered": 19,
    "order_length": 26, "tag": 29,
}


def _uf(u, name):
    try:
        return u[name]
    except Exception:
        return u[FU[name]]



MINERAL_TYPES = (341, 483, 146, 147, 148, 149, 150, 151)   # MineralField / 750 / Rich


def _near_mines(neutral, anchor, limit=8):
    """离基地最近的矿，返回 [(x, y, idx)]。

    idx 是**该单位在 raw_units 里的下标**——pysc2 的 transform_action 用
    `_raw_tags[position]` 解析 target_unit_tag，传真实 tag 会被当成越界下标而失败。
    """
    ms = [(u["x"], u["y"], u["idx"]) for u in neutral if u["type"] in MINERAL_TYPES]
    if not ms:
        return []
    if anchor is None:
        return ms[:limit]
    ax, ay = anchor
    ms.sort(key=lambda t: (t[0] - ax) ** 2 + (t[1] - ay) ** 2)
    return ms[:limit]


def summarize(obs):
    ru = obs.get("raw_units")
    units = []
    if ru is not None:
        for u in ru:
            units.append({
                "idx": len(units),
                "type": int(_uf(u, "unit_type")), "alliance": int(_uf(u, "alliance")),
                "x": int(_uf(u, "x")), "y": int(_uf(u, "y")), "tag": int(_uf(u, "tag")),
                "hp": float(_uf(u, "health")), "build": float(_uf(u, "build_progress")),
                "orders": int(_uf(u, "order_length")), "powered": int(_uf(u, "is_powered")),
            })

    mine = [u for u in units if u["alliance"] == 1]
    their = [u for u in units if u["alliance"] == 4]
    neutral = [u for u in units if u["alliance"] == 3]

    def cnt(seq, t):
        return sum(1 for u in seq if u["type"] == t)

    nexus = [u for u in mine if u["type"] == NEXUS]
    pylon = [u for u in mine if u["type"] == PYLON]
    pylon_done = [u for u in pylon if u["build"] >= 1.0]
    gateway = [u for u in mine if u["type"] == GATEWAY]
    warp = [u for u in mine if u["type"] == WARPGATE]
    cyber = [u for u in mine if u["type"] == CYBER]
    assim = [u for u in mine if u["type"] == ASSIMILATOR]
    battery = [u for u in mine if u["type"] == SHIELDBATTERY]

    army = [u for u in mine if u["type"] in ARMY]
    probes = [u for u in mine if u["type"] == PROBE]

    food_cap = 15 * len(nexus) + 8 * len(pylon_done)
    food_used = _food_used(obs)
    if food_used <= 0:
        food_used = len(probes) + 2 * len(army)

    minerals = _score(obs, 7) - _score(obs, 11) + 50.0
    vespene = _score(obs, 8) - _score(obs, 12)

    busy = {
        "nexus": any(u["orders"] != 0 for u in nexus),
        "gateway": any(u["orders"] != 0 for u in gateway),
        "cyber": any(u["orders"] != 0 for u in cyber),
        "building_pylon": any(u["build"] < 1.0 for u in pylon),
        "building_gateway": any(u["build"] < 1.0 for u in gateway),
        "building_cyber": any(u["build"] < 1.0 for u in cyber),
        "building_assim": any(u["build"] < 1.0 for u in assim),
        "building_nexus": any(u["build"] < 1.0 for u in nexus),
        "building_battery": any(u["build"] < 1.0 for u in battery),
    }

    enemy_near = 1e9
    enemy_near_pos = None
    if their and nexus:
        ax0 = sum(u["x"] for u in nexus) / len(nexus)
        ay0 = sum(u["y"] for u in nexus) / len(nexus)
        for u in their:
            d2 = (u["x"] - ax0) ** 2 + (u["y"] - ay0) ** 2
            if d2 < enemy_near:
                enemy_near = d2
                enemy_near_pos = (u["x"], u["y"])
        enemy_near = enemy_near ** 0.5

    enemy_center = None
    if their:
        enemy_center = (sum(u["x"] for u in their) / len(their),
                        sum(u["y"] for u in their) / len(their))
    elif nexus:
        ax = sum(u["x"] for u in nexus) / len(nexus)
        ay = sum(u["y"] for u in nexus) / len(nexus)
        enemy_center = (63 - ax, 63 - ay)

    anchor = None
    if nexus:
        anchor = (int(sum(u["x"] for u in nexus) / len(nexus)),
                  int(sum(u["y"] for u in nexus) / len(nexus)))

    return {
        "n_probe": len(probes), "n_nexus": len(nexus),
        "n_pylon": len(pylon), "n_pylon_done": len(pylon_done),
        "n_gateway": len(gateway), "n_warpgate": len(warp),
        "n_cyber": len(cyber), "n_cyber_done": sum(1 for u in cyber if u["build"] >= 1.0),
        "n_assim": len(assim), "n_battery": len(battery),
        "n_army": len(army), "n_zealot": cnt(mine, ZEALOT),
        "n_stalker": cnt(mine, STALKER), "n_sentry": cnt(mine, SENTRY),
        "n_adept": cnt(mine, ADEPT),
        "food_used": food_used, "food_cap": food_cap,
        "minerals": minerals, "vespene": vespene, "busy": busy,
        "anchor": anchor, "enemy_center": enemy_center,
        "enemy_near": enemy_near, "enemy_near_pos": enemy_near_pos,
        "occupied": [(u["x"], u["y"]) for u in units],
        "geysers": [(u["x"], u["y"], u["idx"]) for u in neutral if u["type"] == GEYSER],
        "assimilators": [(u["x"], u["y"], u["idx"]) for u in assim if u["build"] >= 1.0],
        "pylons": [(u["x"], u["y"]) for u in pylon_done],
        "n_idle_probe": sum(1 for u in probes if u["orders"] == 0),
        "mineral_fields": _near_mines(neutral, anchor),
        "mm": obs.get("feature_minimap"),
        "raw": units,
    }


# ---------------------------------------------------------------- build order

def _have_counts(summ):
    """从当前状态推断「各类建筑/科技已完成的个数」，用于推进 build order。"""
    return {
        "build_pylon": summ["n_pylon"],
        "build_gateway": summ["n_gateway"] + summ["n_warpgate"],
        "build_cyber": summ["n_cyber"],
        "build_assimilator": summ["n_assim"],
        "build_nexus": max(0, summ["n_nexus"] - 1),      # 初始那个不算
        "build_battery": summ["n_battery"],
        "research_warpgate": 1 if (summ["n_warpgate"] > 0
                                   or summ["n_cyber_done"] > 0
                                   and summ["busy"]["cyber"]) else 0,
    }


def next_build_step(summ, build=DEFAULT_BUILD):
    """按 supply 门槛找下一个「该做且还没做」的建筑/科技步骤。

    返回 (action_name, supply, step_index) 或 (None, None, None)。
    """
    b = BUILDS.get(build)
    if not b:
        return None, None, None
    have = _have_counts(summ)
    want = {}
    for idx, (supply, raw_name) in enumerate(b["steps"]):
        act = ACTION_MAP.get(raw_name)
        if not act:                      # CHRONO / WORKER_SCOUT 等本项目无 head 的步骤
            continue
        if act not in have:              # Robo 系暂未实现
            continue
        if act == "build_nexus" and not ALLOW_EXPAND:
            continue
        want[act] = want.get(act, 0) + 1
        if have.get(act, 0) >= want[act]:
            continue
        if summ["food_used"] < supply:
            continue
        # 视野内没有气矿时跳过 GAS（否则会卡在这一步永远不推进）
        if act == "build_assimilator" and not summ["geysers"]:
            continue
        return act, supply, idx
    return None, None, None


def _can(summ, name):
    m, v = COST.get(name, (0, 0))
    return summ["minerals"] >= m and summ["vespene"] >= v


# ---------------------------------------------------------------- decide

def decide(summ, tick=0, build=DEFAULT_BUILD, style="tech"):
    """返回 (intervene: bool, action: dict|None, reason: str)。

    action = {"name", "id", "kind", "unit_type_hint", "place", "target_tag"}
    """
    fu, fc = summ["food_used"], summ["food_cap"]
    busy = summ["busy"]
    anchor = summ["anchor"] or (19, 23)

    def mk(name, kind, hint=None, place=None, target_tag=None):
        return {"name": name, "id": ACT[name], "raw_name": ACT_RAW[name], "kind": kind,
                "unit_type_hint": hint, "place": place, "target_tag": target_tag}

    # P0 卡人口：留 2 人口余量就补（预留 1 人口给正在造的兵，比卡死更稳）
    if fc > 0 and fu >= fc - 2 and not busy["building_pylon"] and _can(summ, "build_pylon"):
        loc = _pick(summ, tick, name="build_pylon")
        if loc:
            return True, mk("build_pylon", "build", PROBE, loc), "supply_low"

    # P0.5 采矿：有空闲 Probe 就去最近矿（每 8 tick 一次，避免抢占建造帧）
    if tick % 8 == 0 and summ.get("n_idle_probe", 0) > 0 and summ.get("mineral_fields"):
        mf = summ["mineral_fields"][0]
        return True, mk("harvest", "unit", PROBE, None, mf[2]), "gather_minerals"

    # P1 build order 的下一个建筑/科技
    act, supply, idx = next_build_step(summ, build)
    if act:
        already = {
            "build_pylon": busy["building_pylon"],
            "build_gateway": busy["building_gateway"],
            "build_cyber": busy["building_cyber"],
            "build_assimilator": busy["building_assim"],
            "build_nexus": busy["building_nexus"],
            "build_battery": busy["building_battery"],
            "research_warpgate": busy["cyber"],
        }.get(act, False)
        if not already and _can(summ, act):
            reason = "bo#%d_%s@%d" % (idx, act, supply)
            if act == "build_assimilator":
                if not summ["geysers"]:
                    pass
                else:
                    gx, gy, gtag = summ["geysers"][0]
                    return True, mk(act, "build", PROBE, (gx, gy), gtag), reason
            elif act == "research_warpgate":
                return True, mk(act, "research", CYBER), reason
            else:
                loc = _pick(summ, tick + (idx % 5) + 1, need_power=(act == "build_pylon"), name=act)
                if loc:
                    return True, mk(act, "build", PROBE, loc), reason

    # P1.5 采气：必须在 P2/P3 之前，否则产兵/补农民每帧都占满决策，
    #      采气永远轮不到 -> vespene 恒为 0 -> 只能出 Zealot（实测）。
    #      每个气厂只派 3 个 Probe（gas_sent 由实盘脚本回传）。
    gas_target = 3 * len(summ.get("assimilators") or [])
    if gas_target and summ.get("gas_sent", 0) < gas_target \
            and summ["n_probe"] >= 12:
        ax, ay, aidx = summ["assimilators"][0]
        return True, mk("gather_gas", "unit", PROBE, None, aidx), "gather_gas"

    # P2 经济：Nexus 空闲且 Probe 未到上限，且人口不紧（紧时攒钱补 Pylon）
    worker_till = BUILDS.get(build, {}).get("constant_worker_till", 22)
    probe_target = min(worker_till, 22 * max(1, summ["n_nexus"]))
    supply_tight = (fc > 0 and fu >= fc - 2)
    if not supply_tight and not busy["nexus"] and summ["n_probe"] < probe_target \
            and _can(summ, "train_probe"):
        return True, mk("train_probe", "train", NEXUS), "train_probe"

    # P3 产兵：Gateway 空闲（Stalker 需 Cyber 完成才解锁）
    #    兵种配比（地面，不造空军）：Stalker 为主，气不足时补 Adept/Zealot，
    #    气多且 Sentry 少于 2 个时补 Sentry（护盾+力场）。
    if summ["n_gateway"] > 0 and not busy["gateway"]:
        tech_ready = summ["n_cyber_done"] > 0
        if tech_ready and summ["n_sentry"] < 2 and summ["vespene"] >= 100 \
                and _can(summ, "train_sentry"):
            return True, mk("train_sentry", "train", GATEWAY), "train_sentry"
        if tech_ready and summ["vespene"] >= 50 and _can(summ, "train_stalker"):
            return True, mk("train_stalker", "train", GATEWAY), "train_stalker"
        if tech_ready and summ["vespene"] >= 25 and _can(summ, "train_adept"):
            return True, mk("train_adept", "train", GATEWAY), "train_adept"
        if _can(summ, "train_zealot"):
            return True, mk("train_zealot", "train", GATEWAY), "train_zealot"

    # P3.5 补产线：余矿多时把 Gateway 补到 3 个（build order 只给 1-2 个，
    #      难度 7 的内置 AI 出兵快，1 个兵营产兵跟不上）
    if summ["n_gateway"] < 4 and summ["food_cap"] >= 26 \
            and summ["minerals"] >= 250 and not busy["building_gateway"] \
            and _can(summ, "build_gateway"):
        loc = _pick(summ, tick + 11, name="build_gateway")
        if loc:
            return True, mk("build_gateway", "build", PROBE, loc), "more_gateway"

    # P4 兜底补人口（build order 已走完但人口仍紧）
    if fc > 0 and fu >= fc - 3 and not busy["building_pylon"] and _can(summ, "build_pylon"):
        loc = _pick(summ, tick + 7, name="build_pylon")
        if loc:
            return True, mk("build_pylon", "build", PROBE, loc), "supply_topup"

    # P4.5 防守：敌人压到基地 18 格内，部队立刻反击（优先于进攻）
    tgt = summ.get("enemy_near_pos") or summ["enemy_center"]
    if summ["n_army"] >= 3 and summ.get("enemy_near", 1e9) < 13 and tgt:
        return True, mk("attack", "attack", None,
                        (int(tgt[0]), int(tgt[1]))), "defend_base"

    # P5 进攻
    if summ["n_army"] >= 12 and summ["enemy_center"]:
        return True, mk("attack", "attack", None,
                        (int(summ["enemy_center"][0]), int(summ["enemy_center"][1]))), "attack"

    return False, None, "keep_policy"


# ---------------------------------------------------------------- self test

def _fake_obs(food_used, minerals, vespene, units):
    import numpy as np
    arr = np.zeros((len(units), 30), dtype=np.int64)
    for i, (t, x, y, b, o, a) in enumerate(units):
        arr[i, 0] = t      # unit_type
        arr[i, 1] = a      # alliance
        arr[i, 2] = 100    # health
        arr[i, 6] = int(b * 100) if b >= 1.0 else int(b * 100)  # build_progress (pysc2 是 0..1)
        arr[i, 6] = 1 if b >= 1.0 else 0
        arr[i, 12] = x
        arr[i, 13] = y
        arr[i, 26] = o     # order_length
        arr[i, 29] = i + 1000
    sc = np.zeros(13, dtype=np.float64)
    sc[7] = minerals
    sc[11] = 0.0
    sc[8] = vespene
    sc[12] = 0.0
    sbc = np.zeros(11, dtype=np.float64)
    sbc[0] = food_used
    return {"raw_units": arr, "score_cumulative": sc, "score_by_category": sbc}


def self_test():
    print("=== rule_v3 self test (build order 驱动) ===")
    print("已载入 builds:", list(BUILDS.keys()))
    assert BUILDS, "build order 未载入"

    b = DEFAULT_BUILD
    steps = BUILDS[b]["steps"]
    print("使用 build:", b, " 步数:", len(steps))

    # 模拟一局：从 12 probe 开始，按 supply 推进，每步只做一次动作
    units = [(NEXUS, 19, 23, 1.0, 0, 1)] + \
            [(PROBE, 17 + i % 3, 21 + i // 3, 1.0, 0, 1) for i in range(12)] + \
            [(GEYSER, 23, 19, 1.0, 0, 3), (GEYSER, 23, 27, 1.0, 0, 3)]
    done = []
    food = 12
    mins = 200
    gas = 0
    gas_sent = 0
    for t in range(60):
        s = summarize(_fake_obs(food, mins, gas, units))
        s["gas_sent"] = gas_sent
        ok, act, why = decide(s, tick=t, build=b)
        if not ok:
            mins += 100   # 攒钱
            continue
        done.append((food, act["name"], why))
        # 模拟执行效果：补单位/建筑，推进 supply
        if act["name"] == "build_pylon":
            units.append((PYLON, 22 + t % 5, 25 + t % 5, 1.0, 0, 1))
            mins -= 100
        elif act["name"] == "build_gateway":
            units.append((GATEWAY, 25 + t % 4, 27, 1.0, 0, 1))
            mins -= 150
        elif act["name"] == "build_cyber":
            units.append((CYBER, 28, 30, 1.0, 0, 1))
            mins -= 150
        elif act["name"] == "build_assimilator":
            units.append((ASSIMILATOR, 30, 31, 1.0, 0, 1))
            mins -= 75
        elif act["name"] == "train_probe":
            units.append((PROBE, 17, 22, 1.0, 0, 1))
            food += 1
            mins -= 50
        elif act["name"] == "train_stalker":
            units.append((STALKER, 20, 26, 1.0, 0, 1))
            food += 2
            mins -= 125
            gas -= 50
        elif act["name"] == "train_zealot":
            units.append((ZEALOT, 20, 26, 1.0, 0, 1))
            food += 2
            mins -= 100
        elif act["name"] == "train_adept":
            units.append((ADEPT, 20, 26, 1.0, 0, 1))
            food += 2
            mins -= 100
            gas -= 25
        elif act["name"] == "train_sentry":
            units.append((SENTRY, 20, 26, 1.0, 0, 1))
            food += 2
            mins -= 50
            gas -= 100
        elif act["name"] == "gather_gas":
            gas_sent += 3
        elif act["name"] == "research_warpgate":
            mins -= 50
            gas -= 50
        elif act["name"] == "attack":
            break
        mins += 60
        gas += 25

    print("\n--- 执行序列（supply, 动作, 原因）---")
    for f, a, w in done:
        print("  %3d  %-22s %s" % (f, a, w))

    names = [a for _, a, _ in done]
    assert "build_pylon" in names, "未补人口"
    assert "build_gateway" in names, "未造兵营"
    assert "build_cyber" in names, "未造 Cyber"
    assert "train_probe" in names, "未造农民"
    assert any(n in names for n in ("train_stalker", "train_adept", "train_sentry",
                                     "train_zealot")), "未产兵"
    uniq = len(set(names))
    assert uniq >= 5, "动作多样性不足：%d 种" % uniq
    print("\n动作种类 =", uniq, sorted(set(names)))
    print("SELF-TEST PASS：build order 驱动，动作多样，覆盖 人口/兵营/科技/经济/产兵")
    print("VERSION =", VERSION)


if __name__ == "__main__":
    self_test()
