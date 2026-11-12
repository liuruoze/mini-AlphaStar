# -*- coding: utf-8 -*-
"""v4：纯 Zealot rush 规则专家（可独立打赢难度 1）。

与 v3 的差别
------------
1. **不依赖 build order 表**。v3 会卡在 `build_assimilator`（Zealot 不吃气），
   `next_build_step()` 每帧返回同一步，把后面的补兵营/造兵/防守规则全部堵死
   （实测 6,000 步零介入）。
2. **每帧都能出动作**，优先级：人口/首 Pylon → 农民 → 兵营 → 造兵 → 防守 → 进攻 → 采矿。
3. **防守优先于进攻**：敌人压到基地 14 格内先回防，否则攒够 4 个就 A-move。
4. 目标：**learner 全程 noop 时也能独立打赢难度 1（very_easy）**。

复用 v3 的 `summarize()` / `_pick()` / 常量表，只重写决策层。

选址层三条硬约束（都是实测踩出来的）
------------------------------------
- `raw_units` 的 x,y 是 **minimap 尺度**（scale = 64/96），`transform_action` 内部
  用 `back_pt` 转世界坐标 ⇒ **直接传 minimap 坐标是对的**。
- 可建性用 SC2 官方 `placement_grid`（88×96 world），且必须校验**整块 footprint**。
- Protoss 建筑必须落在 Pylon 能量场内（6.5 世界格 ≈ 4.33 minimap 格），
  **且能量场不跨高度层** ⇒ 候选点必须与 Pylon 同 height_map 高度、且已探索。
"""
import os as _os

import numpy as _np

from _rule_expert_v3 import (  # noqa: F401
    summarize, _pick, ACT, ACT_RAW,
    NEXUS, PYLON, ASSIMILATOR, GATEWAY, PROBE, ZEALOT, ARMY,
)

# ---- 追加动作：取消 Nexus 队列里排队的 Probe ----
# SC2 里「排队中」的单位可以全额取消（返还 50 矿）。learner 在 medium 下会疯狂造农民
# （实测堆到 55 个、只出 2 个 Zealot），双命令并行撤不掉它的命令，只能靠取消腾矿。
ACT.setdefault("cancel_probe", 129)
ACT_RAW.setdefault("cancel_probe", "Cancel_Last_quick")

# ---- 可调参数 ----
PROBE_TARGET = 14      # 农民上限 14（10 个时收入 420 矿/分撑不起 15 兵，放到 14）
GATEWAY_TARGET = 3     # 三兵营：双兵营产能追不上 medium 的爆兵
ZEALOT_MAX = 30        # 造兵上限（防止矿全砸兵不出门）
ATTACK_AT = 6          # 兵不到 6 个不进攻（2026-11-10 21:14 用户指定）
DEFEND_AT = 8          # 敌人贴脸才回防（14 太远，兵被一个个调回来送掉）
SUPPLY_RESERVE = 6     # 人口留多少余量就补 Pylon（双兵营吃人口快，4->6）

BUILD_WAIT = 150       # 一条建造令的观察窗口（帧）：期内不重复下单
GATEWAY_WAIT = 70

_BUILDING_TYPES = {59, 60, 61, 62, 69, 70, 71, 72, 73, 74}

# SC2 官方可建位图（88x96 world，1=可建）。由 rule_bot_play.run_one 在 reset 后注入。
# 换算实测：back((19,23))=(29.25,60.75) ⇒ world_x = mmx*1.5+0.75, world_y = 96-(mmy*1.5+0.75)
PLACEMENT = None
_MAP_H = 96.0

# Pylon 建造进度到 1.0 的那一帧，能量场还没在引擎侧生效，
# 同一帧下令建 Gateway 会被 SC2 静默拒掉（实测首局 1/1 失败、后续局 0 失败）。
POWER_READY_DELAY = int(_os.environ.get("MAS_V4_POWER_DELAY", "3"))
_PYLON_DONE = {}      # pylon tag -> 首次观测到 build>=1.0 的 tick

# 在途建造令：key -> (tick, 下单时的实体数, 坐标)。
# 没有它时 P0「补人口」与 P1.5「为 Gateway 建 Pylon」两套冷却各自下令，
# 同一格被多个 Probe 抢建（实测一局堆到 20 个 Pylon、矿堆 8,245 却不造兵）。
_INFLIGHT = {}
_GWFAIL = [0]      # Gateway 连续「下令后实体没出现」的次数：>=2 就先补 Pylon 扩能量场
_ATTACKING = [False]   # 进攻态：一旦攒够 6 个开打就保持，新出的兵立刻全部跟进
# 连续失败的点：下令后整个观察窗口内实体数没增加 ⇒ 该格不可建/被占，本局不再试。
_BAD_POS = set()

# 以 anchor 为中心的候选偏移（先下方/侧方；Simple64 出生点上方就是矿脉）
_ANCHOR_OFF = [(0, 4), (4, 0), (-4, 0), (0, -4), (4, 4), (-4, 4),
               (5, 0), (-5, 0), (0, 5), (0, 6), (0, 9), (3, 4), (-3, 4),
               (6, 4), (-6, 4), (6, 0), (-6, 0), (3, -4), (-3, -4),
               (8, 6), (-8, 6), (6, 8), (-6, 8), (10, 2), (-10, 2),
               (0, 12), (8, 10), (-8, 10), (0, -6), (6, -4), (-6, -4)]

# 以 Pylon 为中心的环形偏移。
# 能量场半径 6.5 世界格，但必须覆盖 Gateway 整块 footprint（3×3 world，半径 1.5），
# 所以圆心距上限 = 6.5-1.5 = 5.0 world = 3.33 minimap ⇒ d2 <= 11。
# 上限放到 4.24 时实测 57% 的 Gateway 命令被 SC2 拒（点在能量场边缘外）。
_RING_OFF = sorted(
    [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5)
     if 4 <= dx * dx + dy * dy <= 11],
    key=lambda d: (d[0] * d[0] + d[1] * d[1], abs(d[0]) + abs(d[1])))

# Pylon 能量场可用的圆心距（minimap 格）：6.5 world 半径 - 1.5 world footprint = 5.0 world
POWER_R = 3.33


def reset():
    """每局开始前清空跨局状态（tick 每局从 1 重计，残留会误判冷却/延迟）。"""
    _LAST_TRY.clear()
    _PYLON_DONE.clear()
    _INFLIGHT.clear()
    _BAD_POS.clear()
    _GWFAIL[0] = 0
    _ATTACKING[0] = False
    _DBG[0] = 0


def _ready_pylons(summ, tick):
    """返回能量场已生效的 Pylon 坐标列表（完成后再等 POWER_READY_DELAY 帧）。"""
    out = []
    for u in _mine(summ):
        try:
            if int(u["type"]) != PYLON or float(u["build"]) < 1.0:
                continue
            tag = int(u["tag"])
            t0 = _PYLON_DONE.get(tag)
            if t0 is None:
                t0 = int(tick)
                _PYLON_DONE[tag] = t0
            if int(tick) - t0 >= POWER_READY_DELAY:
                out.append((int(u["x"]), int(u["y"])))
        except Exception:
            pass
    return out


def _world_ok(mmx, mmy, r_world):
    """检查以 (mmx,mmy) 为中心、半径 r_world（世界格）的方块是否全部可建。"""
    if PLACEMENT is None:
        return True
    h, w = PLACEMENT.shape
    wx = mmx * 1.5 + 0.75
    wy = _MAP_H - (mmy * 1.5 + 0.75)
    x0, x1 = int(round(wx - r_world)), int(round(wx + r_world))
    y0, y1 = int(round(wy - r_world)), int(round(wy + r_world))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if not (0 <= x < w and 0 <= y < h):
                return False
            if int(PLACEMENT[y, x]) == 0:
                return False
    return True


def _occ_r(t):
    """占据半径（minimap 格）= 该建筑半径 + 目标建筑半径（Gateway 1.0）。

    Nexus 5×5 world = 1.67 minimap、Pylon 2×2 = 0.67、其余 3×3 = 1.0。
    中心距小于两者半径之和就是 footprint 重叠，SC2 静默拒建。
    早先对所有建筑统一用 2 格，导致 Gateway 贴着 Nexus 的点被反复选中又失败。
    """
    if t == NEXUS:
        return 2.7
    if t == PYLON:
        return 1.7
    return 2.0


def _ctx(summ):
    """选址上下文：占据点(带半径) / 矿与气 / 高度图 / 可见性图（都是 minimap 尺度）。"""
    occ = []
    for u in _mine(summ):
        try:
            if (float(u["build"]) < 1.0
                    or int(u["type"]) in _BUILDING_TYPES):
                t = int(u["type"])
                occ.append((int(u["x"]), int(u["y"]), _occ_r(t)))
        except Exception:
            pass
    mines = []
    for mf in (summ.get("mineral_fields") or []):
        try:
            mines.append((int(mf[0]), int(mf[1])))
        except Exception:
            pass
    for g in (summ.get("geysers") or []):
        try:
            mines.append((int(g[0]), int(g[1])))
        except Exception:
            pass
    hm = vm = None
    mm = summ.get("mm")
    if mm is not None:
        try:
            hm = _np.asarray(mm["height_map"])
            vm = _np.asarray(mm["visibility_map"])
        except Exception:
            hm = vm = None
    return occ, mines, hm, vm


def _h_at(grid, x, y):
    if grid is None:
        return None
    try:
        h, w = grid.shape
        if 0 <= x < w and 0 <= y < h:
            return int(grid[int(y), int(x)])
    except Exception:
        pass
    return None


def _site_ok(tx, ty, occ, mines, hm, vm, ref_h, size, half, near=None, tag=""):
    """候选格是否可建：边界 / 黑名单 / 占据 / 矿气 / 已探索 / 同高度层 / footprint / 能量场。"""
    pad = max(0.0, float(size) - 2.0)

    def _no(why):
        if _os.environ.get("MAS_V4_DEBUG2") == "1":
            print("[site] %-10s (%2d,%2d) reject: %s" % (tag, tx, ty, why), flush=True)
        return False

    if not (1 <= tx <= 62 and 1 <= ty <= 62):
        return _no("bounds")
    if (tx, ty) in _BAD_POS:
        return _no("badpos")
    for ox, oy, orr in occ:
        if abs(tx - ox) < orr + pad and abs(ty - oy) < orr + pad:
            return _no("occ%d,%d" % (ox, oy))
    for mx, my in mines:
        # 矿脉是 2x1 的细条，中心到边缘约 0.7 minimap；建筑半径 1.0
        # ⇒ 中心距 1.7 就够。原来用 2.2，把 Pylon 北侧整片可用地都判死了
        #（medium 实测 mine 拒 11 次，是 Gateway 建不出的第一原因）。
        if abs(tx - mx) < 1.7 and abs(ty - my) < 1.7:
            return _no("mine")
    if vm is not None:
        h = _h_at(vm, tx, ty)
        if h is None or h <= 0:      # 未探索区下令必失败，且 SC2 不回原因
            return _no("unexplored")
    if ref_h is not None:
        # Pylon 能量场不跨高度层：跨层即使平面距离 <6.5 也会被拒
        h = _h_at(hm, tx, ty)
        # 允许 ±1：height_map 是原始高度值，同一平台上常有 1 级噪声，
        # 精确判等会把整片合法落点判死（实测 Pylon 周围 ring 全被 height 拒）。
        if h is None or abs(int(h) - int(ref_h)) > 1:
            return _no("height")
    if not _world_ok(tx, ty, half):
        return _no("placement")
    if near is not None and not any((tx - px) ** 2 + (ty - py) ** 2 <= near[0] ** 2
                                    for px, py in near[1]):
        return _no("power")
    return True


def _pylon_spots(summ):
    """矿脉外侧的固定 Pylon 候选（替代通用偏移表）。

    `v4spots.py` 用 placement_grid 离线枚举了左上基地（anchor=(19,23)）周围 ±12 格：
    可建区集中在**矿脉外侧**（y<19 那一边），最优 Pylon 点 (19,16)/(18,17) 一带，
    其能量场内能放下 28 个 Gateway 落点；而 Nexus 正下方/东侧几乎全是斜坡与悬崖
    （实测 Pylon 被逼到 (23,23)，Gateway 一个点都没有）。
    所以这里按「Nexus -> 矿脉质心」方向往外推，两个出生点（左上/右下）通用。
    """
    anchor = summ.get("anchor") or (19, 23)
    ax, ay = int(anchor[0]), int(anchor[1])
    # 只用「最近那个矿在我方上方还是下方」这一条稳健信息定方向。
    # 早先用 8 个矿的质心，但 mineral_fields 会混进远处的第三片矿，
    # 质心被拉偏 ⇒ Pylon 跑到距 Nexus 6 格外的平台边缘 (13,19)、Gateway 全废。
    mfs = summ.get("mineral_fields") or []
    sy = -1
    if mfs:
        try:
            sy = 1 if float(mfs[0][1]) > ay else -1
        except Exception:
            sy = -1
    out = []
    for d in (6, 7, 8, 5, 9):              # 矿那一侧、再往外 6~9 格
        for dx in (0, -1, 1, -2, 2):
            out.append((ax + dx, ay + sy * d))
    return out


def _gw_room(occ, mines, hm, vm, px, py, ref_h, need=2, half=1.6):
    """该 Pylon 点的能量场内，还能放下几个合法 Gateway（>= need 才算好点）。

    medium 实测教训：Pylon 落在平台边缘 (13,19) 时，自己的 footprint 合法，
    但周围 3.3 格外全是高度断层（height 拒绝）⇒ Gateway 整局建不出来、
    矿堆到 1,785 也花不掉。选 Pylon 时就必须替 Gateway 留位置。
    """
    ph = _h_at(hm, px, py) if ref_h is None else ref_h
    n = 0
    for dx, dy in _RING_OFF:
        if _site_ok(px + dx, py + dy, occ, mines, hm, vm, ph, 2, half, None, ""):
            n += 1
            if n >= need:
                return True
    return False


def _simple_pick(summ, tick=0, size=2, avoid_mines=True, near_pylon=False,
                 half=1.5):
    """以 Nexus 为锚按偏移表选址（首 Pylon / 补人口用）。"""
    anchor = summ.get("anchor") or (19, 23)
    ax, ay = int(anchor[0]), int(anchor[1])
    occ, mines, hm, vm = _ctx(summ)
    if not avoid_mines:
        mines = []
    pylons = _ready_pylons(summ, tick)
    near = (POWER_R, pylons) if near_pylon else None
    if near_pylon and not pylons:
        return None
    ref_h = _h_at(hm, ax, ay)
    if not near_pylon:
        # 先试「矿脉外侧」的固定点（离线枚举得出），再退回通用偏移表
        for tx, ty in _pylon_spots(summ):
            if (_site_ok(tx, ty, occ, mines, hm, vm, ref_h, size, half, None, "spot")
                    and _gw_room(occ, mines, hm, vm, tx, ty, ref_h)):
                if _os.environ.get("MAS_V4_DEBUG") == "1":
                    print("[pick] spot anchor=%s -> (%d,%d)" % ((ax, ay), tx, ty),
                          flush=True)
                return (tx, ty)
    for dx, dy in _ANCHOR_OFF:
        tx, ty = ax + dx, ay + dy
        if _site_ok(tx, ty, occ, mines, hm, vm, ref_h, size, half, near, "anchor"):
            if _os.environ.get("MAS_V4_DEBUG") == "1":
                print("[pick] anchor=%s pylons=%s -> (%d,%d)"
                      % ((ax, ay), pylons, tx, ty), flush=True)
            return (tx, ty)
    return None


def _pick_near_pylons(summ, pylons, size=2, half=1.5):
    """在已有 Pylon 的能量场内做环形搜索（Gateway/Cyber 必须这样选）。

    只按 anchor 偏移表找点会漏掉 Pylon 四周 45° 的可用格 —— 实测失败局
    Pylon 在基地边缘，anchor 偏移表里没有一个点同时满足「能量场内 + 同高度层
    + 整块可建」，Gateway 整局建不出来（killed 恒 0）。
    """
    if not pylons:
        return None
    occ, mines, hm, vm = _ctx(summ)
    for px, py in pylons:
        ph = _h_at(hm, px, py)
        for dx, dy in _RING_OFF:
            tx, ty = px + dx, py + dy
            if _site_ok(tx, ty, occ, mines, hm, vm, ph, size, half, None, "ring"):
                if _os.environ.get("MAS_V4_DEBUG") == "1":
                    print("[pick-near] pylon=(%d,%d) -> (%d,%d)" % (px, py, tx, ty),
                          flush=True)
                return (tx, ty)
    return None


_LAST_TRY = {}


def _try_once(key, tick, gap=150):
    """同一动作在 gap 帧内只尝试一次（保留给非建造分支用）。"""
    last = _LAST_TRY.get(key, -10 ** 9)
    if tick - last < gap:
        return False
    _LAST_TRY[key] = tick
    return True


def _inflight_block(key, tick, n_now, wait=BUILD_WAIT):
    """该建造是否还在观察窗口内。实体数增加即解除；窗口到期仍未增加 ⇒ 记黑名单。"""
    it = _INFLIGHT.get(key)
    if it is None:
        return False
    t0, n0, pos = it
    if int(n_now) > int(n0):
        _INFLIGHT.pop(key, None)
        if key == "gateway":
            _GWFAIL[0] = 0
        return False
    if int(tick) - int(t0) < wait:
        return True
    _INFLIGHT.pop(key, None)
    if key == "gateway":
        _GWFAIL[0] += 1
    if pos:
        _BAD_POS.add((int(pos[0]), int(pos[1])))
        if _os.environ.get("MAS_V4_DEBUG") == "1":
            print("[v4] %s at %s no-show -> blacklist" % (key, pos), flush=True)
    return False


def _inflight_set(key, tick, n_now, pos):
    _INFLIGHT[key] = (int(tick), int(n_now), pos)


def mk(name, kind, hint=None, place=None, target_tag=None):
    return {"name": name, "id": ACT[name], "raw_name": ACT_RAW[name],
            "kind": kind, "unit_type_hint": hint, "place": place,
            "target_tag": target_tag}


def _mine(summ):
    out = []
    for u in summ.get("raw", []):
        try:
            if int(u["alliance"]) == 1:
                out.append(u)
        except Exception:
            pass
    return out


def _gateway_idle(summ):
    """至少有一个 Gateway 已建完且当前没在生产。"""
    for u in _mine(summ):
        try:
            if (int(u["type"]) == GATEWAY
                    and float(u["build"]) >= 1.0
                    and int(u["orders"]) == 0):
                return True
        except Exception:
            pass
    return False


_DBG = [0]


def decide(summ, tick=0, **_kw):
    """返回 (intervene, action, reason)。每帧调用，保持顺序即优先级。"""
    if _os.environ.get("MAS_V4_DEBUG") == "1" and _DBG[0] < 6:
        _DBG[0] += 1
        print("[v4dbg] tick=%s anchor=%s n_nexus=%s n_probe=%s n_gateway=%s "
              "fu/fc=%s/%s mins=%s occ=%s mf=%s"
              % (tick, summ.get("anchor"), summ.get("n_nexus"), summ.get("n_probe"),
                 summ.get("n_gateway"), summ.get("food_used"), summ.get("food_cap"),
                 int(summ.get("minerals") or 0), len(summ.get("occupied") or []),
                 (summ.get("mineral_fields") or [])[:2]), flush=True)
    # 坏点黑名单会越积越多：一局打到中后期，anchor 周围可能全被拉黑，
    # 于是「基地还在、矿也够」却一个点都选不出来。每 600 tick 解禁一次。
    if tick and int(tick) % 150 == 0:
        _BAD_POS.clear()

    fu = int(summ.get("food_used", 0))
    fc = int(summ.get("food_cap", 0))
    mins = float(summ.get("minerals", 0) or 0)
    busy = summ.get("busy") or {}
    np_ = int(summ.get("n_probe", 0))
    ng = int(summ.get("n_gateway", 0))
    nz = int(summ.get("n_zealot", 0))
    n_pylon = int(summ.get("n_pylon", 0))
    n_pylon_done = int(summ.get("n_pylon_done", 0))
    enemy_near = float(summ.get("enemy_near", 1e9))

    # P000 农民超编：learner 只要在农民 > 10 时还排 Train_Probe，就连续取消
    # （每步一条 Cancel_Last，COOLDOWN=1 时可连发，等效"一次取消多个"）。
    # pysc2 没有 Cancel_Queue_5，只有 Cancel_Last_quick，靠高频连发达到同效果。
    if np_ > PROBE_TARGET:
        nqo = 0
        for u in _mine(summ):
            try:
                if int(u["type"]) == NEXUS:
                    nqo = max(nqo, int(u["orders"]))
            except Exception:
                pass
        if nqo >= 1 and _try_once("cancel_over", tick, gap=1):
            return True, mk("cancel_probe", "cancel", NEXUS), "cancel_over"

    # P00 开局抢矿（最高优先级）：Pylon 还没建时，learner 每步下单 Train_Probe
    # 会把 50 矿一直吃干，导致 v4 的「矿 >= 100 才建 Pylon」永远等不到
    # （实测第一个 Pylon 拖到 146 秒，标准开局 ~60 秒）。
    # 这里主动取消 Nexus 队列里排队的 Probe，把 50 矿抢回来。
    if n_pylon_done == 0 and ng == 0 and mins < 100 and np_ > 0:
        nq0 = 0
        for u in _mine(summ):
            try:
                if int(u["type"]) == NEXUS:
                    nq0 = max(nq0, int(u["orders"]))
            except Exception:
                pass
        if nq0 >= 1 and _try_once("cancel_early", tick, gap=1):
            return True, mk("cancel_probe", "cancel", NEXUS), "cancel_early"

    # P0 人口 + 首个 Pylon（Gateway 必须有能量场）。
    # 两条需求共用同一套在途记账：早先它们各有独立冷却，会同时各下一个 Pylon 令。
    # ⚠ 必须要求 np_ > 0：Probe 被打光时 fu 仍可能落在 fc-6 之上，
    # 于是每帧都想发 build_pylon，但没有 Probe 可派 ⇒ 实测 medium 一局 no_unit=47，
    # 而且这条排在 P1 前面，把"造 Probe 复活"也挡住了。
    need_pylon = ((fc > 0 and fu >= fc - SUPPLY_RESERVE)
                  or (ng < GATEWAY_TARGET and n_pylon_done == 0)
                  # Gateway 连续两次「下令但没出现」= 现有 Pylon 的能量场里没有合法落点，
                  # 先补一个 Pylon 把覆盖范围摊开（medium 实测：矿堆 1,770、2 个 Pylon、gw 恒 0）
                  or (ng < GATEWAY_TARGET and _GWFAIL[0] >= 2 and n_pylon_done < 5))
    if (need_pylon and np_ > 0 and not busy.get("building_pylon")
            and not _inflight_block("pylon", tick, n_pylon)):
        if mins >= 100:
            # Pylon 是 2x2 world（半径 1.0），这里必须用 half=1.0。
            # 之前走默认 1.5（按 4x4 检查），把 v4spots 枚举出来的好点全判死了
            # ⇒ Pylon 被逼到 (23,23) 悬崖边、Gateway 无处可放。
            loc = _simple_pick(summ, tick, size=2, half=1.0)
            if loc:
                _inflight_set("pylon", tick, n_pylon, loc)
                return True, mk("build_pylon", "build", PROBE, loc), "supply"

    # P1 农民：先铺到 16 个（低于 6 个连兵营都建不起）。
    # ⚠ np_==0 时不看 Nexus 忙不忙：medium 下 learner 会把单位送光，
    # 而 learner 一直占着 Nexus 队列 ⇒ 若还要求空闲，农民永远补不出来
    #（实测 4 局全负：fc_no_unit 256–280 次、used_points 0–300、our_units 1–7）。
    if np_ < PROBE_TARGET and (np_ == 0 or not busy.get("nexus")) and mins >= 50:
        # hint 必须是 NEXUS：Train_Probe_quick 是从 Nexus 训练，
        # 传 PROBE 会让 to_func_call 去找"空闲的 Probe"，而它们都在采矿 ⇒ no_unit
        return True, mk("train_probe", "train", NEXUS), "probe"

    # P2 兵营：需要先有 6 个农民支撑经济
    if (ng < GATEWAY_TARGET and np_ >= 6 and mins >= 150
            and not busy.get("building_gateway")
            and not _inflight_block("gateway", tick, ng, wait=GATEWAY_WAIT)):
        loc = _pick_near_pylons(summ, _ready_pylons(summ, tick), size=2, half=1.6)
        if not loc:
            loc = _simple_pick(summ, tick, size=2, near_pylon=True, half=1.6)
        if loc:
            _inflight_set("gateway", tick, ng, loc)
            return True, mk("build_gateway", "build", PROBE, loc), "gateway"

    # P2.5 缺矿造兵时，取消 Nexus 队列里还在排队的 Probe（每个返还 50 矿）。
    # 只在「有 Gateway（能造兵）+ 矿不够 100」时动手，且要求队列长度 >= 2
    # （长度 1 表示只有一个正在造，取消会白丢进度）。
    if ng > 0 and nz < ZEALOT_MAX and mins < 100:
        nq = 0
        for u in _mine(summ):
            try:
                if int(u["type"]) == NEXUS:
                    nq = max(nq, int(u["orders"]))
            except Exception:
                pass
        if nq >= 2 and _try_once("cancel_probe", tick, gap=6):
            return True, mk("cancel_probe", "cancel", NEXUS), "cancel_probe"

    # P3 造兵：Gateway 空闲就出 Zealot（优先级低于补人口，高于进攻）
    if ng > 0 and nz < ZEALOT_MAX and mins >= 100 and _gateway_idle(summ):
        return True, mk("train_zealot", "train", GATEWAY), "zealot"

    # P4 防守：敌人贴脸才回防，且要求至少 3 个兵
    #（r2/r3 实测：nz>0 就回防时，兵被一个一个调回去送掉，永远滚不起来）
    if nz > 0 and enemy_near < DEFEND_AT and summ.get("enemy_near_pos"):
        return True, mk("attack", "attack", None, summ["enemy_near_pos"]), "defend"

    # P5 进攻：攒够 ATTACK_AT 个才开打（用户 21:14 指定：不到 6 个不打）。
    # 一旦开打就进入「进攻态」并保持：
    #   * 所有兵一起 A-move（to_func_call 的 attack 分支不再按 12 个截断）；
    #   * 后面新出的兵一出来就立刻跟进（nz > 0 即继续下发）。
    if nz >= ATTACK_AT:
        _ATTACKING[0] = True
    if _ATTACKING[0] and nz > 0 and summ.get("enemy_center"):
        return True, mk("attack", "attack", None, summ["enemy_center"]), "attack_all"

    # P6 闲农民去采矿。
    # ⚠ 必须限流：n_idle_probe>0 几乎每帧都成立，不限流时 harvest 会占掉
    # 80% 以上的专家动作（实测 684/832），把造兵和进攻的机会全挤掉。
    if int(summ.get("n_idle_probe", 0)) > 0 and summ.get("mineral_fields"):
        if _try_once("harvest", tick, gap=16):
            mf = summ["mineral_fields"][0]
            return True, mk("harvest", "unit", PROBE, None, mf[2]), "gather"

    return False, None, "idle"
