"""从 pysc2（与 SC2 3.16 配套）的动作表反查 unit -> creation ability 映射。

python-sc2 用 `UnitTypeData.ability_id` 判断「造这个单位/建筑用哪个技能」，
但 SC2 3.16 的 unit_type 数据里这个字段恒为 0（4.0 之后才填）。
pysc2 的 `actions.FUNCTIONS` 是按 3.16 生成的、且函数名里带单位名，正好可以反查。

输出 experiments/sc2_unit_ability.json：
    {"units": {"84": 1003, ...},        # unit_type_id -> ability_id
     "upgrades": {"1": 1565, ...}}      # upgrade_id   -> ability_id
"""
import json
import os
import re

from pysc2.lib import actions, units, upgrades

SUFFIX = re.compile(r"_(quick|pt|unit|screen|minimap|targeted|target|location)+$")


def unit_name_map():
    """单位英文名 -> unit_type_id。"""
    out = {}
    for race in ("Protoss", "Terran", "Zerg", "Neutral"):
        group = getattr(units, race, None)
        if group is None:
            continue
        for name in dir(group):
            if name.startswith("_"):
                continue
            val = getattr(group, name)
            out.setdefault(name.lower(), int(val.value))
    return out


def upgrade_name_map():
    # pysc2 的 upgrades.Upgrades 是 EnumMeta，成员名即升级名
    return {m.name.lower(): int(m.value) for m in upgrades.Upgrades}


def strip_suffix(name):
    prev = None
    while prev != name:
        prev = name
        name = SUFFIX.sub("", name)
    return name


def main():
    unames = unit_name_map()
    upnames = upgrade_name_map()
    # uid -> {priority: ability_id}；_pt/_quick 优先（目标类型是点/无目标）
    cand = {}
    upgrade_ability = {}
    unknown = []

    def prio(nm):
        if nm.endswith(("_pt", "_quick")):
            return 0
        if nm.endswith(("_screen", "_minimap")):
            return 2
        return 1

    for f in actions.FUNCTIONS:
        ability_id = int(f.ability_id)
        if ability_id == 0:
            continue
        nm = f.name
        if nm.startswith(("Train", "Build", "Warp")):
            core = strip_suffix(nm)
            parts = core.split("_")
            rest = "_".join(parts[1:])
            uid = unames.get(rest.lower())
            if uid is None:
                unknown.append(nm)
                continue
            cand.setdefault(str(uid), {}).setdefault(prio(nm), ability_id)
        elif nm.startswith("Research_"):
            core = strip_suffix(nm)[len("Research_"):]
            up = upnames.get(core.lower())
            if up is None:
                unknown.append(nm)
                continue
            upgrade_ability.setdefault(str(up), ability_id)

    unit_ability = {k: v[min(v)] for k, v in cand.items()}

    minerals = sorted(v for k, v in unames.items() if "mineralfield" in k)
    geysers = sorted(v for k, v in unames.items() if "geyser" in k)
    out = {"units": unit_ability, "upgrades": upgrade_ability,
           "minerals": minerals, "geysers": geysers}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sc2_unit_ability.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("units=%d upgrades=%d unknown=%d -> %s"
          % (len(unit_ability), len(upgrade_ability), len(unknown), path))
    for k in ("84", "59", "60", "61", "62"):     # Probe/Nexus/Pylon/Gateway/Assimilator(候选)
        print("  unit %s -> %s" % (k, unit_ability.get(k)))
    print("  minerals:", minerals)
    print("  geysers:", geysers)
    print("  unknown 样例:", unknown[:10])


if __name__ == "__main__":
    main()
