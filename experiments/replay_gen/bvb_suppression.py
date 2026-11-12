#!/usr/bin/env python3
"""批量扫描 replay 的「逆风」程度（仅用于挑局，不用于生成训练数据）。

为什么要 --fog：
  训练数据用 disable_fog=False，raw_units 只含**视野内**敌方单位，
  而我方含全部农民+建筑 ⇒ 「敌方>我方」被系统性压低（实测均势局仍 0%）。
  关掉迷雾只用于**测量真实兵力对比**，据此挑出「我方真被打」的局；
  这些局仍然按 disable_fog=False 转换，不引入信息泄漏。

用法:
  python experiments/replay_gen/bvb_suppression.py --fog --step 8 --csv out.csv \
      --glob '/home/liuruoze/bvb/even/*.SC2Replay'
"""
import argparse
import glob
import os
import sys

from absl import flags
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

flags.FLAGS(["x"])

MAP = os.path.expanduser("~/StarCraftII/Maps/Melee/Simple64.SC2Map")


def scan(rc, map_data, path, max_obs, player, step, fog):
    with open(path, "rb") as f:
        data = f.read()
    res = dict(obs=0, loops=0, our_mean=0.0, enemy_mean=0.0, our_max=0, enemy_max=0,
               suppressed=0, weak=0, both=0, our_end=0, enemy_end=0)
    with rc.start() as c:
        c.start_replay(sc_pb.RequestStartReplay(
            replay_data=data,
            map_data=map_data,
            observed_player_id=player,
            options=sc_pb.InterfaceOptions(raw=True, score=False),
            disable_fog=fog,
            realtime=False))
        obs = sup = weak = both = 0
        our_sum = enemy_sum = 0
        our_max = enemy_max = 0
        end_loop = 0
        n1 = n4 = 0
        while obs < max_obs:
            try:
                o = c.observe()
            except Exception:
                break
            gl = o.observation.game_loop
            if gl >= 2 ** 32 - 1:
                break
            end_loop = gl
            n1 = n4 = 0
            for u in o.observation.raw_data.units:
                if u.alliance == 1:
                    n1 += 1
                elif u.alliance == 4:
                    n4 += 1
            obs += 1
            our_sum += n1
            enemy_sum += n4
            our_max = max(our_max, n1)
            enemy_max = max(enemy_max, n4)
            if n4 > n1:
                sup += 1
                if n1 <= 2:
                    both += 1
            if n1 <= 2:
                weak += 1
            try:
                c.step(step)
            except Exception:
                break
        try:
            c.leave()
        except Exception:
            pass
    res.update(obs=obs, loops=end_loop,
               our_mean=our_sum / max(obs, 1), enemy_mean=enemy_sum / max(obs, 1),
               our_max=our_max, enemy_max=enemy_max,
               suppressed=sup, weak=weak, both=both,
               our_end=n1, enemy_end=n4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="*")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-obs", type=int, default=20000)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--player", type=int, default=1)
    ap.add_argument("--fog", action="store_true", help="disable_fog=True（仅测量用）")
    args = ap.parse_args()

    files = list(args.replays)
    if args.glob:
        files += sorted(glob.glob(args.glob))
    if not files:
        print("no replay given", file=sys.stderr)
        return 1

    map_data = open(MAP, "rb").read()
    rc = run_configs.get()
    rows = []
    print("%-24s %5s %6s %6s %6s %6s %9s %9s" %
          ("replay", "obs", "our", "enemy", "ourMx", "enMx", "sup>=1x", "weak<=2"))
    for p in files:
        try:
            r = scan(rc, map_data, p, args.max_obs, args.player, args.step, args.fog)
        except Exception as e:
            print("%-24s ERROR %s" % (os.path.basename(p)[:24], repr(e)[:60]))
            continue
        name = os.path.basename(p)
        den = max(r["obs"], 1)
        print("%-24s %5d %6.1f %6.1f %6d %6d %5.1f%% %6.1f%%" % (
            name[:24], r["obs"], r["our_mean"], r["enemy_mean"],
            r["our_max"], r["enemy_max"],
            100.0 * r["suppressed"] / den, 100.0 * r["weak"] / den))
        r["name"] = name
        r["path"] = p
        rows.append(r)

    if args.csv and rows:
        with open(args.csv, "w") as f:
            f.write("name,path,obs,loops,our_mean,enemy_mean,our_max,enemy_max,"
                    "suppressed,weak,both,our_end,enemy_end,supp_pct,weak_pct\n")
            for r in rows:
                den = max(r["obs"], 1)
                f.write("%s,%s,%d,%d,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f\n" % (
                    r["name"], r["path"], r["obs"], r["loops"],
                    r["our_mean"], r["enemy_mean"], r["our_max"], r["enemy_max"],
                    r["suppressed"], r["weak"], r["both"], r["our_end"], r["enemy_end"],
                    100.0 * r["suppressed"] / den, 100.0 * r["weak"] / den))
        print("csv ->", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
