#!/usr/bin/env python3
"""Check every contact in the cached plans: how real does each bounce look?

    python3 tools/audit.py

For each bounce prints the normal restitution e_n = |v_out_n| / |v_in_n|, the
total speed ratio |v_out| / |v_in| and the change of the sliding speed, plus
the worst ball/ring overlap of the whole round. Run render.py once first so
plans.pkl exists.
"""

import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import physics as ph  # noqa: E402
import render  # noqa: E402
import scenario as sc  # noqa: E402


def main():
    with open(os.path.join(HERE, "plans.pkl"), "rb") as f:
        cache = pickle.load(f)
    g = np.array([0.0, 1.0])
    worst_all = []
    for cfg in sc.ROUNDS:
        plan = cache.get(render.plan_key(cfg))
        if plan is None:
            print(f"{cfg['key']}: no cached plan, run render.py first")
            continue
        geo = ph.geo_from(cfg, sc.CENTER)
        gvec = g * geo.g
        segs = sorted(plan.segs, key=lambda s: s.t0)
        lim = cfg.get("limits") or {}
        print(f"\n{cfg['key']}  (restitution {cfg['restitution'][0]}-{cfg['restitution'][1]}, "
              f"max speed gain x{lim.get('SPEED_GAIN', ph.SPEED_GAIN)} + ring friction)")
        print("   time   ring  |v_in|  |v_out|  speed x   e_n   d(slide)")
        rows = []
        for a, b in zip(segs, segs[1:]):
            if not b.contact or b.t0 >= cfg["t_up"]:
                continue
            tau = b.t0 - a.t0
            v_in = a.v0 + gvec * tau
            n = (b.p0 - geo.C) / np.linalg.norm(b.p0 - geo.C)
            t_hat = np.array([-n[1], n[0]])
            vin_n, vin_t = v_in @ n, v_in @ t_hat
            vo_n, vo_t = b.v0 @ n, b.v0 @ t_hat
            sp_in, sp_out = np.linalg.norm(v_in), np.linalg.norm(b.v0)
            ring = int(round((np.linalg.norm(b.p0 - geo.C) - geo.reff(0)) / geo.pitch))
            e_n = abs(vo_n) / max(abs(vin_n), 1e-9)
            rows.append((b.t0, ring, sp_in, sp_out, sp_out / max(sp_in, 1e-9), e_n, vo_t - vin_t))
        gain = lim.get("SPEED_GAIN", ph.SPEED_GAIN)
        keep = lim.get("SPEED_KEEP", ph.SPEED_KEEP)
        for r in rows:
            flag = ""
            if r[4] > gain + 1e-6 and r[3] > 220:
                flag = "  <- faster than the limit (spinning ring friction)"
            elif r[2] > 350 and r[4] < keep:
                flag = "  <- dead stop"
            print(f"  {r[0]:6.3f}  {r[1]:4d}  {r[2]:6.0f}  {r[3]:7.0f}  {r[4]:6.2f}  {r[5]:5.2f}  {r[6]:+7.0f}{flag}")
        t_end = cfg["t_up"] if cfg["outcome"] == "lose" else (plan.t_free or cfg["t_up"])
        worst, where = ph.verify(plan, cfg, sc.CENTER, cfg["t_show"], t_end)
        worst_all.append(worst)
        if rows:
            ratios = [r[4] for r in rows]
            print(f"  speed ratio range {min(ratios):.2f}-{max(ratios):.2f}, "
                  f"max e_n {max(r[5] for r in rows):.2f}, ring overlap {worst:.3f} px")
    if worst_all:
        print(f"\nworst overlap over all rounds: {max(worst_all):.3f} px")
    return 0 if all(math.isfinite(w) and w <= 0.5 for w in worst_all) else 1


if __name__ == "__main__":
    sys.exit(main())
