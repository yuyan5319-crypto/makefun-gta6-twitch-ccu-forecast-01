"""Grid around the winning hybrid variant on rolling-midnight pseudo-cases.

Reuses validate.py's leave-one-out harness. The hybrid forecast takes each
hour's median log ratio from the regime pool and a normal spread around
it; this script sweeps the spread shrink and the scale estimator.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model
import validate as val

Z = val.Z


def hybrid2(case, p1, shrink, scale):
    pools = model.pool_log_ratios(case, rolling_only=True)
    rows = []
    for hour in range(2, 25):
        pool = pools.get(hour) or []
        if len(pool) >= 2:
            median = model.empirical_quantile(pool, 0.5)
            if scale == "sd":
                s = math.sqrt(sum((x - median) ** 2 for x in pool) / len(pool))
            elif scale == "mad":
                s = 1.4826 * model.empirical_quantile(
                    sorted(abs(x - median) for x in pool), 0.5)
            else:  # iqr
                s = (model.empirical_quantile(pool, 0.75)
                     - model.empirical_quantile(pool, 0.25)) / 1.349
            qs = [p1 * math.exp(median + shrink * z * s) for z in Z]
        elif pool:
            qs = [p1 * math.exp(pool[0])] * 5
        else:
            qs = [p1] * 5
        rows.append({"hour": hour, "q": qs})
    return rows


def main():
    kit = (Path(__file__).resolve().parent.parent.parent
           / "gta6-twitch-ccu-forecast-01-kit" / "input" / "p00.in")
    case = json.loads(kit.read_text(encoding="utf-8"))
    comparables = case["comparables"]
    curves = {}
    for comp in comparables:
        curve = model.day_curve(comp)
        if model.is_launch_curve(comp, curve):
            curves[comp["name"]] = (comp, curve)
    rolling = {n for n, (c, _) in curves.items()
               if model.rolling_unlock(c.get("unlock"))}

    grid = [(s, k) for s in ("sd", "mad", "iqr")
            for k in (0.9, 1.0, 1.1, 1.2)]
    totals = {g: [] for g in grid}
    for name, (comp, curve) in curves.items():
        if name not in rolling:
            continue
        others = [c for c in comparables if c["name"] != name]
        pset = val.pseudo_case(curve, others, comp.get("unlock"))
        actual = {h: curve[h] for h in range(2, 25) if h in curve}
        for g in grid:
            totals[g].append(val.pinball(hybrid2(pset, curve[1], g[1], g[0]), actual))

    print(f"{'scale':5s} {'shrink':>6s} {'loss':>8s} {'miss%':>7s} {'cov90':>6s}")
    for g in grid:
        rows = totals[g]
        loss = sum(r["loss"] for r in rows) / len(rows)
        miss = sum(r["typical_miss_pct"] for r in rows) / len(rows)
        cov = sum(r["coverage90"] for r in rows) / len(rows)
        print(f"{g[0]:5s} {g[1]:6.2f} {loss:8.4f} {miss:7.1f} {cov:6.3f}")


if __name__ == "__main__":
    main()
