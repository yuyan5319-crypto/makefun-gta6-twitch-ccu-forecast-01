"""Leave-one-out cross-validation of the forecast model over comparables.

The contest's practice answers stay on the server, so the honest local
test is this: take every comparable launch pooled in a practice case, hide
it, forecast its release day from the remaining comparables exactly as a
real case is forecast (anchored on the hidden launch's own hour-1 print),
and score the forecast against its recorded curve with the contest's own
loss -- the mean pinball loss on the log scale over hours 2..24 and the
five levels. The house persistence baseline is scored on the same hidden
launches for comparison.

The launches that matter are the ones that unlocked the way GTA VI does:
rolling at local midnight, country by country (the four practice launches
all did, and so does GTA VI). Launches with one simultaneous global unlock
are a different regime -- their hour 1 is already the peak -- so results
are reported for the rolling-midnight pseudo-cases on their own, and for
all pseudo-cases together.

    python3 validate.py [path/to/a-case.in]
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model

LEVELS = model.LEVELS
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def pinball(rows, actual):
    """Mean pinball loss on the log scale; also typical miss and coverage."""
    total, count = 0.0, 0
    miss, hours = 0.0, 0
    covered = 0
    for row in rows:
        y = actual.get(row["hour"])
        if y is None:
            continue
        y = max(1.0, y)
        hours += 1
        miss += abs(math.log(y) - math.log(row["q"][2]))
        if row["q"][0] <= y <= row["q"][4]:
            covered += 1
        for tau, q in zip(LEVELS, row["q"]):
            d = math.log(y) - math.log(max(q, 1e-9))
            total += 2 * max(tau * d, (tau - 1) * d)
            count += 1
    return {
        "loss": total / count,
        "typical_miss_pct": 100 * (math.exp(miss / hours) - 1),
        "coverage90": covered / hours,
        "hours": hours,
    }


def persistence(p1):
    return [
        {"hour": h, "q": [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]}
        for h in range(2, 25)
    ]


def hybrid_forecast(case, p1, rolling_only, shrink):
    """Empirical median per hour; spread from a normal fit around it."""
    pools = model.pool_log_ratios(case, rolling_only=rolling_only)
    rows = []
    for hour in range(2, 25):
        pool = pools.get(hour) or []
        if len(pool) >= 2:
            median = model.empirical_quantile(pool, 0.5)
            sd = math.sqrt(sum((x - median) ** 2 for x in pool) / len(pool))
            qs = [p1 * math.exp(median + shrink * z * sd) for z in Z]
        elif pool:
            qs = [p1 * math.exp(pool[0])] * 5
        else:
            qs = [p1] * 5
        rows.append({"hour": hour, "q": qs})
    return rows


def empirical_forecast(case, p1, rolling_only, shrink):
    """Ablation: raw empirical quantiles of the pooled log ratios."""
    pools = model.pool_log_ratios(case, rolling_only=rolling_only)
    rows = []
    for hour in range(2, 25):
        pool = pools.get(hour) or []
        if pool:
            median = model.empirical_quantile(pool, 0.5)
            qs = [
                p1 * math.exp(median + shrink * (model.empirical_quantile(pool, tau) - median))
                for tau in LEVELS
            ]
        else:
            qs = [p1] * 5
        rows.append({"hour": hour, "q": qs})
    return rows


def pseudo_case(target_curve, others, notes):
    return {
        "levels": LEVELS,
        "game": {"name": "__pseudo_case__"},
        "known": [{"hour": 1, "viewers": target_curve[1]}],
        "forecast": [{"hour": h} for h in range(2, 25)],
        "notes": notes,
        "comparables": others,
    }


def main():
    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent.parent
        / "gta6-twitch-ccu-forecast-01-kit" / "input" / "p00.in"
    )
    case = json.loads(case_path.read_text(encoding="utf-8"))
    comparables = case["comparables"]

    curves = {}
    for comp in comparables:
        curve = model.day_curve(comp)
        if model.is_launch_curve(comp, curve):
            curves[comp["name"]] = (comp, curve)
    rolling_names = {n for n, (c, _) in curves.items() if model.rolling_unlock(c.get("unlock"))}
    print(f"usable launch curves: {len(curves)} of {len(comparables)} comparables; "
          f"{len(rolling_names)} of them rolling-midnight launches")

    variants = [
        "persistence (house baseline)",
        "main.py as shipped (regime hybrid 1.05)",
        "hybrid full pool, shrink 1.05",
        "empirical regime pool, shrink 1.00",
        "hybrid regime pool, shrink 0.85",
    ]
    scores = {v: {"rolling": [], "all": []} for v in variants}

    def record(variant, name, result):
        scores[variant]["all"].append(result)
        if name in rolling_names:
            scores[variant]["rolling"].append(result)

    for name, (comp, curve) in curves.items():
        others = [c for c in comparables if c["name"] != name]
        p1 = curve[1]
        actual = {h: curve[h] for h in range(2, 25) if h in curve}
        case_tight = pseudo_case(curve, others, comp.get("unlock"))
        case_full = pseudo_case(curve, others, "")

        record(variants[0], name, pinball(persistence(p1), actual))
        record(variants[1], name, pinball(
            model.forecast_case(case_tight, "loo")["hours"], actual))
        record(variants[2], name, pinball(hybrid_forecast(case_full, p1, False, 1.05), actual))
        record(variants[3], name, pinball(empirical_forecast(case_tight, p1, True, 1.00), actual))
        record(variants[4], name, pinball(hybrid_forecast(case_tight, p1, True, 0.85), actual))

    for group in ("rolling", "all"):
        print(f"\n--- pseudo-cases: {group} ---")
        print(f"{'variant':38s} {'loss':>8s} {'miss%':>7s} {'cov90':>6s} {'n':>3s}")
        for v in variants:
            rows = scores[v][group]
            if not rows:
                continue
            loss = sum(r["loss"] for r in rows) / len(rows)
            miss = sum(r["typical_miss_pct"] for r in rows) / len(rows)
            cov = sum(r["coverage90"] for r in rows) / len(rows)
            print(f"{v:38s} {loss:8.4f} {miss:7.1f} {cov:6.3f} {len(rows):3d}")


if __name__ == "__main__":
    main()
