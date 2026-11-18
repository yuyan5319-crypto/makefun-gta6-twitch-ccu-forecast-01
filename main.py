"""Forecast a game launch day's hourly Twitch category viewers, hours 2..24.

Written for make.fun's gta6-twitch-ccu-forecast-01. The program is run once
per case in a sandbox with no network. A case file carries everything the
forecast is allowed to know at the end of hour 1 of release day:

  * `known`       -- hour 1's print (the anchor every forecast hangs from),
  * `history`     -- this category's own points before the checkpoint,
  * `comparables` -- past big launches' hourly curves (SullyGnome averages),
                     each with its release-day start and a text account of
                     how that launch unlocked.

Method (an empirical ratio pool, tuned by leave-one-out cross-validation
over the comparables themselves -- see validate.py and README.md):

  1. Every comparable is reduced to its release-day curve: the mean viewers
     of each UTC hour 1..24 of its release day, and the log ratios
     log(v_h / v_1) of hours 2..24 against its own hour 1.
  2. Comparables are split by how they unlocked. A launch that rolled out
     at local midnight country by country (GTA VI's pattern, and the four
     practice launches') has most of its audience still to arrive when the
     UTC day opens; a single global unlock is already at full strength in
     hour 1. When the case's own notes describe a rolling midnight launch,
     the pool is restricted to comparables whose unlock text does too,
     with the full pool as fallback when the subset runs thin.
  3. For each hour, the forecast is a hybrid of the pool's centre and a
     normal spread around it, mapped back through the case's own hour-1
     print:
         q_tau(h) = p1 * exp(median_h + SHRINK * z_tau * sd_h),
     where median_h and sd_h are the median and the standard deviation of
     the pooled log ratios at hour h. Leave-one-out cross-validation over
     the comparables (validate.py) put this at a loss of about 0.337 on
     rolling-midnight pseudo-cases, against the house persistence
     baseline's 0.394 on the same launches (0.387 on the real practice).

Only the Python 3.13 standard library is used. Runtime is a few
milliseconds per case, far inside the 8 second cap.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing else is
# imported from it today, but keep the line so a future model.py would load.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in row order (read back from the case
# and checked against this list).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- Model constants, chosen by leave-one-out cross-validation over the ---
# --- comparables pooled in the practice cases (validate.py).            ---
# A comparable needs an hour-1 print and at least this many of hours 2..24
# before its curve is trusted.
MIN_DAY_HOURS = 16
# A restricted (same-unlock-pattern) pool smaller than this falls back to
# the full pool for that hour.
MIN_POOL = 6
# The pooled log-ratio spread is multiplied by this factor before the
# normal quantiles are taken. Cross-validation found a flat optimum
# between 1.00 and 1.10 on rolling-midnight launches; 1.05 sits in the
# middle of it and brings the 90% coverage close to its honest 0.90.
SPREAD_SHRINK = 1.05
# Spread assumed when an hour's pool holds a single comparable.
SINGLE_POOL_SD = 0.4
# A comparable whose release-day peak never reaches this multiple of its
# pre-release baseline is not a launch curve at all (the GTA VI category's
# own pre-release history rides along in every case's comparables); it is
# left out of the pool.
MIN_LAUNCH_LIFT = 3.0


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def day_curve(comparable):
    """A comparable's release-day curve: {hour 1..24: mean viewers}.

    Points are hourly averages whose `from` opens the hour they cover, so
    each point lands in exactly one release-day hour; an hour with several
    points (a denser source) is averaged, as the contest averages its own
    minute reads into an hour's print.
    """
    start = parse_ts(comparable["day_start"])
    buckets = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        hour = int((parse_ts(point["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            buckets.setdefault(hour, []).append(float(viewers))
    return {h: sum(v) / len(v) for h, v in buckets.items()}


def pre_release_baseline(comparable):
    """Median hourly viewers over the 3 days before the release day."""
    start = parse_ts(comparable["day_start"])
    values = []
    for point in comparable.get("hourly") or []:
        when = parse_ts(point["from"])
        if start - timedelta(days=3) <= when < start:
            if point.get("viewers") is not None:
                values.append(float(point["viewers"]))
    if not values:
        return 0.0
    values.sort()
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def is_launch_curve(comparable, curve):
    """True when the comparable really launched on its day_start day."""
    if len(curve) < MIN_DAY_HOURS or curve.get(1, 0) <= 0:
        return False
    baseline = pre_release_baseline(comparable)
    if baseline <= 0:
        return True
    return max(curve.values()) >= MIN_LAUNCH_LIFT * baseline


def rolling_unlock(text):
    """True when an unlock/notes text describes a rolling local-midnight
    launch (country by country), as opposed to one simultaneous global
    unlock time."""
    low = (text or "").lower()
    if "midnight" not in low:
        return False
    if "single global unlock" in low or "one global unlock" in low:
        return False
    return True


def empirical_quantile(sorted_values, level):
    """The level quantile of sorted values, linearly interpolated."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = level * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def norm_quantile(level):
    """The standard normal's quantile at `level` (Acklam's approximation,
    accurate to about 1e-9 in the central range the contest uses)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if level < plow:
        q = math.sqrt(-2 * math.log(level))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if level > phigh:
        q = math.sqrt(-2 * math.log(1 - level))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = level - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def pool_log_ratios(case, rolling_only):
    """{hour: sorted [log(v_h / v_1)]} pooled over the case's comparables.

    The case's own game is skipped (in the final, the GTA VI category's
    pre-release history is itself one of the comparables), and so is any
    comparable whose release day was not a real launch curve.
    """
    pools = {h: [] for h in range(2, 25)}
    own_name = (case.get("game") or {}).get("name", "")
    for comparable in case.get("comparables") or []:
        if comparable.get("name") == own_name:
            continue
        if rolling_only and not rolling_unlock(comparable.get("unlock")):
            continue
        curve = day_curve(comparable)
        if not is_launch_curve(comparable, curve):
            continue
        first = curve[1]
        for hour in range(2, 25):
            if hour in curve and curve[hour] > 0:
                pools[hour].append(math.log(curve[hour] / first))
    for values in pools.values():
        values.sort()
    return pools


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # GTA VI and the practice launches all rolled out at local midnight;
    # when the case's notes say so, pool the comparables that did the same.
    case_rolling = rolling_unlock(case.get("notes"))
    full_pool = pool_log_ratios(case, rolling_only=False)
    tight_pool = pool_log_ratios(case, rolling_only=True) if case_rolling else None

    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        pool = full_pool.get(hour) or []
        if tight_pool is not None and len(tight_pool.get(hour) or []) >= MIN_POOL:
            pool = tight_pool[hour]
        if pool:
            median = empirical_quantile(pool, 0.5)
            if len(pool) >= 2:
                sd = math.sqrt(sum((x - median) ** 2 for x in pool) / len(pool))
            else:
                sd = SINGLE_POOL_SD
            quantiles = [
                p1 * math.exp(median + SPREAD_SHRINK * norm_quantile(level) * sd)
                for level in LEVELS
            ]
        else:
            # No comparable covers this hour: persist hour 1 with the house
            # baseline's widening spread rather than emit nothing.
            z = [-1.645, -0.674, 0.0, 0.674, 1.645]
            quantiles = [p1 * math.exp(zz * 0.15 * math.sqrt(hour - 1)) for zz in z]
        quantiles = [round2(min(max(q, 0.01), 1e9)) for q in quantiles]
        quantiles.sort()
        rows.append({"hour": hour, "q": quantiles})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
