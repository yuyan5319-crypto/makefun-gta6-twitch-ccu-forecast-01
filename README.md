# GTA VI launch-day Twitch viewers, hour by hour — a forecast program

Entry for make.fun's **gta6-twitch-ccu-forecast-01**. `main.py` (Python 3.13,
standard library only, no network) is run once per case in the contest
sandbox: it reads the case file in the input directory — everything recorded
up to the end of hour 1 of release day, hour 1's print included — and writes
five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) for each of hours 2..24 to
the output directory, in the contest's format.

## What is being forecast

The concurrent live viewers of Twitch's "Grand Theft Auto VI" category
(game id 1867279146) in each UTC hour of 2026-11-19. An hour's print is the
mean of that hour's per-minute reads of Twitch's directory count. Scoring is
the mean pinball loss on the log scale over hours 2..24 and the five levels,
so the forecast must be right in *ratio* terms and honestly spread.

## The launch, and why hour 1 is mid-ramp

Researched and cross-checked against the contest's own notes:

- GTA VI releases **19 November 2026** on PS5 and Xbox Series X|S only
  (no PC until 2027), single-player, pre-load open since 10 November.
- It unlocks at **local midnight, country by country**: New Zealand first
  (2026-11-18 11:00 UTC), central Europe 23:00 UTC, the **UK at 00:00 UTC —
  exactly hour 1 of the scored day** — and the US at 05:00 UTC (midnight
  Eastern; store listings put the whole US on Eastern time, 21:00 Pacific
  on the 18th; the contest text's per-zone reading puts the west coast at
  08:00 UTC — either way the US wave lands in hours 6–9).
- So at the checkpoint the Pacific, Asia, Europe and the UK are already
  playing, but the single largest audience wave is still ~4 hours away.
  Hour 1 is a point on the ramp, not the peak — a persistence forecast
  (every hour = hour 1) is structurally wrong for this launch shape.

## Method

Everything is computed from the case file at run time; the only fitted
numbers are three cross-validated constants in `main.py`.

1. **Ratio curves.** Each of the case's `comparables` (38 past launches in
   the practice cases, 42 in the final; hourly SullyGnome averages) is
   reduced to its release-day curve and to log ratios log(v_h / v_1) for
   hours 2..24. Comparables that did not really launch that day (including
   the GTA VI category's own pre-release history, which rides along in
   every case) are filtered out by a peak-vs-baseline lift test.
2. **Regime split.** Launches fall into two regimes: rolling local-midnight
   unlocks, whose audience is still arriving when the UTC day opens, and
   single global unlocks, which are already at full strength in hour 1
   (in the pool: hour-1-to-peak ratios run from ~1.0 for the global ones
   to 3–12× for staggered ones). GTA VI and all four practice launches are
   rolling-midnight, so when the case's notes say so, the pool is
   restricted to comparables whose unlock text describes the same pattern
   (full pool as fallback when a subset runs thin).
3. **Hybrid quantiles.** Per hour, the median log ratio of the pool is the
   centre and a normal spread with the pool's standard deviation (×1.05)
   gives the five quantiles:
   `q_tau(h) = p1 * exp(median_h + 1.05 * z_tau * sd_h)`.

The pooled median curve this produces dips to ~0.87× hour 1 at hour 2
(the post-unlock lull before the US wakes), climbs through the US unlock
and afternoon to a **peak of ~2.5× hour 1 around hour 19 UTC** (European
evening overlapping US afternoon), and ends the day near ~1.9×. For scale:
the biggest launch days in the pool peaked at 0.5–1.1M viewers (Cyberpunk
2077 ~1.07M, Elden Ring 875k in its first hour). If GTA VI's hour 1 prints
in the 300–500k range, this curve implies a peak around 0.8–1.3M — the
model itself is ratio-based and needs no absolute assumption.

## Validation (no local grade exists; the server grades)

`validate.py` runs leave-one-out cross-validation over the comparables:
each launch is hidden in turn, forecast from the rest by the shipped code,
and scored with the contest's exact loss. On the nine rolling-midnight
pseudo-cases — the regime GTA VI is in:

| forecast | loss | typical miss | 90% coverage |
|---|---|---|---|
| house persistence baseline | 0.394 | 100% | 0.758 |
| **main.py as shipped** | **0.337** | 76% | 0.874 |

(The baseline's 0.394 here against its published 0.387 on the real
practice cases is the sanity check that this harness measures the right
thing.) `tune.py` holds the spread/scale grid the constants came from.
Caveat: a prize also requires beating the house's one-shot model forecast,
whose practice loss the kit does not publish.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 validate.py   # leave-one-out cross-validation over the comparables
```

Data provenance: the comparables and practice cases ship in the contest
kit and come from SullyGnome.com's hourly Twitch category averages, as its
`input/NOTICE.txt` describes; the unlock reporting was cross-checked
against Rockstar/PlayStation unlock announcements in November 2026.
