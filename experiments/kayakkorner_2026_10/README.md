# Elo V1 tuning for KayakKorner (October 2026)

KayakKorner is replacing its rating system with the Elo V1 idea from this
repository (KayakKorner issue #52). KayakKorner reimplements the idea itself,
incrementally; this repository is used to search for the hyperparameters and,
later, to check that KayakKorner's engine computes the same thing.

This folder records the experiments run on 9 October 2026, their results and
the decisions taken.

## Data

- Production database of KayakKorner, 9 October 2026 (`data/raw/` was
  refreshed from it in commit `0fae91a`).
- `data/processed/dataset_kk.json`: exported by KayakKorner
  (`scripts/export_dataset_elo.py`). Same format as `dataset.json`, but the
  prueba metadata come from **KayakKorner's own classifiers**, not from
  `util/build_dataset.py`. Not tracked (it is in `data/processed/`).
- Same scope as `dataset.json`: 32,810 pruebas and 324,080 results (aguas
  tranquilas and mar; no paleo/SUP; boats made only of placeholder athletes
  removed).

### Causal metadata instead of global

`build_dataset.py` fills missing metadata with athlete profiles built from
**all** of each athlete's history, including races after the prueba, and
retrains the length classifier on everything. KayakKorner classifies each
prueba once, when it is imported, using only what is known at that moment, and
never touches it again. The athlete profile is an accumulator (counters) that
grows with every new result.

Filling accuracy on pruebas whose name does state the field, predicting with
the profiles *before* each regatta (no look-ahead):

| Field | KayakKorner (causal) | build_dataset (in-sample) |
|---|---:|---:|
| sexo | 99.6 % | 99.7 % |
| embarcacion_tipo | 98.1 % | 98.7 % |
| categoria | 95.0 % | 95.1 % |
| length_class (Extra Trees, CV grouped by regatta) | 94.5 % | 94.6 % |

Fields read from the name (tipo, embarcacion_num, distancia_exacta) are
identical to `dataset.json` in all 32,810 shared pruebas.

Category differences in KayakKorner (they matter for the birth-year estimate
and for predicting the category of a prueba):

- **Federation age tables**: age reached during the season (cadete 15–16,
  junior 17–18, sub23 19–23, veterano ≥35), not `CATEGORY_AGE_RANGE`, which is
  one year younger.
- A prueba admits its own category and the one below (cadete pruebas include
  infantiles). Senior admits from junior age (17) onwards: juniors do race
  senior pruebas. Senior and veterano give no upper age, so they do not bound
  the birth year.
- The category of a prueba is predicted as the **narrowest prueba category
  that admits at least 90 % of its athletes** (overlap of each athlete's age
  range, weighted by confidence). Predicting from the mean age, as
  `predict_prueba_categoria` does, gave 70 % with the federation tables,
  because the mean of "own category or the one below" falls half a category
  low and open veteran ranges pull senior pruebas up. With the overlap rule:
  92 % with senior from 19, 95 % with senior from 17.

### Elo V1 defaults on causal metadata

| Dataset | Ungrouped acc / Brier / log loss | Grouped acc / Brier / log loss |
|---|---|---|
| `dataset.json` | 72.93 % / 0.1909 / 0.5661 | 74.68 % / 0.1779 / 0.5348 |
| `dataset_kk.json` | 72.96 % / 0.1908 / 0.5659 | 74.73 % / 0.1777 / 0.5342 |

The causal metadata do not hurt the engine.

## Experiment 1: broad search (`search_broad.py`)

600 random configurations of `HYPERPARAM_GRID`, extended so that the current
defaults are inside the grid (K 20–300, field-size multipliers 0.7/1.75,
half-life 730 days, default length transfer). Results:
`results/search_broad.csv`.

The previous `elo_v1_output/hyperparam_search.csv` was not usable: the grid
did not contain the defaults (K=30), it predates the evaluation change of
3 September, and its best configuration (K=300) gave a very dispersed
population (std ≈ 383).

**Finding 1: grouping is a choice of purpose, not a detail.**
Ranked by ungrouped log loss (the ranking in `hyperparam_search`), the top 20
configurations all train with `group_pruebas=False` and K 100–300:
ungrouped log loss 0.438 (accuracy 78.3 %), but grouped log loss 0.568
(72.3 %). Training on merged groups compares women with men and cadets with
juniors in the same race, which hurts prediction inside each prueba but is
what makes ratings comparable across categories.

**Decision: KayakKorner wants a universal rating** (comparable across sexes
and categories), so `group_pruebas=True` and configurations are ranked by
**grouped** metrics. With high K, grouping also stays close to the best
ungrouped log loss (0.47).

**Finding 2: precision against volatility.** Better prediction comes with
bigger rating jumps. Pareto front (grouped, low transfer):

| | Mean jump per update | p95 jump | Final std | Grouped acc | Grouped log loss | Ungrouped log loss |
|---|---:|---:|---:|---:|---:|---:|
| Defaults (K=30) | 10.5 | 29 | 124 | 74.7 % | 0.534 | 0.566 |
| A (K=75) | 15 | 39 | 168 | 76.0 % | 0.502 | 0.537 |
| B (K=100) | 21 | 63 | 209 | 77.2 % | 0.472 | 0.510 |
| C (K=150) | 31 | 101 | 246 | 77.9 % | 0.453 | 0.495 |
| D (K=200) | 55 | 179 | 348 | 78.9 % | 0.435 | 0.471 |

High K predicts better mostly because many athletes are juniors who improve
fast. The jump is what users see after every regatta; std is mostly scale.

**Decision: option A** (about 15 points per update).

## Experiment 2: length transfer (`transfer_matrices.py`)

KayakKorner shows the four distance ratings in a radar chart and wants
athletes to look different across axes, so transfer must be small. Option A
fixed, only the matrix changes. "Axis gap" = max − min of an athlete's
distance ratings, over the 7,178 athletes with at least 3 direct races in two
or more distances. Results: `results/transfer_matrices.json`.

| Matrix | Neighbours | Grouped acc | Grouped log loss | Axis gap mean / median | ss↔sprint gap | fondo↔maratón gap | Corr fondo–maratón |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 (none) | 0 | 74.3 % | 0.514 | 126 / 100 | 105 | 96 | 0.81 |
| 7 (current default) | 0.20 | 75.5 % | 0.509 | 117 / 92 | 91 | 84 | 0.87 |
| V25 | 0.25 | 75.6 % | 0.508 | 115 / 91 | 88 | 80 | 0.88 |
| **2** | 0.35 | 75.7 % | 0.506 | 111 / 87 | 82 | 75 | 0.90 |
| V50 | 0.50 | 75.3 % | 0.505 | 108 / 83 | 74 | 66 | 0.93 |
| 3 | 0.50 (+0.25 sprint↔fondo) | 76.0 % | 0.502 | 103 / 79 | 73 | 69 | 0.93 |
| 4 | 0.75 | 76.2 % | 0.496 | 91 / 67 | 64 | 59 | 0.95 |

The volatility does not depend on the matrix (≈15 per update). The big gain
is going from no transfer to some (+1.2 points of accuracy); each further step
adds 0.1–0.3 points and removes 5–10 % of the axis gap.

**Decision: matrix 2** (`HYPERPARAM_GRID["length_transfer"][2]`): keeps 88 %
of the axis gap of no transfer and most of the prediction gain.

```
              super  sprint  fondo  maratón
super_sprint  1      0.35    0      0
sprint        0.35   1       0.10   0
fondo         0      0.10    1      0.35
maratón       0      0       0.35   1
```

## Experiment 3: fine search (`search_fine.py`)

300 random configurations around A with `group_pruebas=True`,
`context_mode="modifiers"` (needed for the athlete's predominant context) and
matrix 2. Results: `results/search_fine.csv`.

- The landscape is flat: within reasonable ranges, K (50–100), the lambdas,
  inactivity and field-size scaling move the grouped log loss in the third
  decimal.
- `team_update_mode="inverse_sqrt"` is clearly better (25 of the top 30).
- At the same volatility the fine search barely beats A with matrix 2
  (0.501 against 0.506 grouped log loss at ≈15 per update).

## Experiment 4: stabilisation and starting rating (`stabilisation/`)

After the first rebuild, ratings did not look settled: established athletes
kept moving in the same direction for years and the population kept spreading
apart. Request: an athlete should be roughly adjusted within one year of races.
Proposed knobs: a higher K or a different lambda.

These runs use **KayakKorner's own engine** (`prediction/elo` in that
repository, validated against `run_rating_system` to 2e-13), with matrix 2 and
the chosen configuration as baseline (K=60, `uncertainty_k_max_multiplier`
1.5, `primary_lambda` 20).

New measures:

- **Annual drift**: rating at the end of each season in each distance (seasons
  with at least 3 races in it), change between consecutive seasons, by age
  (birth year from KayakKorner): junior < 19, adult 19–34, veteran ≥ 35.
  Veterans do not really improve year after year, so their drift should be
  about zero.
- **Scale**: median end-of-season fondo rating of established athletes.
- **Comparability**: in merged groups, pairs of different sex (oriented to the
  woman: predicted probability that she wins against how often she does) and of
  different category (oriented to the younger one).
- **Axis gap**: median of best − worst distance rating of athletes with at
  least 3 direct races in two or more distances, and correlation between
  neighbouring distances.
- **Newcomers**: log loss of pairs involving an athlete in their first year
  against pairs between established athletes.

### Finding 1: the drift is inflation, and a higher K makes it worse

`results/k_kmax_lambda.json` (27 configurations, K 40/60/90 ×
uncertainty multiplier 1.5/3/5 × lambda 5/20/40). Mean signed drift is
**positive for every configuration and age group**, veterans included
(+11 to +19 per season after their second season), and grows with K:

| Configuration | Veterans, 3rd season on | Adults, 3rd season on | Grouped accuracy |
|---|---:|---:|---:|
| K=40, ×1.5, λ=5 | +11 | +19 | 74.8 % |
| K=60, ×1.5, λ=20 (chosen) | +15 | +25 | 76.0 % |
| K=90, ×5, λ=40 | +19 | +36 | 78.8 % |

Every newcomer starts at 1500, most of them are juniors weaker than that, and
the established athletes who beat them collect the points; when juniors leave,
their losses stay in the system. More K redistributes more points.

### Finding 2: start newcomers near their demonstrated level

**Performance rating**: the rating with which the athlete's result in their
first group would have been exactly the expected one against their rivals'
ratings (bisection between 1100 and 2100; newcomers in the same group are
estimated jointly; in a crew with known partners the newcomer gets what makes
the crew mean equal the performance; with no known boat in the group there is
no reference and the athlete stays at 1500). The first race is still predicted
with 1500, before the start rating is set, so there is no look-ahead.

Strategies (`results/start_strategies.json`, K=60, ×1.5, λ=20):

| Start | Grouped log loss | Grouped acc | Mean jump | Veteran drift | Adult drift | Established fondo median 2016 → 2026 |
|---|---:|---:|---:|---:|---:|---|
| 1500 (current) | 0.501 | 76.0 % | 14.4 | +15 | +25 | 1505 → 1528 |
| 1500 + 0.3·(performance − 1500) | **0.488** | 76.1 % | 13.1 | +10 | +21 | 1504 → 1494 |
| 1500 + 0.5·(performance − 1500) | 0.490 | 75.7 % | 12.6 | +7 | +18 | 1503 → 1464 |
| performance only | 0.547 | 74.0 % | 12.5 | −1 | +10 | 1509 → 1358 |
| mean of known participants of their prueba | 0.498 | 76.1 % | 13.8 | +14 | +22 | 1528 → 1531 |
| mean of their category and sex (active last year) | 0.494 | 76.3 % | 13.9 | +14 | +24 | 1515 → 1542 |
| category-sex mean + 0.3·(performance − mean) | 0.489 | 76.4 % | 12.6 | +6 | +16 | 1513 → 1441 |

- One race is too noisy to trust fully (performance only is the worst).
- Means alone barely touch the drift.
- Part of the drop in veteran drift is the whole scale moving down: relative
  to the population median, veterans rise about 11–13 per season with every
  strategy. The rest of the "spreading" comes from the database starting in
  2015 with everybody at 1500, elite included: with 10–20 points per regatta,
  strong athletes take seasons to get there.
- **1500 + 0.3·(performance − 1500)** predicts best and keeps the scale stable.

### Finding 3: learn fast while uncertain, then settle

With that start, a lower K and a higher uncertainty multiplier move newcomers
fast and established athletes little (`results/start_with_k_grid.json`,
`results/candidates_comparability.json`):

| Configuration (start 0.3) | Grouped log loss | Grouped acc | Mean jump | Jump, established (n ≥ 20) | Axis gap | Corr. sprint–fondo |
|---|---:|---:|---:|---:|---:|---:|
| Chosen before, start 1500 | 0.501 | 76.0 % | 14.4 | 9.7 | 88 | 0.73 |
| K=60, ×1.5, λ=20 | 0.488 | 76.1 % | 13.1 | 9.4 | 79 | 0.81 |
| K=40, ×3, λ=10 | 0.486 | 76.3 % | 14.0 | 7.9 | 84 | 0.81 |
| **K=40, ×3, λ=20** | **0.479** | **76.6 %** | 15.3 | 9.3 | **88** | 0.81 |
| K=40, ×5, λ=5 | 0.476 | 76.8 % | 18.0 | 8.1 | 99 | 0.81 |
| K=60, ×3, λ=5 | 0.474 | 76.9 % | 18.0 | 9.7 | 100 | 0.80 |

Comparability (same runs):

| Configuration | P(woman beats man): predicted → real | P(younger category wins): predicted → real | Log loss, different sex | Newcomer pairs vs established |
|---|---|---|---:|---|
| Chosen before | 38.3 % → 30.5 % (+7.8) | 45.7 % → 37.9 % (+7.8) | 0.468 | 0.573 vs 0.449 |
| **K=40, ×3, λ=20, start 0.3** | 36.8 % → 30.5 % (+6.4) | 44.2 % → 37.9 % (+6.3) | 0.448 | 0.557 vs 0.421 |

- Cross-sex and cross-category pairs are predicted almost as well as the
  rest; every configuration slightly overrates the weaker group against the
  stronger one, a bit less with the new one. A sex/category correction would
  be a separate improvement.
- Higher uncertainty multipliers bring back the axis gap the start reduces;
  correlations between distances do not change.
- Pairs with a first-year athlete stay much worse predicted than established
  pairs in every configuration: two or three races are not enough to know
  anyone.

**Decision**: K=40, uncertainty multiplier 3, lambda 20 and start
1500 + 0.3·(performance − 1500). It is the best predictor within the chosen
volatility (about 15 points per regatta), established athletes move less than
before, the scale stays put, and the axis gap and comparability are kept or
improved.

### Afterwards: K=60 and a form indicator

Looking at a concrete athlete whose 5000 m results clearly improved in winter
2026 (16th of 70 at the national winter championship, 96.5 % of the median
time, after 44th of 58 the year before), the rating moved +98 over the season
with K=40, while the previous Glicko system had jumped +260 (that system also
swung ±300 within a month at the start of each season).

Raising K does not hurt prediction; in every run it improved it (table above:
K=60, ×3 has grouped log loss 0.460 against 0.479). What it costs is volatility
(about 22 points per regatta instead of 15) and a little more drift. **K was
raised to 60** (×3, λ=20, start 0.3): grouped accuracy 77.6 %, log loss 0.460;
that athlete's 2026 season became +128.

KayakKorner also shows a **form** indicator that does not touch the rating:
per distance, how many points above or below their rating an athlete has
competed in the last 6 months (at least 3 regattas), i.e. the shift d with
which the expected scores of those races (with the ratings of the time) would
match what happened: Σ rivals·E(d) = Σ rivals·actual.

## Experiment 5: a rating in units of time (`time_rating/`)

Question: the Elo scale keeps widening and its points do not mean the same
from one year to the next. Can a rating be compared year to year?

Calibration of the current Elo first (per prueba pairs, fitting a scale s in
P(s·d) each year): favourites win more often than predicted (s > 1), and s
grows from about 1.2 (2016–2019) to 1.5–1.7 (2023–2026) for established
athletes; it is about 1.0 in sprint and 1.3–2.0 in the other distances. The
same difference in points means more in later years and in some distances.

The Elo only uses positions. **Time model**: a Kalman filter with one level
θ per athlete and distance class,

    log(time of boat b in prueba j) = μ_j − mean(θ of the crew) + noise(σ_class)

θ is log-speed (+0.01 ≈ 1 % faster). μ_j, the difficulty of the prueba,
absorbs the real distance (announced distances are often wrong, especially
in maratón), conditions, course and boat type, so only times relative to the
same prueba matter. μ_j is a robust weighted mean of what each boat implies
given the levels before the race; with `tau` the pruebas of the same
performance group (same day, distance, boat and phase) share a difficulty
with a deviation of `tau`, which links categories that paddle the same
distance the same day. Levels gain variance with time without racing (more
for children and juniors), residuals are clipped (Huber 2.5) for capsizes,
packs and bad timing.

Both systems are evaluated on the same 2.8 M within-prueba pairs, before
updating. Residuals have long tails (p99 at 6–8 robust SD, worst in sprint),
so raw normal probabilities are overconfident; each system gets one scale per
distance class fitted on 2015–2019 and is evaluated on 2020–2026.

| Variant | Acc 2020–26 | Log loss 2020–26 | Drift per season: children / junior-sub23 / senior / veteran | SD fondo 2017 → 2026 |
|---|---:|---:|---|---|
| Elo (current) | 76.7 % | 0.477 | +32 / +9 / +27 / +4.3 points | 226 → 288 |
| Time, each prueba alone | 77.8 % | 0.465 | +0.31 / +0.14 / +0.17 / +0.02 % | 8.4 → 8.8 % |
| Time, group τ=1 %, q by category | 78.5 % | 0.464 | +0.93 / +0.21 / +0.23 / +0.02 % | 10.8 → 10.9 % |
| **Time, group τ=2 %, q by category** | **78.9 %** | **0.457** | +0.50 / +0.02 / +0.18 / +0.02 % | 10.1 → 10.5 % |

(Drift: change between consecutive seasons with ≥ 3 races in the distance,
from the 3rd season on, by the category raced most that year; veteran =
veteran both seasons. KayakKorner birth years are inferred from categories
and cannot separate adults from veterans.)

- The time model (τ=2 %) predicts better in every distance: super sprint
  81.1 % vs 76.7 %, fondo 80.7 % vs 78.2 %, sprint 74.2 % vs 73.3 %, maratón
  78.7 % vs 78.3 % (maratón gains least: mass starts and packs).
- **Its scale does not widen**: the spread of established fondo athletes
  stays at about 10 % while the Elo goes from 226 to 288. Veterans move
  +0.02 % per season (0.2 % in ten years, 0.02 SD) against +4.3 points (43 in
  ten years, 0.17 SD) in the Elo. A level in % is a physical unit.
- **Children's improvement is not measured in absolute terms**: about +0.5–1 %
  per season when real improvement is several %. Children race their own
  category at their own distances; when the whole category improves, the
  model cannot tell it from an easier prueba. Their level is relative to their
  linked population (closer to a percentile). The same happens to the Elo.
  Measured with the anchor tests and fixed below.
- The time model's yearly s goes from 1.0 (2015–2022) to 1.3 (2024–2026): its
  noise σ is fixed per class and later years look less noisy (more and better
  timed regattas?). This is about the probability, not the meaning of θ; an
  adaptive σ is the next step.

### Anchor tests (`time_rating/anclas.py`)

The real distance does not matter if the course is the same. For every
prueba, the model's difficulty μ estimated only with established boats (≥ 5
races, level SD ≤ 3 %) and without sharing anything with the group.
Venues are `Eventos.lugar` (`time_rating/lugares.json`, normalised).

**1. Same course, different years**: μ = course (venue × announced distance ×
boat × category × sex, in at least 4 years) + year + noise, robust fixed
effects, 90 % interval by bootstrap of venues. Without rivers, from 2019
(2015–2018 still carry the cold start):

| Athletes (τ=2 % model) | Pruebas | Venues | Year slope | 90 % interval |
|---|---:|---:|---:|---|
| Junior and older | 1566 | 21 | −0.05 %/year | −0.42 … +0.13 |
| Veterans | 448 | 16 | −0.09 %/year | −0.38 … +0.74 |
| Children (alevín–cadete) | 1614 | 32 | −1.8 %/year | −2.9 … −1.1 |

**2. Same day, venue, distance and boat, different categories**: μ of the
prueba minus μ of the seniors' (should be 0). Per year:

| Category | 2016 | 2019 | 2023 | 2026 |
|---|---:|---:|---:|---:|
| Cadete | +7.9 % | +2.6 % | +2.4 % | −2.1 % |
| Junior | +2.4 % | +1.4 % | +1.1 % | −0.8 % |
| Veteran | +0.2 % | +0.9 % | +0.8 % | +0.8 % |
| Senior women | (+2.0 % in 2017) | +1.5 % | +1.2 % | +0.6 % |

Sharing the difficulty within the group matters: with each prueba alone
senior women come out +6.9 % and cadetes +4 %.

- **From junior up the scale is stable**: no drift on the same courses
  (at most ~0.4 %/year within the interval), and veterans and women keep a
  constant offset of about 1 % against senior men. A level in 2019 and in 2026
  can be compared.
- **Children are not comparable over the years**: on the same courses the
  same category goes about 1.8 % per year faster than their levels say, and
  against seniors the same day cadetes move from +8 % to −2 %. Their levels
  lag behind their real improvement (whole categories improve together and
  the filter assumes no change on average). Prediction cannot see it (it is
  common to all rivals in a prueba); it needs an expected improvement per
  category, fitted with these links and validated on held-out venues.

### Children: expected improvement and starting level (`time_rating/calibrar.py`)

Two parameters: an expected improvement per year of mejora × steps to senior
/ 7 (steps: prebenjamín 7 … cadete 3, junior 2, sub23 1, senior and veteran
0), added to the level between races, and newcomers starting at −arranque ×
steps instead of 0. Chosen on half of the venues (crc32 of the name) to bring
the children's same-course slope and the same-day gaps against seniors
(infantil, cadete, junior) to zero; improvement fixes the slope, the start
fixes the gaps, and they must go together. Best: **mejora 11 %, arranque 7 %**
(infantil +6.3 %, cadete +4.7 %, junior +3.1 %, sub23 +1.6 % per year;
`grupo_tau2_qcat_mejora`). The linear shape cannot fit everything (infantiles
stay a little high, cadetes a little low).

Explored points on the fitting half (slope in %/year from 2019, gaps against
seniors in %; the first five rows come from an interrupted full grid whose
results were only printed, the rest are in `results/calibracion.jsonl`):

| mejora | arranque | Children slope | Adults slope | Infantil | Cadete | Junior |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | −1.41 | −0.09 | +3.9 | +1.3 | +0.4 |
| 0 | 0.02 | −1.47 | −0.32 | +0.1 | −1.9 | −1.2 |
| 0.03 | 0.03 | −1.07 | −0.18 | +1.0 | −1.0 | −0.5 |
| 0.06 | 0 | −0.61 | +0.45 | +9.4 | +6.2 | +3.9 |
| 0.06 | 0.04 | −0.73 | −0.03 | +2.1 | −0.2 | +0.3 |
| 0.09 | 0.05 | −0.40 | +0.13 | +3.7 | +0.4 | +1.3 |
| 0.10 | 0.065 | −0.24 | +0.03 | +1.5 | −1.1 | +0.7 |
| **0.11** | **0.07** | **−0.02** | **+0.07** | **+1.6** | **−1.1** | **+0.9** |
| 0.11 | 0.08 | −0.05 | −0.06 | −0.5 | −2.6 | +0.0 |
| 0.12 | 0.075 | +0.14 | +0.11 | +1.7 | −1.0 | +1.0 |
| 0.13 | 0.065 | +0.28 | +0.33 | +4.1 | +1.6 | +2.5 |

| | Before | After |
|---|---|---|
| Children, same course, all venues (from 2019) | −1.84 %/year (−2.9 … −1.1) | −0.54 %/year (−1.4 … +0.1) |
| Junior and older, same course | −0.05 %/year | +0.09 %/year |
| Cadete vs seniors same day, by year | +7.9 (2016) … −2.1 (2026) | −0.8 … +0.2, −1.3 (2026) |
| Held-out venues: cadete / infantil / junior gap | +2.9 / +6.3 / +0.2 % | −0.7 / +4.0 / +0.9 % |
| Held-out venues: children's slope | −2.3 %/year (−3.4 … +2.2) | −1.0 %/year (−2.2 … +3.4) |
| Accuracy / log loss 2020–26 | 78.9 % / 0.457 | **79.9 % / 0.446** |
| Improvement per season, children / junior-sub23 / veteran | +0.5 / +0.02 / +0.02 % | +4.6 / +2.1 / −0.06 % |

Prediction improves (merged groups mix categories) and the measured
improvement per season becomes realistic. The held-out half is small
(17 venues for children, 6 for veterans) and its intervals are wide.

### Scale for display

`graficas.py` shows the time rating as points with a fixed meaning: 1500 is
the median of the athletes active in 2025 (≥ 3 races in the distance, every
category and sex) in each distance, and 25 points are 1 % of speed (the ratio
of the Elo spread to the time spread, so the numbers feel like the Elo). The
base year never changes.

### Charts (`time_rating/graficas.py`)

A local server (http://localhost:8765, about 20 s to start, needs the
KayakKorner database running locally) with a name search. For up to six
athletes it draws KayakKorner's radar (same geometry as
`palista-radar/radar-geometria.ts`, relative to the chosen athlete's category)
and evolution chart (like `comparison-elo-history`), with the current Elo
from `EloHistorico` and the time rating side by side. Category cuts are
computed the same way for both (active that year with ≥ 3 races in the
distance, by category and sex), so the Elo percentiles may differ slightly
from the website's. The selection is kept in the URL (`?ids=`).

Below, a table of the chosen athlete's pruebas in the chosen distance:

- **Difficulty**: the prueba's μ against the median μ of all pruebas with the
  same announced distance and boat (≥ 10), in %: how much slower it was than
  usual (wind, current, a longer course).
- **Effective distance**: announced × e^(μ − typical), the distance it would
  have had in normal conditions. Piragua cross over "3000 m" comes out as
  about 4600 m (portages); some regional "4500 m" races as about 3000 m.
- **Field level**: mean level of the boats before the race, in points.
- **Performance**: how much faster (+) or slower (−) than the level predicted.

Between years, the difficulty of children's pruebas keeps whatever bias
their levels still have.

### State and next steps

The time rating (`grupo_tau2_qcat_mejora`) predicts better than the current
Elo (79.9 % vs 76.7 %, log loss 0.446 vs 0.477 on 2020–2026), its scale does
not widen, and from junior up a level is comparable between years; children
are comparable on average after the expected-improvement correction.
Pending:

- adaptive σ per distance and year (the probability scale drifts to 1.3 in
  2024–2026);
- a better shape for the children's improvement than linear in category
  steps (infantiles still high, cadetes low), and more held-out venues;
- how to bring it to KayakKorner: a separate engine next to `prediction/elo`,
  stored and shown in points (1500 = 2025 median, 25 points = 1 %).

## Chosen configuration

```
group_pruebas           True
context_mode            modifiers
length_transfer         matrix 2 (also used as confidence transfer)
k_factor                60
team_update_mode        inverse_sqrt
uncertainty_k_max       3
inactivity_mode         evidence_decay (grace 365 days, half-life 180 days)
primary_switch_delta    20
primary_lambda          20
modifier_lambda         5
field_size_reference    30
field_size_alpha        0.25
field_size_multiplier   0.5 – 1.75
start (KayakKorner)     1500 + 0.3·(performance in first group − 1500)
```

The first eleven come from experiments 1–3 (K was 60 and the uncertainty
multiplier 1.5 until experiment 4, then K=40, and back to 60 afterwards); the
start is not part of this repository's engine (with start 0 KayakKorner's engine reproduces `run_rating_system`).

| | Ungrouped acc / Brier / log loss | Grouped acc / log loss | Mean / established jump | Final std (p05–p95) |
|---|---|---|---|---|
| Defaults | 72.96 % / 0.1908 / 0.5659 | 74.73 % / 0.5342 | 10.5 / — | 124 (1338–1770) |
| After experiment 3 (K=60, ×1.5) | 74.08 % / 0.1792 / 0.5375 | 75.96 % / 0.5012 | 15.0 / 9.7 | 165 (1297–1864) |
| After experiment 4 (K=40) | — | 76.6 % / 0.4789 | 15.3 / 9.3 | — |
| **Final (K=60)** | — | **77.6 % / 0.4601** | 21.8 / 13.0 | — |

In experiment 3, among the configurations with a mean jump ≤ 15, the chosen
one was second by grouped log loss (the first won by 0.0005, which is noise),
had the best ungrouped log loss and kept inactivity decay, which is part of
the design.

**Caveat:** the configuration was chosen among about 1,000 evaluated on the same
2015–2026 data. The evaluation predicts each group before updating, so there is
no temporal leakage, but choosing among many configurations on the same data
overfits a little. The landscape is flat, so any configuration nearby would
give almost the same.

## Reproducing

From `elo/`, after exporting `dataset_kk.json` from KayakKorner:

```bash
python ../experiments/kayakkorner_2026_10/search_broad.py /tmp/broad 600 6
python ../experiments/kayakkorner_2026_10/transfer_matrices.py /tmp/transfer.json
python ../experiments/kayakkorner_2026_10/search_fine.py /tmp/fine 300 6
```

Each configuration takes about 100–150 s and 1.3 GB per worker; the broad
search took about 3 hours with 6 workers on a laptop.

Experiment 4 runs KayakKorner's engine: with the KayakKorner repository at
`~/projects/KayakKorner` and `stabilisation/annos.json` exported from its
database, edit the configurations at the bottom of `stabilisation/experiment.py`
and run it (about 7 minutes per 6 configurations).

KayakKorner replaced its ELO with this time rating (the Índice de rendimiento,
branch `feat/indice-rendimiento`) and deleted `prediction/elo`. Experiments 1–4
and the Elo replay of experiment 5 need a KayakKorner checkout from before that
change (for example commit `e044025`).

Experiment 5: `python time_rating/experiment.py [variant ...]` with the KayakKorner
virtualenv. The first run replays the Elo (about 4 minutes) and caches it in
`time_rating/cache.pkl`; each time-model variant then takes about 35 s.
`time_rating/anclas.py [variant ...] [--sedes ajuste|prueba]` runs the anchor
tests (about 75 s per variant), on all venues or on one half.
`time_rating/calibrar.py [mejora,arranque ...]` evaluates pairs on the fitting
half and appends them to `results/calibracion.jsonl` (about 1–5 minutes each;
several processes can run in parallel).
`time_rating/graficas.py` serves the charts at http://localhost:8765;
`graficas.py <ids or names>` writes the same page as a static
`time_rating/results/graficas.html` (not tracked).
