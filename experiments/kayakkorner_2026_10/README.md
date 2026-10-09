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

## Chosen configuration

```
group_pruebas           True
context_mode            modifiers
length_transfer         matrix 2 (also used as confidence transfer)
k_factor                60
team_update_mode        inverse_sqrt
uncertainty_k_max       1.5
inactivity_mode         evidence_decay (grace 365 days, half-life 180 days)
primary_switch_delta    20
primary_lambda          20
modifier_lambda         5
field_size_reference    30
field_size_alpha        0.25
field_size_multiplier   0.5 – 1.75
```

| | Ungrouped acc / Brier / log loss | Grouped acc / Brier / log loss | Mean / p95 / max jump | Final std (p05–p95) |
|---|---|---|---|---|
| Defaults | 72.96 % / 0.1908 / 0.5659 | 74.73 % / 0.1777 / 0.5342 | 10.5 / 29 / 66 | 124 (1338–1770) |
| **Chosen** | 74.08 % / 0.1792 / 0.5375 | 75.96 % / 0.1650 / 0.5012 | 15.0 / 42 / 110 | 165 (1297–1864) |

Among the configurations with a mean jump ≤ 15 it is second by grouped log
loss (the first wins by 0.0005, which is noise), has the best ungrouped log
loss and keeps inactivity decay, which is part of the design.

**Caveat:** the configuration was chosen among about 900 evaluated on the same
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
