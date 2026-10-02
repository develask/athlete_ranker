---
marp: true
title: Athlete Rating Engine
description: Data, design evolution, validation, and current state
paginate: true
size: 16:9
theme: default
style: |
  section { font-size: 27px; padding: 52px 64px; }
  h1 { color: #123B5D; }
  h2 { color: #176B87; }
  strong { color: #123B5D; }
  table { font-size: 21px; }
  code { background: #EEF5F8; }
  .small { font-size: 20px; }
  .tiny { font-size: 16px; }
  .accent { color: #D05A37; }
  .muted { color: #5F6B73; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 34px; }
  .metric { font-size: 38px; color: #176B87; font-weight: 700; }
---

# Athlete Rating Engine

## From raw canoe/kayak results to contextual, chronological Elo ratings

**Data coverage:** 2015–2026  
**Current implementation:** Elo V1  
**Repository snapshot:** 2 October 2026

---

# The story in one slide

1. We reconstruct structured races from noisy federation CSV exports.
2. We infer missing race metadata and convert exact distance into four distance profiles.
3. We clean and merge comparable classifications into performance groups.
4. We begin with standard Elo, then adapt it for fields, crews, distance, boat/type context, uncertainty, and inactivity.
5. Every prediction is evaluated **before** that result updates the athlete.

<span class="metric">20,783</span> rated athletes · <span class="metric">7,139</span> usable groups · <span class="metric">279,973</span> timed entries

---

# 1 · The raw data

Fourteen annual CSV files are present; 2013 and 2014 are empty, so effective coverage is **2015–2026**.

| Raw coverage | Value |
|---|---:|
| Result rows | 344,940 |
| Regattas, summed within annual files | 1,437 |
| Pruebas / classifications | 34,577 |
| Unique athlete IDs | 23,763 |
| Rows without time | 3,693 |
| Rows without position | 524 |

The raw unit is one boat’s result in one prueba. A boat can contain one or many athlete IDs.

---

# 2 · What information do we have?

<div class="columns">

**Regatta-level**

- Date
- Organizer ID and name
- Regatta ID and name
- League / discipline label

**Prueba-level**

- Prueba ID and name
- Phase and series
- Classification context embedded in free text

**Result-level**

- One or more athlete IDs
- Lane, bib, finishing position
- Finish time, points, annotations

</div>

The engine ultimately trusts **valid finish time** for order; raw position, points, and annotations are retained but not used for rating updates.

---

# 3 · What we derive from the source

The preparation pipeline parses race names and participation history to add:

- `tipo`: flatwater (`aguas_tranquilas`) or sea (`mar`)
- `sexo`: male, female, or mixed
- canonical age category: 9 supported categories
- boat type: kayak, canoe, or dragon boat
- crew size: mode of athlete count actually observed per boat
- exact distance, when it can be parsed from the prueba name
- distance class: super sprint, sprint, fondo, or marathon
- finish-time summary statistics for inspection and distance inference

Only flatwater and sea events are retained; paddleboard and paleo boats are excluded.

---

# 4 · The processed dataset

| Dimension | Distribution across 31,926 pruebas |
|---|---|
| Discipline | 28,856 flatwater · 3,070 sea |
| Sex | 17,986 male · 12,619 female · 1,321 mixed |
| Boat | 24,356 kayak · 6,967 canoe · 590 dragon boat · 13 unknown |
| Distance class | 7,088 super sprint · 6,784 sprint · 13,051 fondo · 5,003 marathon |
| Category | Infantil, cadete, junior, sub-23, senior, veterano, alevín, benjamín, prebenjamín |

Final processed scale: **1,378 regattas, 31,926 pruebas, 317,491 results, 23,110 athlete IDs**.

---

# 5 · Missing information: how it is reconstructed

Before filling, retained domain data had:

- 269 pruebas without sex
- 442 without category
- 1,314 without boat type
- 1,318 without exact distance / distance class

Known athlete history supplies profiles for sex, preferred discipline, preferred boat, and an estimated birth-year range. Missing fields are filled iteratively; original values are never overwritten, and inferred categories do not feed later birth-year estimates.

Internal reconstruction checks on known labels: **99.7% sex, 98.8% discipline, 98.7% boat, 95.1% category**.

<span class="tiny muted">These checks reuse full athlete histories and are data-reconstruction diagnostics, not strict future-prediction benchmarks.</span>

---

# 6 · Distance classes

| Class | Flatwater rule | Sea rule |
|---|---:|---:|
| Super sprint | ≤ 500 m | — |
| Sprint | 501–2,000 m | — |
| Fondo | 2,001–8,000 m | ≤ 8,000 m |
| Marathon | > 8,000 m | > 8,000 m |

For the 1,318 pruebas without a reliable exact distance, an Extra Trees classifier uses finish-time quantiles, field size, crew size, boat, discipline, sex, and category.

**Grouped 10-fold CV:** 94.63% accuracy and 0.9425 macro F1 on 30,608 labeled pruebas.

---

# 7 · From classifications to comparable performance groups

A rating group shares:

`regatta + date + phase + discipline + boat + crew size + exact distance + distance class`

Compatible pruebas can therefore be compared even when the source split them by sex, category, or series.

- 31,926 source pruebas become 7,517 raw groups
- 5,281 groups merge more than one source prueba
- 4,384 contain multiple sex classifications
- 3,275 contain multiple age categories
- Unknown-distance pruebas stay isolated instead of being merged blindly

This grouping expands the comparison network while preserving the physical race context.

---

# 8 · Cleaning before rating

Only entries with a valid athlete list, expected crew size, and time between **10 seconds and 3 hours** survive.

| Cleaning action | Removed |
|---|---:|
| Time below 10 s | 32,742 |
| Unparseable time | 2,430 |
| Time above 3 h | 225 |
| Crew-size mismatch | 1,638 |
| Same-crew near duplicate | 152 |
| Pruebas missing required context | 13 pruebas |

Same-crew records within 0.5 s are collapsed and assigned their mean time. Groups with fewer than two valid entries are skipped: **7,139 usable, 378 skipped**.

---

# 9 · Base model: standard Elo

For two competitors A and B:

$$E_A = \frac{1}{1 + 10^{(R_B-R_A)/400}}$$

$$\Delta_A = K(S_A-E_A)$$

- Every athlete starts at **1500**.
- $E_A$ is the expected score from the rating difference.
- $S_A$ is 1 for a win, 0 for a loss, 0.5 for a tie.
- A surprise result creates a larger update than an expected result.

This is the foundation. Everything that follows answers a limitation of plain two-player Elo.

---

# 10 · Addition 1: a whole finishing field

Each entry is compared pairwise with every valid opponent in its group.

$$S_i = \operatorname{mean}_{j\ne i}(S_{ij}), \qquad E_i = \operatorname{mean}_{j\ne i}(E_{ij})$$

$$\Delta_i = K_{eff}(S_i-E_i)$$

- Finish time determines win, loss, or tie.
- Pairwise values are averaged, so merely having more opponents does not multiply the update.
- Two entries sharing an athlete are not compared against each other.
- All ratings are frozen until every update for the group has been calculated.

Result: one mass-start field becomes a coherent multiplayer Elo event without update-order bias.

---

# 11 · Addition 2: crews and team boats

For a crew, the team rating is the mean of its athletes’ effective ratings:

$$R_{team}=\frac{1}{m}\sum_{a=1}^{m}R_a$$

The team result is converted to an athlete update with:

$$\Delta_{athlete}=\Delta_{team}\times\frac{1}{\sqrt{m}}$$

- K1 receives the full update.
- K2/C2 receives about 71% per athlete.
- K4/C4 receives 50% per athlete.
- The rule avoids crediting every crew member as though each raced alone.

If an athlete legitimately appears multiple times in one merged group, their pending updates are averaged before being applied.

---

# 12 · Addition 3: four ratings, not one

Each athlete carries one base rating for each distance profile:

`super_sprint · sprint · fondo · marathon`

Why? The physiology and tactics of 200 m, 1,000 m, 5,000 m, and marathon racing are related—but not identical.

A result updates the raced distance directly. It can also transfer limited evidence to nearby distances:

| From \ To | Super sprint | Sprint | Fondo | Marathon |
|---|---:|---:|---:|---:|
| Super sprint | 1.00 | 0.20 | 0.00 | 0.00 |
| Sprint | 0.20 | 1.00 | 0.05 | 0.00 |
| Fondo | 0.00 | 0.05 | 1.00 | 0.20 |
| Marathon | 0.00 | 0.00 | 0.20 | 1.00 |

Transferred updates are further reduced when the target rating is already well established.

---

# 13 · Addition 4: discipline and boat context

The default model represents performance as:

$$R_{effective}=R_{length}+M_{discipline}+M_{boat}$$

An athlete’s primary discipline and boat are the zero-point reference. Other contexts learn offsets—for example, the same athlete may be +35 in sea kayak relative to their flatwater baseline.

Three implementations can be tested:

- **Modifiers**: shared distance skill plus discipline/boat offsets — current default
- **Ignore**: distance skill only
- **Independent**: separate distance-rating sets for every discipline × boat context

Modifiers share scarce evidence without pretending all contexts are identical.

---

# 14 · Addition 5: confidence controls learning

For a component with effective evidence $n$:

$$U(n)=\frac{\lambda}{\lambda+n}$$

- Distance ratings use $\lambda=10$.
- Context modifiers use $\lambda=5$.
- New or weakly observed components have high uncertainty.
- The athlete update is allocated across distance, discipline, and boat in proportion to their uncertainties.
- The update is also multiplied by a factor from 1× to 2× according to the largest active uncertainty.

Interpretation: the model learns quickly when it knows little, then stabilizes as evidence accumulates.

---

# 15 · Addition 6: field-size-aware K

Large fields contain more ranking information than very small fields, so K is scaled:

$$K_{eff}=K\times\operatorname{clip}\left[\left(\frac{N}{20}\right)^{0.25},\ 0.70,\ 1.75\right]$$

With the current base $K=30$:

- very small field: minimum effective K = 21
- 20 entries: effective K = 30
- very large field: maximum effective K = 52.5

The shallow exponent and hard limits keep field size influential without letting mass events dominate the history.

---

# 16 · Addition 7: inactivity and changing specialism

**Inactivity**

- Rating points do not decay.
- After a 365-day grace period, effective evidence decays with a 730-day half-life.
- Returning athletes therefore become more responsive without being reset to average.

**Changing primary context**

- A challenger discipline or boat becomes primary after leading the current primary by 10 races.
- The engine rebases the baseline and modifiers so every effective rating is unchanged at the switch.

This separates “what we think the skill is” from “how certain we are about it.”

---

# 17 · One race through the final engine

1. Build the comparable performance group and clean its entries.
2. Decay stale evidence as of the race date.
3. Compute each crew’s pre-race effective rating.
4. Convert finish times into pairwise actual outcomes.
5. Compute pairwise expectations and one field-level team delta.
6. Scale for field size, crew size, and athlete uncertainty.
7. Allocate the delta to distance, discipline, and boat components.
8. Transfer limited evidence to adjacent distance classes.
9. Average duplicate athlete appearances within the group.
10. Apply all updates simultaneously and record evidence.

---

# 18 · Evaluation without look-ahead

Groups are processed chronologically. For every event:

**Predict first → score the prediction → update afterward**

Metrics:

- **Decisive accuracy:** was the higher-rated entry actually faster?
- **Brier score:** squared error of the win probability; lower is better.
- **Log loss:** strongly penalizes confident wrong predictions; lower is better.

Evaluation support is fixed across training configurations:

- strict, within-prueba view: 27,719 pruebas and 2,811,844 pairs
- grouped view: 7,137 units and 16,381,928 pairs

---

# 19 · Current default configuration

| Decision | Current default |
|---|---|
| Comparable pruebas | Merge into performance groups |
| Context representation | Additive modifiers |
| Crew update | $1/\sqrt{crew\ size}$ |
| Base K | 30 |
| Field scaling | reference 20, exponent 0.25, range 0.70–1.75 |
| Uncertainty multiplier | up to 2× |
| Inactivity | 365-day grace, 730-day evidence half-life |
| Primary-context switch | challenger leads by 10 races |
| Distance transfer | conservative adjacent-class matrix |

This is a deliberately conservative operating point, not the most aggressive search result.

---

# 20 · Current chronological results

| View | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Within original pruebas | **72.79%** | **0.1914** | **0.5673** |
| Grouped comparable performances | **74.57%** | **0.1784** | **0.5360** |

Other current-run facts:

- 20,783 athletes created
- 376,254 athlete-group updates
- Median absolute update: 8.0 points; 95th percentile: 29.4
- 2026 accuracy: 74.26% within pruebas and 78.50% grouped
- 2026 active-rating median: 1495; P05–P95: 1340–1769

<span class="tiny muted">Metrics were regenerated from the current code into a temporary output directory on 2 October 2026.</span>

---

# 21 · What the hyperparameter search found

The latest search contains **2,000 reproducibly sampled configurations** and ranks them by strict prueba-level log loss, then Brier score, then accuracy.

The top predictive candidate reached:

- **78.54%** strict decisive accuracy
- **0.1408** Brier score
- **0.4362** log loss

But it uses K=300, all-to-all distance transfer, no prueba merging for training, and produces a much wider, less stable population: final standard deviation 383 vs 123 for current defaults, with grouped accuracy only 72.10%.

**Conclusion:** it is a useful research lead, not an automatic production choice. Selection should include a held-out time period and explicit stability constraints.

---

# 22 · What the output contains

For every athlete ID:

- four distance ratings
- direct and effective evidence counts
- primary discipline and boat
- learned discipline and boat modifiers
- context-specific ratings when independent mode is used
- race counts, last evidence dates, and last-seen date

Supporting outputs include:

- final ratings in CSV and JSON
- group-level diagnostics
- run summary and year-by-year population health
- hyperparameter search results

The result is both a leaderboard input and an auditable model state.

---

# 23 · What the rating means—and does not mean

**It means:** expected relative finishing strength in a specified distance, discipline, and boat context, based on historical opponents and results.

**It does not mean:** an absolute time prediction, a causal measure of talent, or a guarantee across unobserved contexts.

Important limitations:

- IDs are assumed to represent athletes consistently across years.
- Parsed and inferred metadata can still be wrong.
- Group merging assumes same physical context is sufficient for comparison.
- Team ratings use the arithmetic mean; crew interaction is not modeled.
- The search currently uses the same historical span for tuning and reporting.
- Average rating is not strictly conserved after individual crew and uncertainty scaling.

---

# 24 · Recommended next steps

1. Reserve the latest season—or use rolling-origin windows—as a genuine untouched test set.
2. Run controlled ablations: add one feature at a time against the same support.
3. Rank hyperparameters on a combined objective: prediction + grouped performance + population stability.
4. Calibrate predicted probabilities by season and event type.
5. Audit inferred metadata with confidence flags and manual samples.
6. Decide the product-facing score: raw contextual Elo, percentile, or a stabilized public index.

The engine is operational; the next phase is model selection and product calibration.

---

# Takeaways

- The data pipeline turns noisy classifications into a consistent longitudinal competition graph.
- The base is familiar Elo; the additions solve real sport-specific problems one by one.
- The final rating combines distance ability with discipline and boat effects.
- Evidence, inactivity, crew size, and field size control how fast the engine learns.
- Current defaults deliver **72.79% strict** and **74.57% grouped** pairwise accuracy.
- Search results show more predictive headroom, but also a clear accuracy–stability tradeoff.

## The model is explainable, chronological, contextual, and ready for a disciplined validation phase.

---

# Appendix · Build evolution

| Stage | Main addition |
|---|---|
| Data foundation | Annual extraction, nested regatta/prueba/result structure |
| Data enrichment | Metadata parsing, athlete-profile imputation, distance classifier |
| Initial Elo engine | Performance groups, cleaning, multiplayer Elo, crews, distance ratings, context modifiers, uncertainty |
| Field-size revision | Field-aware K and reproducible hyperparameter search |
| Engine refactor | Context modes, evidence decay, uncertainty-driven K, primary-context switching, population diagnostics |
| Evaluation revision | Fixed canonical support and consistent grouped/ungrouped pre-update evaluation |

Code-history anchors: `1262b67` → `28e98a9` → `00f5209` → `89fd499`.

---

# Appendix · A simple numerical example

Suppose two K1 athletes race at sprint distance:

- A effective rating: 1600
- B effective rating: 1500
- Expected A win probability: 64.0%
- A wins, so the raw surprise is $1-0.640=0.360$
- With a two-entry field, current field multiplier is about 0.70
- Team delta before uncertainty: $30\times0.70\times0.360\approx7.6$

That 7.6 is then scaled by A’s uncertainty and allocated across A’s sprint rating and any non-primary discipline/boat modifiers. A fraction may transfer to super sprint or fondo.

The exact update is therefore contextual and confidence-aware, while the core logic remains standard Elo.
