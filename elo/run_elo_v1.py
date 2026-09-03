import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import rating_engine
from performance_groups import prepare_performance_groups
from rating_engine import (
    CONTEXT_MODES,
    FIELD_SIZE_ALPHA,
    FIELD_SIZE_MAX_MULTIPLIER,
    FIELD_SIZE_MIN_MULTIPLIER,
    FIELD_SIZE_REFERENCE,
    EVIDENCE_HALF_LIFE_DAYS,
    INACTIVITY_GRACE_DAYS,
    INACTIVITY_MODE,
    INACTIVITY_MODES,
    K_FACTOR,
    LENGTH_CLASSES,
    LENGTH_TRANSFER,
    MODIFIER_LAMBDA,
    PRIMARY_LAMBDA,
    PRIMARY_SWITCH_DELTA,
    TEAM_UPDATE_MODES,
    UNCERTAINTY_K_MAX_MULTIPLIER,
    athlete_debug_dict,
    build_evaluation_batches,
    calculate_evaluation_support,
    evaluate_performance_group,
    process_performance_group,
    run_rating_system,
)

NO_LENGTH_TRANSFER = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0,
        "fondo": 0.00,
        "maraton": 0.00,
    },
    "sprint": {"super_sprint": 0, "sprint": 1.00, "fondo": 0, "maraton": 0.00},
    "fondo": {"super_sprint": 0.00, "sprint": 0, "fondo": 1.00, "maraton": 0},
    "maraton": {"super_sprint": 0.00, "sprint": 0.00, "fondo": 0, "maraton": 1.00},
}

ALL_LENGTH_TRANSFER = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 1.00,
        "fondo": 1.00,
        "maraton": 1.00,
    },
    "sprint": {"super_sprint": 1.00, "sprint": 1.00, "fondo": 1.00, "maraton": 1.00},
    "fondo": {"super_sprint": 1.00, "sprint": 1.00, "fondo": 1.00, "maraton": 1.00},
    "maraton": {"super_sprint": 1.00, "sprint": 1.00, "fondo": 1.00, "maraton": 1.00},
}

LENGTH_TRANSFER_1 = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0.35,
        "fondo": 0.00,
        "maraton": 0.00,
    },
    "sprint": {"super_sprint": 0.35, "sprint": 1.00, "fondo": 0.10, "maraton": 0.00},
    "fondo": {"super_sprint": 0.00, "sprint": 0.10, "fondo": 1.00, "maraton": 0.35},
    "maraton": {"super_sprint": 0.00, "sprint": 0.00, "fondo": 0.35, "maraton": 1.00},
}

LENGTH_TRANSFER_2 = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0.5,
        "fondo": 0.05,
        "maraton": 0.00,
    },
    "sprint": {"super_sprint": 0.5, "sprint": 1.00, "fondo": 0.25, "maraton": 0.05},
    "fondo": {"super_sprint": 0.05, "sprint": 0.25, "fondo": 1.00, "maraton": 0.5},
    "maraton": {"super_sprint": 0.00, "sprint": 0.05, "fondo": 0.5, "maraton": 1.00},
}

LENGTH_TRANSFER_3 = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0.75,
        "fondo": 0.25,
        "maraton": 0.1,
    },
    "sprint": {"super_sprint": 0.75, "sprint": 1.00, "fondo": 0.5, "maraton": 0.15},
    "fondo": {"super_sprint": 0.15, "sprint": 0.5, "fondo": 1.00, "maraton": 0.75},
    "maraton": {"super_sprint": 0.1, "sprint": 0.25, "fondo": 0.75, "maraton": 1.00},
}

LENGTH_TRANSFER_4 = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0.75,
        "fondo": 0.5,
        "maraton": 0.25,
    },
    "sprint": {"super_sprint": 0.75, "sprint": 1.00, "fondo": 0.75, "maraton": 0.5},
    "fondo": {"super_sprint": 0.5, "sprint": 0.75, "fondo": 1.00, "maraton": 0.75},
    "maraton": {"super_sprint": 0.25, "sprint": 0.5, "fondo": 0.75, "maraton": 1.00},
}

LENGTH_TRANSFER_5 = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0.9,
        "fondo": 0.5,
        "maraton": 0.25,
    },
    "sprint": {"super_sprint": 0.9, "sprint": 1.00, "fondo": 0.9, "maraton": 0.5},
    "fondo": {"super_sprint": 0.5, "sprint": 0.9, "fondo": 1.00, "maraton": 0.9},
    "maraton": {"super_sprint": 0.25, "sprint": 0.5, "fondo": 0.9, "maraton": 1.00},
}

HYPERPARAM_GRID = {
    "group_pruebas": (True, False),
    "context_mode": ("modifiers", "ignore", "independent"),
    "team_update_mode": ("inverse_sqrt", "none"),
    "inactivity_mode": ("none", "evidence_decay"),
    "inactivity_grace_days": (180.0, 365.0),
    "evidence_half_life_days": (180.0, 365.0),
    "uncertainty_k_max_multiplier": (1.0, 1.5, 2.0, 3.0, 4.0),
    "primary_switch_delta": (
        5,
        10,
        20,
        999_999_999,
    ),  # 999_999_999 is a placeholder for no switch
    "k_factor": (50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0),
    "field_size_alpha": (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    "field_size_reference": (5, 10, 20, 30, 50, 75, 100),
    "field_size_min_multiplier": (0.50, 0.667, 0.8),
    "field_size_max_multiplier": (1.25, 1.5, 2.0),
    "primary_lambda": (10.0, 15.0, 20.0),  # (15.0, 20.0),
    "modifier_lambda": (5.0, 10.0),  # (5.0, 10.0),
    "length_transfer": (
        NO_LENGTH_TRANSFER,
        ALL_LENGTH_TRANSFER,
        LENGTH_TRANSFER_1,
        LENGTH_TRANSFER_2,
        LENGTH_TRANSFER_3,
        LENGTH_TRANSFER_4,
        LENGTH_TRANSFER_5,
    ),
}

_SEARCH_GROUPS = None
_SEARCH_EXPECTED_SUPPORT = None
HYPERPARAM_RANDOM_SEED = 42


def safe_mean(values):
    values = list(values)

    if not values:
        return None

    return sum(values) / len(values)


def percentile(values, q):
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return values[lower]

    weight = position - lower

    return values[lower] * (1.0 - weight) + values[upper] * weight


def distribution(values):
    values = [
        float(value)
        for value in values
        if (value is not None and math.isfinite(float(value)))
    ]

    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p01": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    mean = safe_mean(values)
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p01": percentile(values, 0.01),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(safe_mean((value - mean) ** 2 for value in values)),
    }


def serialize_counter(counter):
    return {str(key): value for key, value in counter.items()}


def year_from_fecha(fecha):
    year = str(fecha or "")[:4]
    return year if year.isdigit() else "unknown"


class PopulationHealthTracker:
    """Collect coordinate-consistent population and update diagnostics by year."""

    def __init__(self, context_mode):
        self.context_mode = context_mode
        self.current_year = None
        self.active_rating_keys = set()
        self.new_athletes = set()
        self.seen_athletes = set()
        self.year_updates = []
        self.all_updates = []
        self.previous_snapshot = {}
        self.by_year = {}

    def before_group(self, athletes, group):
        year = year_from_fecha(group.fecha)
        if self.current_year is None:
            self.current_year = year
        elif year != self.current_year:
            self._finish_year(athletes)
            self.current_year = year

    def after_group(self, group, result):
        for athlete_id, update in result.athlete_updates.items():
            if athlete_id not in self.seen_athletes:
                self.new_athletes.add(athlete_id)
                self.seen_athletes.add(athlete_id)

            if self.context_mode == "independent":
                key = (
                    athlete_id,
                    update.direct_length,
                    update.tipo,
                    update.boat,
                )
            else:
                key = (athlete_id, update.direct_length, None, None)
            self.active_rating_keys.add(key)
            self.year_updates.append(update.base_delta)
            self.all_updates.append(update.base_delta)

    def finalize(self, athletes):
        if self.current_year is not None:
            self._finish_year(athletes)

    def _rating_value(self, athletes, key):
        athlete_id, length_class, tipo, boat = key
        athlete = athletes[athlete_id]
        if self.context_mode != "independent":
            tipo = athlete.primary_tipo
            boat = athlete.primary_boat
        return rating_engine.get_effective_rating(
            athlete,
            length_class,
            tipo,
            boat,
            context_mode=self.context_mode,
        )

    def _finish_year(self, athletes):
        snapshot = {
            key: self._rating_value(athletes, key) for key in self.active_rating_keys
        }
        rating_values = list(snapshot.values())
        rating_report = distribution(rating_values)
        cohort_deltas = [
            rating - self.previous_snapshot[key]
            for key, rating in snapshot.items()
            if key in self.previous_snapshot
        ]
        absolute_updates = [abs(value) for value in self.year_updates]

        by_length = {}
        for length_class in LENGTH_CLASSES:
            values = [
                rating for key, rating in snapshot.items() if key[1] == length_class
            ]
            by_length[length_class] = distribution(values)

        self.by_year[self.current_year] = {
            "active_athletes": len({key[0] for key in snapshot}),
            "active_rating_records": len(snapshot),
            "new_athletes": len(self.new_athletes),
            "ratings": rating_report,
            "ratings_by_length": by_length,
            "below_1000": sum(value < 1000 for value in rating_values),
            "above_2000": sum(value > 2000 for value in rating_values),
            "below_1000_percentage": (
                sum(value < 1000 for value in rating_values) / len(rating_values)
                if rating_values
                else None
            ),
            "above_2000_percentage": (
                sum(value > 2000 for value in rating_values) / len(rating_values)
                if rating_values
                else None
            ),
            "cohort_rating_change": distribution(cohort_deltas),
            "updates": {
                "net_delta": sum(self.year_updates),
                "signed": distribution(self.year_updates),
                "absolute": distribution(absolute_updates),
            },
        }
        self.previous_snapshot = snapshot
        self.active_rating_keys = set()
        self.new_athletes = set()
        self.year_updates = []

    def aggregate(self):
        years = list(self.by_year.values())
        spreads = [
            report["ratings"]["p95"] - report["ratings"]["p05"]
            for report in years
            if report["ratings"]["p95"] is not None
        ]
        cohort_medians = [
            report["cohort_rating_change"]["median"]
            for report in years
            if report["cohort_rating_change"]["median"] is not None
        ]
        final_ratings = years[-1]["ratings"] if years else distribution([])
        return {
            "final_rating_mean": final_ratings["mean"],
            "final_rating_median": final_ratings["median"],
            "final_rating_std": final_ratings["std"],
            "final_rating_p05": final_ratings["p05"],
            "final_rating_p95": final_ratings["p95"],
            "max_p95_p05_spread": max(spreads) if spreads else None,
            "spread_change": spreads[-1] - spreads[0] if spreads else None,
            "max_abs_cohort_median_drift": (
                max(abs(value) for value in cohort_medians) if cohort_medians else None
            ),
            "net_delta": sum(self.all_updates),
            "absolute_updates": distribution(abs(value) for value in self.all_updates),
        }


def write_final_json(athletes, path):
    payload = {
        athlete_id: (athlete_debug_dict(athlete))
        for athlete_id, athlete in sorted(athletes.items(), key=lambda item: item[0])
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_final_csv(athletes, path):
    fieldnames = ["athlete_id", "primary_tipo", "primary_boat", "last_seen"]

    for length_class in LENGTH_CLASSES:
        fieldnames.extend(
            [
                (length_class + "_elo"),
                (length_class + "_n_direct"),
                (length_class + "_n_effective"),
            ]
        )

    fieldnames.extend(
        [
            "tipo_modifiers_json",
            "boat_modifiers_json",
            "context_length_ratings_json",
            "tipo_race_counts_json",
            "boat_race_counts_json",
            "tipo_effective_counts_json",
            "boat_effective_counts_json",
        ]
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for athlete_id, athlete in sorted(athletes.items(), key=lambda item: item[0]):
            row = {
                "athlete_id": athlete_id,
                "primary_tipo": (athlete.primary_tipo),
                "primary_boat": (athlete.primary_boat),
                "last_seen": athlete.last_seen,
            }

            for length_class in LENGTH_CLASSES:
                rating = athlete.length_ratings[length_class]

                row[length_class + "_elo"] = rating.elo
                row[length_class + "_n_direct"] = rating.n_direct
                row[length_class + "_n_effective"] = rating.n_effective

            row["tipo_modifiers_json"] = json.dumps(
                {
                    key: {
                        "modifier": (value.raw_modifier),
                        "n_direct": (value.n_direct),
                        "n_effective": (value.n_effective),
                    }
                    for key, value in athlete.tipo_modifiers.items()
                },
                ensure_ascii=False,
                sort_keys=True,
            )

            row["boat_modifiers_json"] = json.dumps(
                {
                    key: {
                        "modifier": (value.raw_modifier),
                        "n_direct": (value.n_direct),
                        "n_effective": (value.n_effective),
                    }
                    for key, value in athlete.boat_modifiers.items()
                },
                ensure_ascii=False,
                sort_keys=True,
            )

            row["context_length_ratings_json"] = json.dumps(
                {
                    f"{tipo}|{boat}": {
                        length: {
                            "elo": rating.elo,
                            "n_direct": rating.n_direct,
                            "n_effective": rating.n_effective,
                        }
                        for length, rating in ratings.items()
                    }
                    for (tipo, boat), ratings in athlete.context_length_ratings.items()
                },
                ensure_ascii=False,
                sort_keys=True,
            )

            row["tipo_race_counts_json"] = json.dumps(
                dict(athlete.tipo_race_counts), ensure_ascii=False, sort_keys=True
            )

            row["boat_race_counts_json"] = json.dumps(
                dict(athlete.boat_race_counts), ensure_ascii=False, sort_keys=True
            )

            row["tipo_effective_counts_json"] = json.dumps(
                dict(athlete.tipo_effective_counts),
                ensure_ascii=False,
                sort_keys=True,
            )

            row["boat_effective_counts_json"] = json.dumps(
                dict(athlete.boat_effective_counts),
                ensure_ascii=False,
                sort_keys=True,
            )

            writer.writerow(row)


def build_rating_distributions(athletes, context_mode="modifiers"):
    output = {}

    for length_class in LENGTH_CLASSES:
        all_values = []
        direct_values = []

        for athlete in athletes.values():
            if context_mode == "independent":
                ratings = (
                    context_ratings[length_class]
                    for context_ratings in athlete.context_length_ratings.values()
                )
            else:
                ratings = (athlete.length_ratings[length_class],)

            for rating in ratings:
                all_values.append(rating.elo)
                if rating.n_direct > 0:
                    direct_values.append(rating.elo)

        output[length_class] = {
            "all_athletes": (distribution(all_values)),
            "direct_participants": (distribution(direct_values)),
        }

    return output


def build_modifier_distributions(athletes):
    tipo_values = []
    boat_values = []

    for athlete in athletes.values():
        tipo_values.extend(
            modifier.raw_modifier for modifier in athlete.tipo_modifiers.values()
        )

        boat_values.extend(
            modifier.raw_modifier for modifier in athlete.boat_modifiers.values()
        )

    return {"tipo": distribution(tipo_values), "boat": distribution(boat_values)}


def evaluation_summary(metrics):
    comparisons = metrics["pairwise_comparisons"]
    decisive = metrics["pairwise_decisive"]

    return {
        **serialize_counter(metrics),
        "brier_score": (metrics["brier_sum"] / comparisons if comparisons else None),
        "log_loss": (metrics["log_loss_sum"] / comparisons if comparisons else None),
        "decisive_accuracy": (
            metrics["correct_decisive"] / decisive if decisive else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=("Run canoe/kayak Elo V1 " "chronologically.")
    )
    parser.add_argument(
        "--dataset",
        default="../data/processed/dataset.json",
        help="Path to dataset.json",
    )
    parser.add_argument(
        "--output-dir",
        default="elo_v1_output",
        help=("Directory for outputs " "(default: elo_v1_output)"),
    )
    parser.add_argument(
        "--k", type=float, default=K_FACTOR, help=("K factor " f"(default: {K_FACTOR})")
    )
    parser.add_argument(
        "--field-size-reference",
        type=float,
        default=FIELD_SIZE_REFERENCE,
        help=f"Reference number of entries (default: {FIELD_SIZE_REFERENCE})",
    )
    parser.add_argument(
        "--field-size-alpha",
        type=float,
        default=FIELD_SIZE_ALPHA,
        help=f"Field-size scaling exponent (default: {FIELD_SIZE_ALPHA})",
    )
    parser.add_argument(
        "--field-size-min-multiplier",
        type=float,
        default=FIELD_SIZE_MIN_MULTIPLIER,
        help=f"Minimum field-size multiplier (default: {FIELD_SIZE_MIN_MULTIPLIER})",
    )
    parser.add_argument(
        "--field-size-max-multiplier",
        type=float,
        default=FIELD_SIZE_MAX_MULTIPLIER,
        help=f"Maximum field-size multiplier (default: {FIELD_SIZE_MAX_MULTIPLIER})",
    )
    parser.add_argument(
        "--hyperparam-search",
        action="store_true",
        help="Run HYPERPARAM_GRID instead of producing final ratings",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel hyperparameter workers; -1 uses every CPU (default: 1)",
    )
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=1000,
        help="Random hyperparameter configurations to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--group-pruebas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge compatible pruebas for updates",
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_MODES,
        default="modifiers",
        help="How tipo/boat contexts are represented (default: modifiers)",
    )
    parser.add_argument(
        "--team-update-mode",
        choices=TEAM_UPDATE_MODES,
        default="inverse_sqrt",
        help="Per-athlete team delta scaling (default: inverse_sqrt)",
    )
    parser.add_argument(
        "--inactivity-mode",
        choices=INACTIVITY_MODES,
        default=INACTIVITY_MODE,
        help=f"How inactivity affects evidence and updates (default: {INACTIVITY_MODE})",
    )
    parser.add_argument(
        "--inactivity-grace-days",
        type=float,
        default=INACTIVITY_GRACE_DAYS,
        help=f"Days before inactivity takes effect (default: {INACTIVITY_GRACE_DAYS:g})",
    )
    parser.add_argument(
        "--evidence-half-life-days",
        type=float,
        default=EVIDENCE_HALF_LIFE_DAYS,
        help=f"Inactive evidence half-life (default: {EVIDENCE_HALF_LIFE_DAYS:g})",
    )
    parser.add_argument(
        "--uncertainty-k-max-multiplier",
        type=float,
        default=UNCERTAINTY_K_MAX_MULTIPLIER,
        help=f"Maximum uncertainty-based update multiplier (default: {UNCERTAINTY_K_MAX_MULTIPLIER:g})",
    )
    parser.add_argument(
        "--primary-switch-delta",
        type=int,
        default=PRIMARY_SWITCH_DELTA,
        help=f"Usage lead required to change a primary context (default: {PRIMARY_SWITCH_DELTA})",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", dataset_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if args.hyperparam_search:
        group_sets = {}
        grouping_modes = set(HYPERPARAM_GRID["group_pruebas"])
        grouping_modes.add(True)  # Canonical groups used by every evaluation.
        for group_pruebas in sorted(grouping_modes, reverse=True):
            if group_pruebas in group_sets:
                continue
            print(f"Building performance groups (group_pruebas={group_pruebas})...")
            group_sets[group_pruebas], _ = prepare_performance_groups(
                raw_data,
                keep_unusable=True,
                group_pruebas=group_pruebas,
            )
        results = hyperparam_search(
            group_sets,
            n_jobs=args.n_jobs,
            n_experiments=args.n_experiments,
        )
        json_path = output_dir / "hyperparam_search.json"
        csv_path = output_dir / "hyperparam_search.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        csv_fields = [
            "rank",
            "k_mode",
            "group_pruebas",
            "context_mode",
            "team_update_mode",
            "inactivity_mode",
            "inactivity_grace_days",
            "evidence_half_life_days",
            "uncertainty_k_max_multiplier",
            "primary_switch_delta",
            "k_factor",
            "field_size_reference",
            "field_size_alpha",
            "field_size_min_multiplier",
            "field_size_max_multiplier",
            "primary_lambda",
            "modifier_lambda",
            "length_transfer_index",
            "pruebas_evaluated",
            "pairwise_comparisons",
            "brier_score",
            "log_loss",
            "decisive_accuracy",
            "grouped_units_evaluated",
            "grouped_pairwise_comparisons",
            "grouped_pairwise_decisive",
            "grouped_brier_score",
            "grouped_log_loss",
            "grouped_decisive_accuracy",
            "final_rating_mean",
            "final_rating_median",
            "final_rating_std",
            "final_rating_p05",
            "final_rating_p95",
            "max_p95_p05_spread",
            "spread_change",
            "max_abs_cohort_median_drift",
            "net_delta",
            "mean_absolute_update",
            "p95_absolute_update",
            "max_absolute_update",
            "elapsed_seconds",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(
                {key: result[key] for key in csv_fields} for result in results
            )

        print("\nHyperparameter search outputs:", output_dir.resolve())
        return

    print(f"Building performance groups (group_pruebas={args.group_pruebas})...")
    groups, prep_stats = prepare_performance_groups(
        raw_data,
        keep_unusable=True,
        group_pruebas=args.group_pruebas,
    )
    if args.group_pruebas:
        evaluation_groups = groups
    else:
        print("Building canonical groups for evaluation...")
        evaluation_groups, _ = prepare_performance_groups(
            raw_data,
            keep_unusable=True,
            group_pruebas=True,
        )
    evaluation_batches = build_evaluation_batches(groups, evaluation_groups)

    athletes = {}
    population_tracker = PopulationHealthTracker(args.context_mode)
    rating_stats = Counter()
    evaluations = {
        "ungrouped": Counter(),
        "grouped": Counter(),
    }
    evaluations_by_year = {
        "ungrouped": {},
        "grouped": {},
    }
    rating_distributions_by_year = {}
    current_year = None

    diagnostics_path = output_dir / "group_diagnostics.csv"

    diagnostic_fields = [
        "regata_id",
        "fecha",
        "fase",
        "tipo",
        "boat",
        "boat_num",
        "distance",
        "length_class",
        "source_pruebas",
        "source_series",
        "source_sexes",
        "source_categories",
        "entries",
        "n_entries",
        "base_k",
        "field_size_multiplier",
        "effective_k",
        "remaining_overlap_athletes",
        "near_duplicate_rows_removed",
        "crew_size_mismatch_removed",
        "processed",
        "skip_reason",
        "mean_abs_team_delta",
        "pairwise_comparisons",
    ]

    print("Running chronological Elo...")

    with diagnostics_path.open("w", newline="", encoding="utf-8") as diagnostic_file:
        writer = csv.DictWriter(diagnostic_file, fieldnames=(diagnostic_fields))
        writer.writeheader()

        processed_group_count = 0
        for evaluation_group, training_groups in evaluation_batches:
            group_year = year_from_fecha(evaluation_group.fecha)
            if current_year is None:
                current_year = group_year
            elif group_year != current_year:
                rating_distributions_by_year[current_year] = build_rating_distributions(
                    athletes, context_mode=args.context_mode
                )
                current_year = group_year

            population_tracker.before_group(athletes, evaluation_group)
            ungrouped_evaluation, grouped_evaluation = evaluate_performance_group(
                athletes=athletes,
                group=evaluation_group,
                context_mode=args.context_mode,
                inactivity_mode=args.inactivity_mode,
                inactivity_grace_days=args.inactivity_grace_days,
                evidence_half_life_days=args.evidence_half_life_days,
            )
            evaluations["ungrouped"].update(ungrouped_evaluation)
            evaluations["grouped"].update(grouped_evaluation)

            if ungrouped_evaluation["pairwise_comparisons"]:
                evaluations_by_year["ungrouped"].setdefault(
                    group_year, Counter()
                ).update(ungrouped_evaluation)
            if grouped_evaluation["pairwise_comparisons"]:
                evaluations_by_year["grouped"].setdefault(group_year, Counter()).update(
                    grouped_evaluation
                )

            for child_index, group in enumerate(training_groups):
                processed_group_count += 1
                result = process_performance_group(
                    athletes=athletes,
                    group=group,
                    k=args.k,
                    field_size_reference=args.field_size_reference,
                    field_size_alpha=args.field_size_alpha,
                    field_size_min_multiplier=args.field_size_min_multiplier,
                    field_size_max_multiplier=args.field_size_max_multiplier,
                    context_mode=args.context_mode,
                    team_update_mode=args.team_update_mode,
                    inactivity_mode=args.inactivity_mode,
                    inactivity_grace_days=args.inactivity_grace_days,
                    evidence_half_life_days=args.evidence_half_life_days,
                    uncertainty_k_max_multiplier=args.uncertainty_k_max_multiplier,
                    primary_switch_delta=args.primary_switch_delta,
                )

                if result.processed:
                    population_tracker.after_group(group, result)
                    rating_stats["groups_processed"] += 1
                    rating_stats["entries_processed"] += len(group.entries)
                    rating_stats["athlete_group_updates"] += len(result.athlete_updates)
                    rating_stats["athletes_with_multiple_appearances"] += result.stats[
                        "athletes_with_multiple_appearances"
                    ]
                else:
                    rating_stats["groups_skipped"] += 1
                    rating_stats["skip_" + str(result.skip_reason)] += 1

                mean_abs_delta = None

                if result.team_base_deltas:
                    mean_abs_delta = safe_mean(
                        abs(value) for value in result.team_base_deltas
                    )

                writer.writerow(
                    {
                        "regata_id": (group.regata_id),
                        "fecha": group.fecha,
                        "fase": (group.prueba_fase),
                        "tipo": group.tipo,
                        "boat": (group.embarcacion_tipo),
                        "boat_num": (group.embarcacion_num),
                        "distance": (group.distancia_exacta),
                        "length_class": (group.length_class),
                        "source_pruebas": (
                            "|".join(str(value) for value in group.source_prueba_ids)
                        ),
                        "source_series": (
                            "|".join(str(value) for value in group.source_series)
                        ),
                        "source_sexes": (
                            "|".join(str(value) for value in group.source_sexes)
                        ),
                        "source_categories": (
                            "|".join(str(value) for value in group.source_categories)
                        ),
                        "entries": len(group.entries),
                        "n_entries": result.n_entries,
                        "base_k": result.base_k,
                        "field_size_multiplier": result.field_size_multiplier,
                        "effective_k": result.effective_k,
                        "remaining_overlap_athletes": (
                            len(group.overlapping_athlete_ids)
                        ),
                        "near_duplicate_rows_removed": (
                            group.cleaning_stats["discard_near_duplicate_crew_entry"]
                        ),
                        "crew_size_mismatch_removed": (
                            group.cleaning_stats["discard_crew_size_mismatch"]
                        ),
                        "processed": (result.processed),
                        "skip_reason": (result.skip_reason),
                        "mean_abs_team_delta": (mean_abs_delta),
                        "pairwise_comparisons": (
                            ungrouped_evaluation["pairwise_comparisons"]
                            if child_index == 0
                            else 0
                        ),
                    }
                )

                if processed_group_count % 500 == 0:
                    print(f"  processed {processed_group_count}/{len(groups)} groups")

    if current_year is not None:
        rating_distributions_by_year[current_year] = build_rating_distributions(
            athletes, context_mode=args.context_mode
        )
    population_tracker.finalize(athletes)

    rating_stats["athletes_created"] = len(athletes)

    print("Writing final ratings...")

    final_json = output_dir / "final_ratings.json"
    final_csv = output_dir / "final_ratings.csv"

    write_final_json(athletes, final_json)
    write_final_csv(athletes, final_csv)

    summary = {
        "version": "elo_v1",
        "parameters": {
            "k_mode": "field_size_scaled",
            "group_pruebas": args.group_pruebas,
            "context_mode": args.context_mode,
            "team_update_mode": args.team_update_mode,
            "inactivity_mode": args.inactivity_mode,
            "inactivity_grace_days": args.inactivity_grace_days,
            "evidence_half_life_days": args.evidence_half_life_days,
            "uncertainty_k_max_multiplier": args.uncertainty_k_max_multiplier,
            "primary_switch_delta": args.primary_switch_delta,
            "k_factor": args.k,
            "field_size_reference": args.field_size_reference,
            "field_size_alpha": args.field_size_alpha,
            "field_size_min_multiplier": args.field_size_min_multiplier,
            "field_size_max_multiplier": args.field_size_max_multiplier,
            "primary_lambda": PRIMARY_LAMBDA,
            "modifier_lambda": MODIFIER_LAMBDA,
            "length_transfer": (LENGTH_TRANSFER),
        },
        "preprocessing": (serialize_counter(prep_stats)),
        "rating_run": (serialize_counter(rating_stats)),
        "prediction_evaluation": {
            scope: evaluation_summary(metrics) for scope, metrics in evaluations.items()
        },
        "prediction_evaluation_by_year": {
            scope: {
                year: evaluation_summary(yearly_metrics)
                for year, yearly_metrics in sorted(yearly_values.items())
            }
            for scope, yearly_values in evaluations_by_year.items()
        },
        "rating_distributions": build_rating_distributions(
            athletes, context_mode=args.context_mode
        ),
        "rating_distributions_by_year": {
            year: values
            for year, values in sorted(rating_distributions_by_year.items())
        },
        "modifier_distributions": (build_modifier_distributions(athletes)),
        "population_health_by_year": population_tracker.by_year,
        "population_health_summary": population_tracker.aggregate(),
    }

    summary_path = output_dir / "run_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Athletes:", len(athletes))
    print("Groups processed:", rating_stats["groups_processed"])
    print("Groups skipped:", rating_stats["groups_skipped"])

    def metric_text(value):
        return f"{value:.6f}" if value is not None else "n/a"

    def elo_text(value):
        return f"{value:.1f}" if value is not None else "n/a"

    for scope, yearly_reports in summary["prediction_evaluation_by_year"].items():
        unit_label = "Groups" if scope == "grouped" else "Pruebas"
        print(f"\nPrediction evaluation by year ({scope}):")
        print(
            f"{'Year':<8} {unit_label:>8} "
            f"{'Pairs':>12} {'Decisive':>12} "
            f"{'Brier':>10} {'Log loss':>10} "
            f"{'Accuracy':>10}"
        )
        for year, report in yearly_reports.items():
            print(
                f"{year:<8} "
                f"{report['pruebas_evaluated']:>8} "
                f"{report['pairwise_comparisons']:>12} "
                f"{report['pairwise_decisive']:>12} "
                f"{metric_text(report['brier_score']):>10} "
                f"{metric_text(report['log_loss']):>10} "
                f"{metric_text(report['decisive_accuracy']):>10}"
            )

    print("\nOverall prediction evaluation:")
    for scope, evaluation_report in summary["prediction_evaluation"].items():
        print(f"  {scope}:")
        print("    Pairwise Brier:", metric_text(evaluation_report["brier_score"]))
        print("    Pairwise log loss:", metric_text(evaluation_report["log_loss"]))
        print(
            "    Decisive accuracy:",
            metric_text(evaluation_report["decisive_accuracy"]),
        )

    print("\nElo population health by year:")
    print(
        f"{'Year':<8} {'Active':>7} {'New':>7} "
        f"{'Mean':>9} {'Median':>9} {'Std':>9} "
        f"{'P05':>9} {'P95':>9} {'Spread':>9} "
        f"{'CohortΔ':>9} {'NetΔ':>11} {'P95|Δ|':>9}"
    )
    for year, report in summary["population_health_by_year"].items():
        ratings = report["ratings"]
        updates = report["updates"]
        cohort = report["cohort_rating_change"]
        spread = ratings["p95"] - ratings["p05"] if ratings["p95"] is not None else None
        print(
            f"{year:<8} {report['active_athletes']:>7} "
            f"{report['new_athletes']:>7} "
            f"{elo_text(ratings['mean']):>9} "
            f"{elo_text(ratings['median']):>9} "
            f"{elo_text(ratings['std']):>9} "
            f"{elo_text(ratings['p05']):>9} "
            f"{elo_text(ratings['p95']):>9} "
            f"{elo_text(spread):>9} "
            f"{elo_text(cohort['median']):>9} "
            f"{elo_text(updates['net_delta']):>11} "
            f"{elo_text(updates['absolute']['p95']):>9}"
        )

    print("\nEnd-of-year Elo distributions (direct participants):")
    print(
        f"{'Year':<8} {'Length':<14} {'N':>7} "
        f"{'Min':>9} {'P05':>9} {'P25':>9} "
        f"{'Median':>9} {'P75':>9} {'P95':>9} "
        f"{'Max':>9} {'Mean':>9}"
    )
    for year, length_reports in summary["rating_distributions_by_year"].items():
        for length_class in LENGTH_CLASSES:
            report = length_reports[length_class]["direct_participants"]
            print(
                f"{year:<8} {length_class:<14} "
                f"{report['count']:>7} "
                f"{elo_text(report['min']):>9} "
                f"{elo_text(report['p05']):>9} "
                f"{elo_text(report['p25']):>9} "
                f"{elo_text(report['median']):>9} "
                f"{elo_text(report['p75']):>9} "
                f"{elo_text(report['p95']):>9} "
                f"{elo_text(report['max']):>9} "
                f"{elo_text(report['mean']):>9}"
            )

    print("\nOutputs:", output_dir.resolve())


def _evaluation_support_signature(evaluations):
    keys = (
        "pruebas_evaluated",
        "pairwise_comparisons",
        "pairwise_decisive",
        "pairwise_ties",
    )
    return tuple(
        evaluations[scope][key] for scope in ("ungrouped", "grouped") for key in keys
    )


def _init_search_worker(groups, expected_support):
    global _SEARCH_GROUPS, _SEARCH_EXPECTED_SUPPORT
    _SEARCH_GROUPS = groups
    _SEARCH_EXPECTED_SUPPORT = expected_support


def _evaluate_hyperparam_candidate(candidate):
    if _SEARCH_GROUPS is None or _SEARCH_EXPECTED_SUPPORT is None:
        raise RuntimeError("Hyperparameter worker has no performance groups")

    original_parameters = {
        "primary_lambda": rating_engine.PRIMARY_LAMBDA,
        "modifier_lambda": rating_engine.MODIFIER_LAMBDA,
        "length_transfer": rating_engine.LENGTH_TRANSFER,
        "confidence_transfer": rating_engine.CONFIDENCE_TRANSFER,
    }
    try:
        rating_engine.PRIMARY_LAMBDA = candidate["primary_lambda"]
        rating_engine.MODIFIER_LAMBDA = candidate["modifier_lambda"]
        rating_engine.LENGTH_TRANSFER = candidate["length_transfer"]
        rating_engine.CONFIDENCE_TRANSFER = candidate["length_transfer"]

        started = perf_counter()
        population_tracker = PopulationHealthTracker(candidate["context_mode"])
        groups = _SEARCH_GROUPS[candidate["group_pruebas"]]
        _, _, evaluations, _ = run_rating_system(
            groups,
            k=candidate["k_factor"],
            field_size_reference=candidate["field_size_reference"],
            field_size_alpha=candidate["field_size_alpha"],
            field_size_min_multiplier=candidate["field_size_min_multiplier"],
            field_size_max_multiplier=candidate["field_size_max_multiplier"],
            context_mode=candidate["context_mode"],
            team_update_mode=candidate["team_update_mode"],
            inactivity_mode=candidate["inactivity_mode"],
            inactivity_grace_days=candidate["inactivity_grace_days"],
            evidence_half_life_days=candidate["evidence_half_life_days"],
            uncertainty_k_max_multiplier=candidate["uncertainty_k_max_multiplier"],
            primary_switch_delta=candidate["primary_switch_delta"],
            evaluation_groups=_SEARCH_GROUPS[True],
            population_tracker=population_tracker,
        )
        ungrouped_report = evaluation_summary(evaluations["ungrouped"])
        grouped_report = evaluation_summary(evaluations["grouped"])
        actual_support = _evaluation_support_signature(evaluations)
        if actual_support != _SEARCH_EXPECTED_SUPPORT:
            raise RuntimeError(
                "Evaluation support changed with the training configuration: "
                f"expected {_SEARCH_EXPECTED_SUPPORT}, got {actual_support}"
            )
        population_summary = population_tracker.aggregate()
        elapsed = perf_counter() - started
    finally:
        rating_engine.PRIMARY_LAMBDA = original_parameters["primary_lambda"]
        rating_engine.MODIFIER_LAMBDA = original_parameters["modifier_lambda"]
        rating_engine.LENGTH_TRANSFER = original_parameters["length_transfer"]
        rating_engine.CONFIDENCE_TRANSFER = original_parameters["confidence_transfer"]

    return {
        "rank": None,
        **candidate,
        # Keep the established names for strict prueba-level metrics so old
        # analysis scripts continue to work and ranking remains comparable.
        "pruebas_evaluated": ungrouped_report["pruebas_evaluated"],
        "pairwise_comparisons": ungrouped_report["pairwise_comparisons"],
        "pairwise_decisive": ungrouped_report["pairwise_decisive"],
        "brier_score": ungrouped_report["brier_score"],
        "log_loss": ungrouped_report["log_loss"],
        "decisive_accuracy": ungrouped_report["decisive_accuracy"],
        "grouped_units_evaluated": grouped_report["pruebas_evaluated"],
        "grouped_pairwise_comparisons": grouped_report["pairwise_comparisons"],
        "grouped_pairwise_decisive": grouped_report["pairwise_decisive"],
        "grouped_brier_score": grouped_report["brier_score"],
        "grouped_log_loss": grouped_report["log_loss"],
        "grouped_decisive_accuracy": grouped_report["decisive_accuracy"],
        "population_health": population_tracker.by_year,
        **population_summary,
        "mean_absolute_update": population_summary["absolute_updates"]["mean"],
        "p95_absolute_update": population_summary["absolute_updates"]["p95"],
        "max_absolute_update": population_summary["absolute_updates"]["max"],
        "elapsed_seconds": elapsed,
    }


def _print_search_result(completed, total, result):
    print(
        f"  [{completed}/{total}] "
        f"group={result['group_pruebas']}, "
        f"context={result['context_mode']}, "
        f"team={result['team_update_mode']}, "
        f"inactivity={result['inactivity_mode']}, "
        f"uncertainty_k={result['uncertainty_k_max_multiplier']:g}, "
        f"primary_delta={result['primary_switch_delta']}, "
        f"k={result['k_factor']:g}, "
        f"alpha={result['field_size_alpha']:g}, "
        f"field=({result['field_size_reference']:g}, "
        f"{result['field_size_min_multiplier']:g}, "
        f"{result['field_size_max_multiplier']:g}), "
        f"primary_lambda={result['primary_lambda']:g}, "
        f"modifier_lambda={result['modifier_lambda']:g}, "
        f"transfer={result['length_transfer_index']} -> "
        f"ungrouped=(log_loss={result['log_loss']:.6f}, "
        f"brier={result['brier_score']:.6f}, "
        f"accuracy={result['decisive_accuracy']:.6f}), "
        f"grouped=(log_loss={result['grouped_log_loss']:.6f}, "
        f"brier={result['grouped_brier_score']:.6f}, "
        f"accuracy={result['grouped_decisive_accuracy']:.6f}), "
        f"pruebas={result['pruebas_evaluated']}, "
        f"{result['elapsed_seconds']:.1f}s"
    )


def _grid_values_at_index(values, flat_index):
    """Decode one Cartesian-product index without materializing the product."""
    selected = [None] * len(values)
    for position in range(len(values) - 1, -1, -1):
        flat_index, value_index = divmod(flat_index, len(values[position]))
        selected[position] = (value_index, values[position][value_index])
    return selected


def hyperparam_search(groups, param_grid=None, n_jobs=1, n_experiments=1000):
    """
    Evaluate a reproducible random sample of the hyperparameter grid.

    Every candidate starts with fresh athlete state. Ranking uses strict
    prueba-level log loss, then Brier score, then decisive accuracy. Grouped
    metrics are reported alongside them but do not change the ranking. The
    module-level rating parameters are restored even if a run fails. If
    ``n_experiments`` is larger than the grid, every combination is evaluated.
    """
    grid = HYPERPARAM_GRID if param_grid is None else param_grid
    required = (
        "group_pruebas",
        "context_mode",
        "team_update_mode",
        "inactivity_mode",
        "inactivity_grace_days",
        "evidence_half_life_days",
        "uncertainty_k_max_multiplier",
        "primary_switch_delta",
        "k_factor",
        "field_size_alpha",
        "primary_lambda",
        "modifier_lambda",
        "field_size_reference",
        "field_size_min_multiplier",
        "field_size_max_multiplier",
        "length_transfer",
    )
    missing = [name for name in required if name not in grid]
    if missing:
        raise ValueError(f"Missing hyperparameter grid entries: {missing}")

    values = [tuple(grid[name]) for name in required]
    if any(not candidates for candidates in values):
        raise ValueError("Every hyperparameter must have at least one candidate")
    if n_experiments < 1:
        raise ValueError("n_experiments must be a positive integer")

    total_combinations = math.prod(len(candidates) for candidates in values)
    experiment_count = min(n_experiments, total_combinations)
    sampled_indices = random.Random(HYPERPARAM_RANDOM_SEED).sample(
        range(total_combinations), experiment_count
    )

    candidates = []
    for flat_index in sampled_indices:
        selected = _grid_values_at_index(values, flat_index)
        selected_indices = [value_index for value_index, _ in selected]
        (
            group_pruebas,
            context_mode,
            team_update_mode,
            inactivity_mode,
            inactivity_grace_days,
            evidence_half_life_days,
            uncertainty_k_max_multiplier,
            primary_switch_delta,
            k_factor,
            field_size_alpha,
            primary_lambda,
            modifier_lambda,
            field_size_reference,
            field_size_min_multiplier,
            field_size_max_multiplier,
            length_transfer,
        ) = [value for _, value in selected]
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"Unsupported context_mode candidate: {context_mode}")
        if team_update_mode not in TEAM_UPDATE_MODES:
            raise ValueError(
                f"Unsupported team_update_mode candidate: {team_update_mode}"
            )
        if inactivity_mode not in INACTIVITY_MODES:
            raise ValueError(
                f"Unsupported inactivity_mode candidate: {inactivity_mode}"
            )
        if inactivity_grace_days < 0:
            raise ValueError("Inactivity grace candidates must be non-negative")
        if evidence_half_life_days <= 0:
            raise ValueError("Evidence half-life candidates must be positive")
        if uncertainty_k_max_multiplier < 1:
            raise ValueError("Uncertainty K multiplier candidates must be at least 1")
        if primary_switch_delta < 0:
            raise ValueError("Primary switch delta candidates must be non-negative")
        if min(primary_lambda, modifier_lambda) <= 0:
            raise ValueError("All lambda candidates must be positive")
        if field_size_reference <= 0:
            raise ValueError("All field-size reference candidates must be positive")
        if (
            field_size_min_multiplier <= 0
            or field_size_max_multiplier < field_size_min_multiplier
        ):
            raise ValueError("Field-size multiplier candidate bounds are invalid")
        candidates.append(
            {
                "k_mode": "field_size_scaled",
                "group_pruebas": bool(group_pruebas),
                "context_mode": str(context_mode),
                "team_update_mode": str(team_update_mode),
                "inactivity_mode": str(inactivity_mode),
                "inactivity_grace_days": float(inactivity_grace_days),
                "evidence_half_life_days": float(evidence_half_life_days),
                "uncertainty_k_max_multiplier": float(uncertainty_k_max_multiplier),
                "primary_switch_delta": int(primary_switch_delta),
                "k_factor": float(k_factor),
                "field_size_reference": float(field_size_reference),
                "field_size_alpha": float(field_size_alpha),
                "field_size_min_multiplier": float(field_size_min_multiplier),
                "field_size_max_multiplier": float(field_size_max_multiplier),
                "primary_lambda": float(primary_lambda),
                "modifier_lambda": float(modifier_lambda),
                "length_transfer_index": selected_indices[-1],
                "length_transfer": length_transfer,
            }
        )

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    n_jobs = min(n_jobs, len(candidates))

    print(
        f"Randomly selected {len(candidates)} of {total_combinations} "
        f"hyperparameter configurations (seed={HYPERPARAM_RANDOM_SEED}) "
        f"with n_jobs={n_jobs}..."
    )
    expected_support = _evaluation_support_signature(
        calculate_evaluation_support(groups[True])
    )
    print(
        "Canonical evaluation support: "
        f"strict={expected_support[0]} pruebas/{expected_support[1]} pairs, "
        f"grouped={expected_support[4]} units/{expected_support[5]} pairs"
    )
    results = []
    if n_jobs == 1:
        _init_search_worker(groups, expected_support)
        for completed, candidate in enumerate(candidates, start=1):
            result = _evaluate_hyperparam_candidate(candidate)
            results.append(result)
            _print_search_result(completed, len(candidates), result)
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_search_worker,
            initargs=(groups, expected_support),
        ) as executor:
            futures = {
                executor.submit(_evaluate_hyperparam_candidate, candidate): candidate
                for candidate in candidates
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                _print_search_result(completed, len(candidates), result)

    results.sort(
        key=lambda result: (
            result["log_loss"],
            result["brier_score"],
            -result["decisive_accuracy"],
        )
    )
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    print("\nHyperparameter ranking:")
    print(
        f"{'Rank':>4} {'Group':>5} {'Context':>11} {'Team':>12} "
        f"{'Inactivity':>14} {'UK':>4} {'PΔ':>4} "
        f"{'K':>6} {'Alpha':>6} {'Ref':>6} {'Min':>5} {'Max':>5} "
        f"{'Primary':>8} {'Modifier':>8} "
        f"{'Transfer':>8} "
        f"{'U LogLoss':>10} {'U Brier':>10} {'U Acc':>10} "
        f"{'G LogLoss':>10} {'G Brier':>10} {'G Acc':>10} "
        f"{'PopMean':>9} {'PopStd':>9} {'Drift':>9}"
    )
    for result in results:
        population_mean = result["final_rating_mean"]
        population_std = result["final_rating_std"]
        cohort_drift = result["max_abs_cohort_median_drift"]
        print(
            f"{result['rank']:>4} {str(result['group_pruebas']):>5} "
            f"{result['context_mode']:>11} {result['team_update_mode']:>12} "
            f"{result['inactivity_mode']:>14} "
            f"{result['uncertainty_k_max_multiplier']:>4.1f} "
            f"{result['primary_switch_delta']:>4} "
            f"{result['k_factor']:>6.1f} "
            f"{result['field_size_alpha']:>6.2f} "
            f"{result['field_size_reference']:>6.1f} "
            f"{result['field_size_min_multiplier']:>5.2f} "
            f"{result['field_size_max_multiplier']:>5.2f} "
            f"{result['primary_lambda']:>8.1f} "
            f"{result['modifier_lambda']:>8.1f} "
            f"{result['length_transfer_index']:>8} "
            f"{result['log_loss']:>10.6f} {result['brier_score']:>10.6f} "
            f"{result['decisive_accuracy']:>10.6f} "
            f"{result['grouped_log_loss']:>10.6f} "
            f"{result['grouped_brier_score']:>10.6f} "
            f"{result['grouped_decisive_accuracy']:>10.6f} "
            f"{population_mean if population_mean is not None else float('nan'):>9.1f} "
            f"{population_std if population_std is not None else float('nan'):>9.1f} "
            f"{cohort_drift if cohort_drift is not None else float('nan'):>9.1f}"
        )

    return results


if __name__ == "__main__":
    main()
