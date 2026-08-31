import argparse
import csv
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from time import perf_counter

import rating_engine
from performance_groups import prepare_performance_groups
from rating_engine import (
    FIELD_SIZE_ALPHA,
    FIELD_SIZE_MAX_MULTIPLIER,
    FIELD_SIZE_MIN_MULTIPLIER,
    FIELD_SIZE_REFERENCE,
    K_FACTOR,
    LENGTH_CLASSES,
    LENGTH_TRANSFER,
    MODIFIER_LAMBDA,
    PRIMARY_LAMBDA,
    athlete_debug_dict,
    process_performance_group,
    run_rating_system,
)

STRONG_LENGTH_TRANSFER = {
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

STRONG_LENGTH_TRANSFER_2 = {
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

STRONG_LENGTH_TRANSFER_3 = {
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

HYPERPARAM_GRID = {
    "k_factor": (400.0, 500.0, 600.0, 700.0),
    "field_size_alpha": (0.0, 0.10, 0.20, 0.25, 0.33, 0.50),
    "primary_lambda": (15.0, 20.0),
    "modifier_lambda": (5.0, 10.0),
    "field_size_reference": (10.0, 20.0, 40.0),
    "field_size_min_multiplier": (0.50, 0.70, 0.85),
    "field_size_max_multiplier": (1.40, 1.75, 2.25),
    "length_transfer": (
        # LENGTH_TRANSFER,
        # STRONG_LENGTH_TRANSFER,
        STRONG_LENGTH_TRANSFER_2,
        STRONG_LENGTH_TRANSFER_3,
    ),
}

_SEARCH_GROUPS = None


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
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": safe_mean(values),
    }


def serialize_counter(counter):
    return {str(key): value for key, value in counter.items()}


def year_from_fecha(fecha):
    year = str(fecha or "")[:4]
    return year if year.isdigit() else "unknown"


def write_final_json(athletes, path):
    payload = {
        athlete_id: (athlete_debug_dict(athlete))
        for athlete_id, athlete in sorted(athletes.items(), key=lambda item: item[0])
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_final_csv(athletes, path):
    fieldnames = ["athlete_id", "primary_tipo", "primary_boat"]

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
            "tipo_race_counts_json",
            "boat_race_counts_json",
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

            row["tipo_race_counts_json"] = json.dumps(
                dict(athlete.tipo_race_counts), ensure_ascii=False, sort_keys=True
            )

            row["boat_race_counts_json"] = json.dumps(
                dict(athlete.boat_race_counts), ensure_ascii=False, sort_keys=True
            )

            writer.writerow(row)


def build_rating_distributions(athletes):
    output = {}

    for length_class in LENGTH_CLASSES:
        all_values = []
        direct_values = []

        for athlete in athletes.values():
            rating = athlete.length_ratings[length_class]
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

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", dataset_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Building performance groups...")

    groups, prep_stats = prepare_performance_groups(raw_data, keep_unusable=True)

    if args.hyperparam_search:
        results = hyperparam_search(groups, n_jobs=args.n_jobs)
        json_path = output_dir / "hyperparam_search.json"
        csv_path = output_dir / "hyperparam_search.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        csv_fields = [
            "rank",
            "k_mode",
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

    athletes = {}
    rating_stats = Counter()
    evaluation = Counter()
    evaluation_by_year = {}
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

        for index, group in enumerate(groups, start=1):
            group_year = year_from_fecha(group.fecha)
            if current_year is None:
                current_year = group_year
            elif group_year != current_year:
                rating_distributions_by_year[current_year] = build_rating_distributions(
                    athletes
                )
                current_year = group_year

            result = process_performance_group(
                athletes=athletes,
                group=group,
                k=args.k,
                field_size_reference=args.field_size_reference,
                field_size_alpha=args.field_size_alpha,
                field_size_min_multiplier=args.field_size_min_multiplier,
                field_size_max_multiplier=args.field_size_max_multiplier,
            )

            if result.processed:
                rating_stats["groups_processed"] += 1
                rating_stats["entries_processed"] += len(group.entries)
                rating_stats["athlete_group_updates"] += len(result.athlete_updates)
                rating_stats["athletes_with_multiple_appearances"] += result.stats[
                    "athletes_with_multiple_appearances"
                ]

                for key, value in result.evaluation.items():
                    evaluation[key] += value

                if result.evaluation["pairwise_comparisons"]:
                    yearly_evaluation = evaluation_by_year.setdefault(
                        group_year, Counter()
                    )
                    yearly_evaluation.update(result.evaluation)
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
                    "remaining_overlap_athletes": (len(group.overlapping_athlete_ids)),
                    "near_duplicate_rows_removed": (
                        group.cleaning_stats["discard_near_duplicate_crew_entry"]
                    ),
                    "crew_size_mismatch_removed": (
                        group.cleaning_stats["discard_crew_size_mismatch"]
                    ),
                    "processed": (result.processed),
                    "skip_reason": (result.skip_reason),
                    "mean_abs_team_delta": (mean_abs_delta),
                    "pairwise_comparisons": (result.evaluation["pairwise_comparisons"]),
                }
            )

            if index % 500 == 0:
                print(f"  processed {index}/" f"{len(groups)} groups")

    if current_year is not None:
        rating_distributions_by_year[current_year] = build_rating_distributions(
            athletes
        )

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
        "prediction_evaluation": (evaluation_summary(evaluation)),
        "prediction_evaluation_by_year": {
            year: evaluation_summary(yearly_metrics)
            for year, yearly_metrics in sorted(evaluation_by_year.items())
        },
        "rating_distributions": (build_rating_distributions(athletes)),
        "rating_distributions_by_year": {
            year: values
            for year, values in sorted(rating_distributions_by_year.items())
        },
        "modifier_distributions": (build_modifier_distributions(athletes)),
    }

    summary_path = output_dir / "run_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Athletes:", len(athletes))
    print("Groups processed:", rating_stats["groups_processed"])
    print("Groups skipped:", rating_stats["groups_skipped"])

    evaluation_report = summary["prediction_evaluation"]

    def metric_text(value):
        return f"{value:.6f}" if value is not None else "n/a"

    def elo_text(value):
        return f"{value:.1f}" if value is not None else "n/a"

    print("\nPrediction evaluation by year:")
    print(
        f"{'Year':<8} {'Pruebas':>8} "
        f"{'Pairs':>12} {'Decisive':>12} "
        f"{'Brier':>10} {'Log loss':>10} "
        f"{'Accuracy':>10}"
    )
    for year, report in summary["prediction_evaluation_by_year"].items():
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
    print("Pairwise Brier:", metric_text(evaluation_report["brier_score"]))
    print("Pairwise log loss:", metric_text(evaluation_report["log_loss"]))
    print("Decisive accuracy:", metric_text(evaluation_report["decisive_accuracy"]))

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


def _init_search_worker(groups):
    global _SEARCH_GROUPS
    _SEARCH_GROUPS = groups


def _evaluate_hyperparam_candidate(candidate):
    if _SEARCH_GROUPS is None:
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
        _, _, evaluation, _ = run_rating_system(
            _SEARCH_GROUPS,
            k=candidate["k_factor"],
            field_size_reference=candidate["field_size_reference"],
            field_size_alpha=candidate["field_size_alpha"],
            field_size_min_multiplier=candidate["field_size_min_multiplier"],
            field_size_max_multiplier=candidate["field_size_max_multiplier"],
        )
        report = evaluation_summary(evaluation)
        elapsed = perf_counter() - started
    finally:
        rating_engine.PRIMARY_LAMBDA = original_parameters["primary_lambda"]
        rating_engine.MODIFIER_LAMBDA = original_parameters["modifier_lambda"]
        rating_engine.LENGTH_TRANSFER = original_parameters["length_transfer"]
        rating_engine.CONFIDENCE_TRANSFER = original_parameters["confidence_transfer"]

    return {
        "rank": None,
        **candidate,
        "pruebas_evaluated": report["pruebas_evaluated"],
        "pairwise_comparisons": report["pairwise_comparisons"],
        "pairwise_decisive": report["pairwise_decisive"],
        "brier_score": report["brier_score"],
        "log_loss": report["log_loss"],
        "decisive_accuracy": report["decisive_accuracy"],
        "elapsed_seconds": elapsed,
    }


def _print_search_result(completed, total, result):
    print(
        f"  [{completed}/{total}] "
        f"k={result['k_factor']:g}, "
        f"alpha={result['field_size_alpha']:g}, "
        f"field=({result['field_size_reference']:g}, "
        f"{result['field_size_min_multiplier']:g}, "
        f"{result['field_size_max_multiplier']:g}), "
        f"primary_lambda={result['primary_lambda']:g}, "
        f"modifier_lambda={result['modifier_lambda']:g}, "
        f"transfer={result['length_transfer_index']} -> "
        f"log_loss={result['log_loss']:.6f}, "
        f"brier={result['brier_score']:.6f}, "
        f"accuracy={result['decisive_accuracy']:.6f}, "
        f"pruebas={result['pruebas_evaluated']}, "
        f"{result['elapsed_seconds']:.1f}s"
    )


def hyperparam_search(groups, param_grid=None, n_jobs=1):
    """
    Exhaustively evaluate Elo hyperparameters over the full chronology.

    Every candidate starts with fresh athlete state. Ranking uses strict
    prueba-level log loss, then Brier score, then decisive accuracy. The
    module-level rating parameters are restored even if a run fails.
    """
    grid = HYPERPARAM_GRID if param_grid is None else param_grid
    required = (
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

    transfer_candidates = values[-1]
    candidates = []
    for (
        k_factor,
        field_size_alpha,
        primary_lambda,
        modifier_lambda,
        field_size_reference,
        field_size_min_multiplier,
        field_size_max_multiplier,
        length_transfer,
    ) in product(*values):
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
                "k_factor": float(k_factor),
                "field_size_reference": float(field_size_reference),
                "field_size_alpha": float(field_size_alpha),
                "field_size_min_multiplier": float(field_size_min_multiplier),
                "field_size_max_multiplier": float(field_size_max_multiplier),
                "primary_lambda": float(primary_lambda),
                "modifier_lambda": float(modifier_lambda),
                "length_transfer_index": transfer_candidates.index(length_transfer),
                "length_transfer": length_transfer,
            }
        )

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    n_jobs = min(n_jobs, len(candidates))

    print(
        f"Running {len(candidates)} hyperparameter configurations "
        f"with n_jobs={n_jobs}..."
    )
    results = []
    if n_jobs == 1:
        _init_search_worker(groups)
        for completed, candidate in enumerate(candidates, start=1):
            result = _evaluate_hyperparam_candidate(candidate)
            results.append(result)
            _print_search_result(completed, len(candidates), result)
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_search_worker,
            initargs=(groups,),
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
        f"{'Rank':>4} {'K':>6} {'Alpha':>6} {'Ref':>6} {'Min':>5} {'Max':>5} "
        f"{'Primary':>8} {'Modifier':>8} "
        f"{'Transfer':>8} {'Log loss':>10} {'Brier':>10} {'Accuracy':>10}"
    )
    for result in results:
        print(
            f"{result['rank']:>4} {result['k_factor']:>6.1f} "
            f"{result['field_size_alpha']:>6.2f} "
            f"{result['field_size_reference']:>6.1f} "
            f"{result['field_size_min_multiplier']:>5.2f} "
            f"{result['field_size_max_multiplier']:>5.2f} "
            f"{result['primary_lambda']:>8.1f} "
            f"{result['modifier_lambda']:>8.1f} "
            f"{result['length_transfer_index']:>8} "
            f"{result['log_loss']:>10.6f} {result['brier_score']:>10.6f} "
            f"{result['decisive_accuracy']:>10.6f}"
        )

    return results


if __name__ == "__main__":
    main()
