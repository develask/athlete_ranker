import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from performance_groups import prepare_performance_groups
from rating_engine import (
    K_FACTOR,
    LAMBDA_BOAT,
    LAMBDA_LENGTH,
    LAMBDA_TIPO,
    LENGTH_CLASSES,
    LENGTH_TRANSFER,
    athlete_debug_dict,
    process_performance_group,
)


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

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", dataset_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("Building performance groups...")

    groups, prep_stats = prepare_performance_groups(raw_data, keep_unusable=True)

    athletes = {}
    rating_stats = Counter()
    evaluation = Counter()
    evaluation_by_year = {}
    evaluation_group_support_by_year = Counter()
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

            result = process_performance_group(athletes=athletes, group=group, k=args.k)

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
                    evaluation_group_support_by_year[group_year] += 1
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
            "k_factor": args.k,
            "lambda_length": (LAMBDA_LENGTH),
            "lambda_tipo": (LAMBDA_TIPO),
            "lambda_boat": (LAMBDA_BOAT),
            "length_transfer": (LENGTH_TRANSFER),
        },
        "preprocessing": (serialize_counter(prep_stats)),
        "rating_run": (serialize_counter(rating_stats)),
        "prediction_evaluation": (evaluation_summary(evaluation)),
        "prediction_evaluation_by_year": {
            year: {
                "prueba_group_support": (evaluation_group_support_by_year[year]),
                **evaluation_summary(yearly_metrics),
            }
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
        f"{'Year':<8} {'Groups':>8} "
        f"{'Pairs':>12} {'Decisive':>12} "
        f"{'Brier':>10} {'Log loss':>10} "
        f"{'Accuracy':>10}"
    )
    for year, report in summary["prediction_evaluation_by_year"].items():
        print(
            f"{year:<8} "
            f"{report['prueba_group_support']:>8} "
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


if __name__ == "__main__":
    main()
