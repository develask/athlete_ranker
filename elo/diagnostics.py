import csv
import json
import sys
from collections import Counter, defaultdict

from performance_groups import prepare_performance_groups


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def entry_signature(entry):
    """
    Signature useful for identifying likely duplicate listings.
    """
    return (
        tuple(entry.athlete_ids),
        entry.time_seconds,
        entry.prueba_id,
        entry.prueba_serie,
        entry.sexo,
        entry.categoria,
    )


def crew_signature(entry):
    return tuple(sorted(entry.athlete_ids))


def analyze_overlap_group(group):
    athlete_to_entries = defaultdict(list)

    for entry_index, entry in enumerate(group.entries):
        for athlete_id in entry.athlete_ids:
            athlete_to_entries[athlete_id].append(
                (entry_index, entry)
            )

    repeated = {
        athlete_id: appearances
        for athlete_id, appearances
        in athlete_to_entries.items()
        if len(appearances) > 1
    }

    rows = []
    group_summary = Counter()

    for athlete_id, appearances in repeated.items():
        n = len(appearances)

        times = [
            entry.time_seconds
            for _, entry in appearances
        ]
        prueba_ids = [
            entry.prueba_id
            for _, entry in appearances
        ]
        series = [
            entry.prueba_serie
            for _, entry in appearances
        ]
        sexes = [
            entry.sexo
            for _, entry in appearances
        ]
        categories = [
            entry.categoria
            for _, entry in appearances
        ]
        crews = [
            crew_signature(entry)
            for _, entry in appearances
        ]

        same_time = len(set(times)) == 1
        same_prueba = len(set(prueba_ids)) == 1
        same_series = len(set(series)) == 1
        same_sex = len(set(sexes)) == 1
        same_category = len(set(categories)) == 1
        same_crew = len(set(crews)) == 1

        # Likely duplicated classification/listing:
        # same crew and exactly same performance time,
        # but appears in more than one source entry.
        likely_duplicate_listing = (
            same_time
            and same_crew
        )

        # Likely genuine separate performances:
        # same paddler appears with different times and/or crews.
        likely_genuine_multiple = (
            not same_time
            or not same_crew
        )

        if likely_duplicate_listing:
            group_summary[
                "likely_duplicate_athletes"
            ] += 1

        if likely_genuine_multiple:
            group_summary[
                "likely_genuine_multiple_athletes"
            ] += 1

        group_summary[
            f"appearance_count_{n}"
        ] += 1

        row = {
            "regata_id": group.regata_id,
            "fecha": group.fecha,
            "fase": group.prueba_fase,
            "tipo": group.tipo,
            "boat": group.embarcacion_tipo,
            "boat_num": group.embarcacion_num,
            "distance": group.distancia_exacta,
            "length_class": group.length_class,
            "athlete_id": athlete_id,
            "n_appearances": n,
            "same_time": same_time,
            "same_crew": same_crew,
            "same_prueba": same_prueba,
            "same_series": same_series,
            "same_sex": same_sex,
            "same_category": same_category,
            "likely_duplicate_listing": (
                likely_duplicate_listing
            ),
            "likely_genuine_multiple": (
                likely_genuine_multiple
            ),
            "times": " | ".join(
                str(value)
                for value in times
            ),
            "prueba_ids": " | ".join(
                str(value)
                for value in prueba_ids
            ),
            "series": " | ".join(
                str(value)
                for value in series
            ),
            "sexes": " | ".join(
                str(value)
                for value in sexes
            ),
            "categories": " | ".join(
                str(value)
                for value in categories
            ),
            "crews": " | ".join(
                ",".join(crew)
                for crew in crews
            ),
        }

        rows.append(row)

    return repeated, rows, group_summary


def main(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    groups, stats = prepare_performance_groups(
        raw_data,
        keep_unusable=True,
    )

    usable_groups = [
        group
        for group in groups
        if group.usable
    ]

    overlap_groups = [
        group
        for group in usable_groups
        if group.overlapping_athlete_ids
    ]

    field_sizes = [
        len(group.entries)
        for group in usable_groups
    ]

    source_prueba_counts = [
        group.n_source_pruebas
        for group in usable_groups
    ]

    overlap_by_boat_num = Counter()
    overlap_by_tipo = Counter()
    overlap_by_length = Counter()
    overlap_by_phase = Counter()

    overlap_appearance_distribution = Counter()

    total_overlap_athlete_ids = 0
    total_overlap_appearances = 0

    likely_duplicate_athletes = 0
    likely_genuine_multiple_athletes = 0

    overlap_rows = []

    groups_all_overlap_likely_duplicates = 0
    groups_with_any_genuine_multiple = 0

    for group in overlap_groups:
        overlap_by_boat_num[
            group.embarcacion_num
        ] += 1

        overlap_by_tipo[
            group.tipo
        ] += 1

        overlap_by_length[
            group.length_class
        ] += 1

        overlap_by_phase[
            group.prueba_fase
        ] += 1

        repeated, rows, group_summary = (
            analyze_overlap_group(group)
        )

        overlap_rows.extend(rows)

        total_overlap_athlete_ids += len(repeated)

        group_has_genuine = False
        group_all_duplicate = True

        for athlete_id, appearances in (
            repeated.items()
        ):
            n = len(appearances)

            overlap_appearance_distribution[n] += 1
            total_overlap_appearances += n

        for row in rows:
            if row["likely_duplicate_listing"]:
                likely_duplicate_athletes += 1
            else:
                group_all_duplicate = False

            if row["likely_genuine_multiple"]:
                likely_genuine_multiple_athletes += 1
                group_has_genuine = True

        if group_all_duplicate:
            groups_all_overlap_likely_duplicates += 1

        if group_has_genuine:
            groups_with_any_genuine_multiple += 1

    merged_abc_groups = 0
    multi_series_groups = 0

    for group in usable_groups:
        normalized_series = {
            str(value).strip().casefold()
            for value in group.source_series
            if value is not None
        }

        if len(normalized_series) > 1:
            multi_series_groups += 1

        if (
            "final a" in normalized_series
            and (
                "final b" in normalized_series
                or "final c" in normalized_series
            )
        ):
            merged_abc_groups += 1

    # -------------------------------------------------------------
    # Write detailed overlap CSV
    # -------------------------------------------------------------
    csv_path = "overlap_audit.csv"

    if overlap_rows:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    overlap_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(overlap_rows)

    # -------------------------------------------------------------
    # Print diagnostics
    # -------------------------------------------------------------
    print("\n=== CORE COUNTS ===")
    print(
        "regattas_seen:",
        stats["regattas_seen"],
    )
    print(
        "pruebas_seen:",
        stats["pruebas_seen"],
    )
    print(
        "raw_results_seen:",
        stats["raw_results_seen"],
    )
    print(
        "raw_groups_built:",
        stats["raw_groups_built"],
    )
    print(
        "usable_groups:",
        len(usable_groups),
    )
    print(
        "unusable_groups:",
        len(groups) - len(usable_groups),
    )

    print("\n=== TIME CLEANING ===")
    for key in (
        "discard_time_unparseable",
        "discard_time_below_min",
        "discard_time_above_max",
    ):
        print(key + ":", stats[key])

    print("\n=== GROUP MERGING ===")
    print(
        "cross_prueba_groups:",
        stats["groups_cross_prueba"],
    )
    print(
        "multi_series_groups:",
        multi_series_groups,
    )
    print(
        "merged_final_abc_groups:",
        merged_abc_groups,
    )
    print(
        "multiple_sexes:",
        stats["groups_multiple_sexes"],
    )
    print(
        "multiple_categories:",
        stats["groups_multiple_categories"],
    )

    print("\n=== OVERLAPS ===")
    print(
        "usable_groups_with_overlap:",
        len(overlap_groups),
    )
    print(
        "share_of_usable_groups:",
        (
            len(overlap_groups)
            / len(usable_groups)
            if usable_groups
            else 0
        ),
    )
    print(
        "unique_overlap_athlete_occurrences_by_group:",
        total_overlap_athlete_ids,
    )
    print(
        "total_entry_appearances_of_repeated_athletes:",
        total_overlap_appearances,
    )
    print(
        "appearance_count_distribution:",
        dict(
            sorted(
                overlap_appearance_distribution.items()
            )
        ),
    )

    print(
        "likely_duplicate_listing_athletes:",
        likely_duplicate_athletes,
    )
    print(
        "likely_genuine_multiple_athletes:",
        likely_genuine_multiple_athletes,
    )

    print(
        "groups_all_overlaps_look_like_duplicates:",
        groups_all_overlap_likely_duplicates,
    )
    print(
        "groups_with_any_genuine_multiple:",
        groups_with_any_genuine_multiple,
    )

    print(
        "overlap_by_boat_num:",
        dict(
            overlap_by_boat_num.most_common()
        ),
    )
    print(
        "overlap_by_tipo:",
        dict(
            overlap_by_tipo.most_common()
        ),
    )
    print(
        "overlap_by_length:",
        dict(
            overlap_by_length.most_common()
        ),
    )
    print(
        "overlap_by_phase:",
        dict(
            overlap_by_phase.most_common()
        ),
    )

    print("\n=== TEAM VALIDATION ===")
    print(
        "crew_size_mismatch_entries:",
        stats["crew_size_mismatch"],
    )

    print("\n=== FIELD SIZE ===")
    if field_sizes:
        print("min:", min(field_sizes))
        print(
            "p25:",
            percentile(field_sizes, 0.25),
        )
        print(
            "median:",
            percentile(field_sizes, 0.50),
        )
        print(
            "p75:",
            percentile(field_sizes, 0.75),
        )
        print(
            "p90:",
            percentile(field_sizes, 0.90),
        )
        print(
            "p95:",
            percentile(field_sizes, 0.95),
        )
        print("max:", max(field_sizes))
        print(
            "mean:",
            sum(field_sizes) / len(field_sizes),
        )

    print("\n=== SOURCE PRUEBAS PER GROUP ===")
    if source_prueba_counts:
        print(
            "mean:",
            sum(source_prueba_counts)
            / len(source_prueba_counts),
        )
        print(
            "max:",
            max(source_prueba_counts),
        )

    print("\n=== OVERLAP CSV ===")
    print(
        "Detailed overlap audit written to:",
        csv_path,
    )

    print(
        "\nPlease send me the OVERLAPS section "
        "from this output. The CSV is only needed "
        "if we want to inspect specific examples."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python diagnostics_v2.py dataset.json"
        )

    main(sys.argv[1])