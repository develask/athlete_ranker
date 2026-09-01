import math
import re
from collections import Counter, OrderedDict
from copy import copy
from datetime import datetime

MIN_VALID_TIME_SECONDS = 10.0
MAX_VALID_TIME_SECONDS = 3 * 60 * 60
DUPLICATE_TIME_MARGIN_SECONDS = 0.5


def normalize_text(value):
    """
    Grouping normalization only:
    - strip
    - collapse repeated whitespace
    - casefold

    Deliberately does NOT map semantic variants such as
    "Final" -> "Finales".
    """
    if value is None:
        return None

    text = " ".join(str(value).split())
    if not text:
        return None

    return text.casefold()


def normalize_intlike(value):
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if number.is_integer():
        return int(number)

    return number


def normalize_distance(value):
    return normalize_intlike(value)


def normalize_athlete_ids(value):
    """
    Return a tuple of athlete IDs, or None when unusable.
    """
    if value is None:
        return None

    if isinstance(value, (str, int)):
        value = [value]

    if not isinstance(value, (list, tuple)):
        return None

    athlete_ids = []

    for athlete_id in value:
        if athlete_id is None:
            return None

        athlete_id = str(athlete_id).strip()

        if not athlete_id:
            return None

        athlete_ids.append(athlete_id)

    if not athlete_ids:
        return None

    # Same athlete listed twice inside one crew is malformed.
    if len(set(athlete_ids)) != len(athlete_ids):
        return None

    return tuple(athlete_ids)


_TIME_RE = re.compile(r"^\s*(\d{1,3}):(\d{1,2}):(\d{1,2}(?:[.,]\d+)?)\s*$")


def parse_time_seconds(value):
    """
    Parse HH:MM:SS[.ff]. Decimal comma is accepted.

    Numeric inputs are treated as already being seconds.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if math.isfinite(seconds) else None

    match = _TIME_RE.match(str(value))

    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3).replace(",", "."))

    if not (0 <= minutes < 60):
        return None

    if not (0 <= seconds < 60):
        return None

    return hours * 3600.0 + minutes * 60.0 + seconds


def classify_time(value):
    seconds = parse_time_seconds(value)

    if seconds is None:
        return "unparseable", None

    if seconds < MIN_VALID_TIME_SECONDS:
        return "below_min", seconds

    if seconds > MAX_VALID_TIME_SECONDS:
        return "above_max", seconds

    return "valid", seconds


def team_update_factor(team_size, mode="inverse_sqrt"):
    if team_size <= 0:
        raise ValueError("team_size must be positive")
    if mode == "inverse_sqrt":
        return 1.0 / math.sqrt(team_size)
    if mode == "none":
        return 1.0
    raise ValueError(f"Unsupported team_update_mode: {mode}")


def entries_share_athlete(entry_a, entry_b):
    return bool(set(entry_a.athlete_ids) & set(entry_b.athlete_ids))


class PerformanceEntry:
    def __init__(
        self,
        athlete_ids,
        raw_time,
        prueba_id,
        prueba_name,
        prueba_serie,
        sexo,
        categoria,
        original_position,
        source_prueba_index,
        source_result_index,
    ):
        self.athlete_ids_raw = athlete_ids
        self.raw_time = raw_time

        self.prueba_id = prueba_id
        self.prueba_name = prueba_name
        self.prueba_serie = prueba_serie
        self.sexo = sexo
        self.categoria = categoria

        self.original_position = original_position

        self.source_prueba_index = source_prueba_index
        self.source_result_index = source_result_index

        # Filled during cleaning.
        self.athlete_ids = None
        self.time_seconds = None
        self.time_status = None

        self.crew_size = None
        self.crew_size_mismatch = False

        self.merged_rank = None
        self.merged_order_index = None

        # Duplicate-classification metadata.
        self.deduplicated_count = 1
        self.deduplicated_times = ()
        self.deduplicated_prueba_ids = ()


class PerformanceGroup:
    def __init__(
        self,
        key,
        regata_id,
        fecha,
        prueba_fase,
        tipo,
        embarcacion_tipo,
        embarcacion_num,
        distancia_exacta,
        length_class,
        source_regatta_index,
        first_source_prueba_index,
        distance_is_known,
    ):
        self.key = key

        self.regata_id = regata_id
        self.fecha = fecha
        self.prueba_fase = prueba_fase

        self.tipo = tipo
        self.embarcacion_tipo = embarcacion_tipo
        self.embarcacion_num = embarcacion_num

        self.distancia_exacta = distancia_exacta
        self.length_class = length_class

        self.source_regatta_index = source_regatta_index
        self.first_source_prueba_index = first_source_prueba_index

        self.distance_is_known = distance_is_known

        self.entries = []
        self.evaluation_entries = []
        self.source_pruebas = []
        self.cleaning_stats = Counter()

        self.overlapping_athlete_ids = ()
        self.usable = False

    def add_source_prueba(self, prueba, prueba_index):
        self.source_pruebas.append(
            {
                "prueba_id": (prueba.get("prueba_id")),
                "prueba_name": (prueba.get("prueba_name")),
                "prueba_serie": (prueba.get("prueba_serie")),
                "sexo": prueba.get("sexo"),
                "categoria": (prueba.get("categoria")),
                "source_prueba_index": (prueba_index),
            }
        )

    @property
    def n_source_pruebas(self):
        return len(self.source_pruebas)

    @property
    def source_prueba_ids(self):
        return tuple(item["prueba_id"] for item in self.source_pruebas)

    @property
    def source_series(self):
        return tuple(
            sorted(
                {
                    item["prueba_serie"]
                    for item in self.source_pruebas
                    if (item["prueba_serie"] is not None)
                },
                key=lambda value: str(value),
            )
        )

    @property
    def source_sexes(self):
        return tuple(
            sorted(
                {
                    item["sexo"]
                    for item in self.source_pruebas
                    if item["sexo"] is not None
                },
                key=lambda value: str(value),
            )
        )

    @property
    def source_categories(self):
        return tuple(
            sorted(
                {
                    item["categoria"]
                    for item in self.source_pruebas
                    if (item["categoria"] is not None)
                },
                key=lambda value: str(value),
            )
        )

    @property
    def is_cross_prueba_merge(self):
        prueba_ids = {item["prueba_id"] for item in self.source_pruebas}

        return len(prueba_ids) > 1


def make_performance_group_key(
    regata_id,
    fecha,
    prueba,
    source_regatta_index,
    prueba_index,
    group_pruebas=True,
):
    phase = normalize_text(prueba.get("prueba_fase"))
    tipo = normalize_text(prueba.get("tipo"))
    boat = normalize_text(prueba.get("embarcacion_tipo"))
    boat_num = normalize_intlike(prueba.get("embarcacion_num"))
    length_class = normalize_text(prueba.get("length_class"))
    distance = normalize_distance(prueba.get("distancia_exacta"))

    # Required rating/grouping context.
    if (
        regata_id is None
        or fecha is None
        or phase is None
        or tipo is None
        or boat is None
        or boat_num is None
        or length_class is None
    ):
        return None, None

    prueba_id = prueba.get("prueba_id")
    if prueba_id is None:
        prueba_identity = (
            "source:" + str(source_regatta_index) + ":" + str(prueba_index)
        )
    else:
        prueba_identity = str(prueba_id)

    if distance is None:
        distance_key = ("unknown_distance_prueba", prueba_identity)
        distance_is_known = False
    elif not group_pruebas:
        distance_key = ("known_distance_prueba", distance, prueba_identity)
        distance_is_known = True
    else:
        distance_key = ("known_distance", distance)
        distance_is_known = True

    key = (
        str(regata_id),
        str(fecha),
        phase,
        tipo,
        boat,
        boat_num,
        distance_key,
        length_class,
    )

    context = {
        "phase": phase,
        "tipo": tipo,
        "boat": boat,
        "boat_num": boat_num,
        "distance": distance,
        "length_class": length_class,
        "distance_is_known": (distance_is_known),
    }

    return key, context


def build_performance_groups(raw_data, group_pruebas=True):
    groups_by_key = OrderedDict()
    stats = Counter()

    for regatta_index, regatta in enumerate(raw_data):
        stats["regattas_seen"] += 1

        regata_id = regatta.get("regata_id")
        fecha = regatta.get("fecha")

        pruebas = regatta.get("pruebas") or []

        for prueba_index, prueba in enumerate(pruebas):
            stats["pruebas_seen"] += 1

            results = prueba.get("results") or []

            stats["raw_results_seen"] += len(results)

            key, context = make_performance_group_key(
                regata_id=regata_id,
                fecha=fecha,
                prueba=prueba,
                source_regatta_index=(regatta_index),
                prueba_index=prueba_index,
                group_pruebas=group_pruebas,
            )

            if key is None:
                stats["pruebas_skipped_missing_context"] += 1
                continue

            if not context["distance_is_known"]:
                stats["pruebas_unknown_distance"] += 1

            group = groups_by_key.get(key)

            if group is None:
                group = PerformanceGroup(
                    key=key,
                    regata_id=str(regata_id),
                    fecha=str(fecha),
                    prueba_fase=context["phase"],
                    tipo=context["tipo"],
                    embarcacion_tipo=context["boat"],
                    embarcacion_num=context["boat_num"],
                    distancia_exacta=context["distance"],
                    length_class=context["length_class"],
                    source_regatta_index=(regatta_index),
                    first_source_prueba_index=(prueba_index),
                    distance_is_known=context["distance_is_known"],
                )

                groups_by_key[key] = group

            group.add_source_prueba(prueba, prueba_index)

            for result_index, result in enumerate(results):
                group.entries.append(
                    PerformanceEntry(
                        athlete_ids=(result.get("lista_palista_id")),
                        raw_time=result.get("tiempo"),
                        prueba_id=prueba.get("prueba_id"),
                        prueba_name=prueba.get("prueba_name"),
                        prueba_serie=prueba.get("prueba_serie"),
                        sexo=prueba.get("sexo"),
                        categoria=prueba.get("categoria"),
                        original_position=(result.get("posicion")),
                        source_prueba_index=(prueba_index),
                        source_result_index=(result_index),
                    )
                )

    stats["raw_groups_built"] = len(groups_by_key)

    return list(groups_by_key.values()), stats


def _crew_signature(entry):
    # Crew order should not affect duplicate
    # detection.
    return tuple(sorted(entry.athlete_ids))


def _collapse_near_duplicate_entries(
    entries, margin_seconds=(DUPLICATE_TIME_MARGIN_SECONDS)
):
    """
    Collapse same-crew repeated listings whose
    times are no more than margin_seconds apart.

    A cluster is anchored at its fastest time,
    so every retained member is within the
    requested margin of the cluster start.
    The representative time is the arithmetic
    mean of the clustered times.
    """
    by_crew = OrderedDict()

    for entry in entries:
        by_crew.setdefault(_crew_signature(entry), []).append(entry)

    output = []
    duplicates_removed = 0
    duplicate_clusters = 0

    for crew_entries in by_crew.values():
        crew_entries.sort(
            key=lambda entry: (
                entry.time_seconds,
                entry.source_prueba_index,
                entry.source_result_index,
            )
        )

        clusters = []
        current = []

        for entry in crew_entries:
            if not current:
                current = [entry]
                continue

            cluster_start = current[0].time_seconds

            if entry.time_seconds - cluster_start <= margin_seconds:
                current.append(entry)
            else:
                clusters.append(current)
                current = [entry]

        if current:
            clusters.append(current)

        for cluster in clusters:
            representative = cluster[0]

            if len(cluster) > 1:
                times = tuple(item.time_seconds for item in cluster)
                prueba_ids = tuple(item.prueba_id for item in cluster)

                representative.time_seconds = sum(times) / len(times)
                representative.deduplicated_count = len(cluster)
                representative.deduplicated_times = times
                representative.deduplicated_prueba_ids = prueba_ids

                duplicates_removed += len(cluster) - 1
                duplicate_clusters += 1

            output.append(representative)

    output.sort(
        key=lambda entry: (
            entry.time_seconds,
            entry.source_prueba_index,
            entry.source_result_index,
        )
    )

    return (output, duplicates_removed, duplicate_clusters)


def clean_performance_group(group):
    """
    Clean one group in place.

    V1:
    - annotations ignored
    - time only
    - 10 <= time <= 10800 seconds
    - malformed athlete lists discarded
    - crew-size mismatches discarded
    - same-crew duplicate classifications
      within 0.5 seconds collapsed
    - duplicate cluster time = mean
    - rank by cleaned time
    """
    stats = Counter()
    valid_entries = []

    expected_crew_size = normalize_intlike(group.embarcacion_num)

    for entry in group.entries:
        athlete_ids = normalize_athlete_ids(entry.athlete_ids_raw)

        if athlete_ids is None:
            stats["discard_invalid_athlete_ids"] += 1
            continue

        status, seconds = classify_time(entry.raw_time)
        entry.time_status = status

        if status != "valid":
            stats["discard_time_" + status] += 1
            continue

        entry.athlete_ids = athlete_ids
        entry.time_seconds = seconds
        entry.crew_size = len(athlete_ids)

        if (
            isinstance(expected_crew_size, int)
            and expected_crew_size > 0
            and (entry.crew_size != expected_crew_size)
        ):
            entry.crew_size_mismatch = True
            stats["discard_crew_size_mismatch"] += 1
            continue

        valid_entries.append(entry)

    # Evaluation remains strictly within each original prueba. Copies are
    # required because merged-group deduplication below mutates representative
    # entries and may collapse the same crew across different pruebas.
    evaluation_entries = []
    entries_by_prueba = OrderedDict()
    for entry in valid_entries:
        entries_by_prueba.setdefault(entry.source_prueba_index, []).append(copy(entry))

    for prueba_entries in entries_by_prueba.values():
        cleaned_entries, _, _ = _collapse_near_duplicate_entries(prueba_entries)
        evaluation_entries.extend(cleaned_entries)

    valid_entries, duplicates_removed, duplicate_clusters = (
        _collapse_near_duplicate_entries(valid_entries)
    )

    stats["discard_near_duplicate_crew_entry"] += duplicates_removed
    stats["near_duplicate_crew_clusters"] += duplicate_clusters

    previous_time = None
    current_rank = None

    for index, entry in enumerate(valid_entries):
        if previous_time is None or (entry.time_seconds != previous_time):
            current_rank = index + 1

        entry.merged_rank = current_rank
        entry.merged_order_index = index

        previous_time = entry.time_seconds

    athlete_entry_counts = Counter()

    for entry in valid_entries:
        for athlete_id in entry.athlete_ids:
            athlete_entry_counts[athlete_id] += 1

    group.overlapping_athlete_ids = tuple(
        sorted(
            athlete_id
            for athlete_id, count in athlete_entry_counts.items()
            if count > 1
        )
    )

    if group.overlapping_athlete_ids:
        stats["group_has_remaining_athlete_overlap"] = 1

    group.entries = valid_entries
    group.evaluation_entries = evaluation_entries
    group.cleaning_stats = stats
    group.usable = len(valid_entries) >= 2

    return group


def _date_sort_key(value):
    text = str(value)

    try:
        parsed = datetime.fromisoformat(text)
        return (
            0,
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
        )
    except ValueError:
        return (1, text)


def prepare_performance_groups(raw_data, keep_unusable=False, group_pruebas=True):
    groups, stats = build_performance_groups(raw_data, group_pruebas=group_pruebas)

    prepared = []

    for group in groups:
        clean_performance_group(group)

        stats.update(group.cleaning_stats)

        if group.is_cross_prueba_merge:
            stats["groups_cross_prueba"] += 1

        if len(group.source_series) > 1:
            stats["groups_multiple_series"] += 1

        if len(group.source_sexes) > 1:
            stats["groups_multiple_sexes"] += 1

        if len(group.source_categories) > 1:
            stats["groups_multiple_categories"] += 1

        if group.usable:
            stats["groups_usable"] += 1
        else:
            stats["groups_unusable"] += 1

        if keep_unusable or group.usable:
            prepared.append(group)

    prepared.sort(
        key=lambda group: (
            _date_sort_key(group.fecha),
            group.source_regatta_index,
            group.first_source_prueba_index,
        )
    )

    return prepared, stats


def group_debug_dict(group):
    return {
        "regata_id": group.regata_id,
        "fecha": group.fecha,
        "fase": group.prueba_fase,
        "tipo": group.tipo,
        "boat": group.embarcacion_tipo,
        "boat_num": (group.embarcacion_num),
        "distance": (group.distancia_exacta),
        "length_class": (group.length_class),
        "distance_is_known": (group.distance_is_known),
        "source_prueba_ids": (group.source_prueba_ids),
        "source_series": (group.source_series),
        "source_sexes": (group.source_sexes),
        "source_categories": (group.source_categories),
        "valid_entries": len(group.entries),
        "overlapping_athlete_ids": (group.overlapping_athlete_ids),
        "cleaning_stats": dict(group.cleaning_stats),
        "usable": group.usable,
    }
