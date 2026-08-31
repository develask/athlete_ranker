import math
from collections import Counter

from performance_groups import entries_share_athlete, team_update_factor

# ============================================================
# V1 CONFIGURATION
# ============================================================

DEFAULT_ELO = 1500.0

# Provisional until chronological backtesting.
K_FACTOR = 30.0

# U = lambda / (lambda + n_effective)
LAMBDA_LENGTH = 10.0
LAMBDA_TIPO = 5.0
LAMBDA_BOAT = 5.0

LENGTH_CLASSES = ("super_sprint", "sprint", "fondo", "maraton")

LENGTH_TRANSFER = {
    "super_sprint": {
        "super_sprint": 1.00,
        "sprint": 0.20,
        "fondo": 0.00,
        "maraton": 0.00,
    },
    "sprint": {"super_sprint": 0.20, "sprint": 1.00, "fondo": 0.05, "maraton": 0.00},
    "fondo": {"super_sprint": 0.00, "sprint": 0.05, "fondo": 1.00, "maraton": 0.20},
    "maraton": {"super_sprint": 0.00, "sprint": 0.00, "fondo": 0.20, "maraton": 1.00},
}

# V1: confidence transfer uses the same weights.
CONFIDENCE_TRANSFER = LENGTH_TRANSFER


# ============================================================
# STATE CLASSES
# ============================================================


class LengthRating:
    def __init__(
        self, elo=DEFAULT_ELO, n_direct=0, n_effective=0.0, last_direct_race=None
    ):
        self.elo = float(elo)
        self.n_direct = int(n_direct)
        self.n_effective = float(n_effective)
        self.last_direct_race = last_direct_race


class TipoModifier:
    def __init__(self, raw_modifier=0.0, n_direct=0, n_effective=0.0, last_race=None):
        self.raw_modifier = float(raw_modifier)
        self.n_direct = int(n_direct)
        self.n_effective = float(n_effective)
        self.last_race = last_race


class BoatModifier:
    def __init__(self, raw_modifier=0.0, n_direct=0, n_effective=0.0, last_race=None):
        self.raw_modifier = float(raw_modifier)
        self.n_direct = int(n_direct)
        self.n_effective = float(n_effective)
        self.last_race = last_race


class AthleteState:
    def __init__(self, athlete_id):
        self.athlete_id = str(athlete_id)

        self.length_ratings = {
            length_class: LengthRating() for length_class in LENGTH_CLASSES
        }

        self.tipo_modifiers = {}
        self.boat_modifiers = {}

        # V1: first observed contexts remain
        # primary. No automatic rebasing.
        self.primary_tipo = None
        self.primary_boat = None

        self.tipo_race_counts = Counter()
        self.boat_race_counts = Counter()


class AthletePendingUpdate:
    def __init__(self, athlete_id):
        self.athlete_id = str(athlete_id)

        self.base_delta = 0.0

        self.length_deltas = Counter()
        self.tipo_delta = 0.0
        self.boat_delta = 0.0

        self.length_effective_evidence = Counter()

        self.direct_length = None
        self.direct_length_count = 0

        self.tipo = None
        self.tipo_direct_count = 0
        self.tipo_effective_evidence = 0.0

        self.boat = None
        self.boat_direct_count = 0
        self.boat_effective_evidence = 0.0

        self.team_size = None
        self.team_factor = None

        # Number of legitimate entries this
        # athlete had in this merged group.
        self.appearance_count = 1


class GroupProcessResult:
    def __init__(self):
        self.processed = False
        self.skip_reason = None

        self.team_ratings = []
        self.team_base_deltas = []

        self.athlete_updates = {}
        self.stats = Counter()

        self.evaluation = {
            "pairwise_comparisons": 0,
            "pairwise_decisive": 0,
            "pairwise_ties": 0,
            "brier_sum": 0.0,
            "log_loss_sum": 0.0,
            "correct_decisive": 0,
        }


# ============================================================
# NORMALIZATION / ATHLETE HELPERS
# ============================================================


def normalize_context(value):
    if value is None:
        return None

    text = " ".join(str(value).split()).casefold()

    return text or None


def normalize_length_class(value):
    value = normalize_context(value)

    if value in LENGTH_CLASSES:
        return value

    return None


def get_athlete(athletes, athlete_id):
    athlete_id = str(athlete_id)

    if athlete_id not in athletes:
        athletes[athlete_id] = AthleteState(athlete_id)

    return athletes[athlete_id]


def initialize_primary_roles(athlete, tipo, boat):
    if athlete.primary_tipo is None:
        athlete.primary_tipo = tipo

    if athlete.primary_boat is None:
        athlete.primary_boat = boat


def get_tipo_modifier(athlete, tipo, create=True):
    tipo = normalize_context(tipo)

    if tipo == athlete.primary_tipo:
        return None

    modifier = athlete.tipo_modifiers.get(tipo)

    if modifier is None and create:
        modifier = TipoModifier()
        athlete.tipo_modifiers[tipo] = modifier

    return modifier


def get_boat_modifier(athlete, boat, create=True):
    boat = normalize_context(boat)

    if boat == athlete.primary_boat:
        return None

    modifier = athlete.boat_modifiers.get(boat)

    if modifier is None and create:
        modifier = BoatModifier()
        athlete.boat_modifiers[boat] = modifier

    return modifier


def effective_tipo_modifier(athlete, tipo):
    tipo = normalize_context(tipo)

    if tipo == athlete.primary_tipo:
        return 0.0

    modifier = get_tipo_modifier(athlete, tipo, create=False)

    if modifier is None:
        return 0.0

    return modifier.raw_modifier


def effective_boat_modifier(athlete, boat):
    boat = normalize_context(boat)

    if boat == athlete.primary_boat:
        return 0.0

    modifier = get_boat_modifier(athlete, boat, create=False)

    if modifier is None:
        return 0.0

    return modifier.raw_modifier


def get_effective_rating(athlete, length_class, tipo, boat):
    return (
        athlete.length_ratings[length_class].elo
        + effective_tipo_modifier(athlete, tipo)
        + effective_boat_modifier(athlete, boat)
    )


# ============================================================
# UNCERTAINTY
# ============================================================


def uncertainty_factor(n_effective, lambda_):
    n_effective = max(0.0, float(n_effective))

    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")

    return float(lambda_) / (float(lambda_) + n_effective)


def length_uncertainty(athlete, length_class):
    rating = athlete.length_ratings[length_class]

    return uncertainty_factor(rating.n_effective, LAMBDA_LENGTH)


def tipo_uncertainty(athlete, tipo):
    tipo = normalize_context(tipo)

    if tipo == athlete.primary_tipo:
        return 0.0

    modifier = get_tipo_modifier(athlete, tipo, create=False)

    n_effective = 0.0 if modifier is None else modifier.n_effective

    return uncertainty_factor(n_effective, LAMBDA_TIPO)


def boat_uncertainty(athlete, boat):
    boat = normalize_context(boat)

    if boat == athlete.primary_boat:
        return 0.0

    modifier = get_boat_modifier(athlete, boat, create=False)

    n_effective = 0.0 if modifier is None else modifier.n_effective

    return uncertainty_factor(n_effective, LAMBDA_BOAT)


# ============================================================
# EXPECTATION / PERFORMANCE
# ============================================================


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_pairwise_score(entry_a, entry_b):
    if entry_a.time_seconds < entry_b.time_seconds:
        return 1.0

    if entry_a.time_seconds > entry_b.time_seconds:
        return 0.0

    return 0.5


def ensure_group_athletes(athletes, group, tipo, boat):
    for entry in group.entries:
        for athlete_id in entry.athlete_ids:
            athlete = get_athlete(athletes, athlete_id)

            initialize_primary_roles(athlete, tipo=tipo, boat=boat)


def get_team_rating(athletes, entry, length_class, tipo, boat):
    ratings = [
        get_effective_rating(
            athletes[str(athlete_id)], length_class=length_class, tipo=tipo, boat=boat
        )
        for athlete_id in entry.athlete_ids
    ]

    if not ratings:
        raise ValueError("Cannot rate an empty crew")

    return sum(ratings) / len(ratings)


def calculate_team_base_deltas(group, team_ratings, k=K_FACTOR):
    """
    Pairwise multiplayer Elo.

    For each entry:
      actual   = average score against
                 valid opponents
      expected = average expectation
                 against valid opponents
      delta    = K * (actual - expected)

    Two entries sharing an athlete are not
    compared against one another.
    """
    n = len(group.entries)

    if n < 2:
        return [0.0] * n

    output = []

    for i, entry_i in enumerate(group.entries):
        actual_total = 0.0
        expected_total = 0.0
        opponent_count = 0

        for j, entry_j in enumerate(group.entries):
            if i == j:
                continue

            if entries_share_athlete(entry_i, entry_j):
                continue

            actual_total += actual_pairwise_score(entry_i, entry_j)
            expected_total += expected_score(team_ratings[i], team_ratings[j])
            opponent_count += 1

        if opponent_count == 0:
            output.append(0.0)
            continue

        actual = actual_total / opponent_count
        expected = expected_total / opponent_count

        output.append(float(k) * (actual - expected))

    return output


def calculate_pairwise_evaluation(group, team_ratings):
    """
    Pre-update prediction metrics over unordered
    valid entry pairs.
    """
    metrics = {
        "pairwise_comparisons": 0,
        "pairwise_decisive": 0,
        "pairwise_ties": 0,
        "brier_sum": 0.0,
        "log_loss_sum": 0.0,
        "correct_decisive": 0,
    }

    epsilon = 1e-15

    for i in range(len(group.entries)):
        for j in range(i + 1, len(group.entries)):
            entry_i = group.entries[i]
            entry_j = group.entries[j]

            if entries_share_athlete(entry_i, entry_j):
                continue

            probability = expected_score(team_ratings[i], team_ratings[j])
            actual = actual_pairwise_score(entry_i, entry_j)

            metrics["pairwise_comparisons"] += 1

            if actual == 0.5:
                metrics["pairwise_ties"] += 1
            else:
                metrics["pairwise_decisive"] += 1

                predicted_i_wins = probability > 0.5
                actual_i_wins = actual == 1.0

                if predicted_i_wins == actual_i_wins:
                    metrics["correct_decisive"] += 1

            metrics["brier_sum"] += (probability - actual) ** 2

            p = min(1.0 - epsilon, max(epsilon, probability))

            metrics["log_loss_sum"] += -(
                actual * math.log(p) + (1.0 - actual) * math.log(1.0 - p)
            )

    return metrics


# ============================================================
# COMPONENT ALLOCATION
# ============================================================


def calculate_component_shares(athlete, length_class, tipo, boat):
    u_length = length_uncertainty(athlete, length_class)
    u_tipo = tipo_uncertainty(athlete, tipo)
    u_boat = boat_uncertainty(athlete, boat)

    total = u_length + u_tipo + u_boat

    if total <= 0:
        return {"length": 1.0, "tipo": 0.0, "boat": 0.0}

    return {"length": u_length / total, "tipo": u_tipo / total, "boat": u_boat / total}


def calculate_length_deltas(athlete, raced_length, allocated_length_delta):
    """
    Direct length receives the full allocated
    length delta.

    Other lengths:
      delta_length
      * transfer_weight
      * target_uncertainty

    There is no recursive propagation.
    """
    output = {}

    for target_length in LENGTH_CLASSES:
        weight = LENGTH_TRANSFER[raced_length][target_length]

        if weight <= 0:
            output[target_length] = 0.0
            continue

        if target_length == raced_length:
            output[target_length] = allocated_length_delta
            continue

        output[target_length] = (
            allocated_length_delta * weight * length_uncertainty(athlete, target_length)
        )

    return output


def calculate_athlete_pending_update(
    athlete, entry, team_base_delta, length_class, tipo, boat
):
    update = AthletePendingUpdate(athlete.athlete_id)

    team_size = len(entry.athlete_ids)
    factor = team_update_factor(team_size)

    athlete_base_delta = team_base_delta * factor

    shares = calculate_component_shares(
        athlete, length_class=length_class, tipo=tipo, boat=boat
    )

    allocated_length_delta = athlete_base_delta * shares["length"]
    update.tipo_delta = athlete_base_delta * shares["tipo"]
    update.boat_delta = athlete_base_delta * shares["boat"]

    length_deltas = calculate_length_deltas(
        athlete,
        raced_length=length_class,
        allocated_length_delta=(allocated_length_delta),
    )

    update.base_delta = athlete_base_delta

    for target_length in LENGTH_CLASSES:
        update.length_deltas[target_length] = length_deltas[target_length]

        update.length_effective_evidence[target_length] = (
            factor * CONFIDENCE_TRANSFER[length_class][target_length]
        )

    update.direct_length = length_class
    update.direct_length_count = 1

    update.tipo = tipo
    update.tipo_direct_count = 1
    update.tipo_effective_evidence = factor

    update.boat = boat
    update.boat_direct_count = 1
    update.boat_effective_evidence = factor

    update.team_size = team_size
    update.team_factor = factor

    return update


def average_pending_updates(athlete_id, updates):
    """
    Same athlete may legitimately appear more
    than once in one merged performance group.

    V1:
    - average all Elo/component deltas
    - average effective evidence
    - n_direct increments once
    """
    if not updates:
        raise ValueError("updates must not be empty")

    if len(updates) == 1:
        updates[0].appearance_count = 1
        return updates[0]

    count = len(updates)
    first = updates[0]

    combined = AthletePendingUpdate(athlete_id)
    combined.appearance_count = count

    combined.base_delta = sum(update.base_delta for update in updates) / count

    for length_class in LENGTH_CLASSES:
        combined.length_deltas[length_class] = (
            sum(update.length_deltas[length_class] for update in updates) / count
        )

        combined.length_effective_evidence[length_class] = (
            sum(update.length_effective_evidence[length_class] for update in updates)
            / count
        )

    combined.tipo_delta = sum(update.tipo_delta for update in updates) / count
    combined.boat_delta = sum(update.boat_delta for update in updates) / count

    combined.direct_length = first.direct_length
    combined.direct_length_count = 1

    combined.tipo = first.tipo
    combined.tipo_direct_count = 1
    combined.tipo_effective_evidence = (
        sum(update.tipo_effective_evidence for update in updates) / count
    )

    combined.boat = first.boat
    combined.boat_direct_count = 1
    combined.boat_effective_evidence = (
        sum(update.boat_effective_evidence for update in updates) / count
    )

    combined.team_size = first.team_size
    combined.team_factor = sum(update.team_factor for update in updates) / count

    return combined


# ============================================================
# APPLY UPDATES
# ============================================================


def apply_pending_update(athlete, update, group):
    for length_class in LENGTH_CLASSES:
        athlete.length_ratings[length_class].elo += update.length_deltas[length_class]

    direct_rating = athlete.length_ratings[update.direct_length]

    direct_rating.n_direct += 1
    direct_rating.last_direct_race = group.fecha

    for length_class in LENGTH_CLASSES:
        athlete.length_ratings[
            length_class
        ].n_effective += update.length_effective_evidence[length_class]

    # One literal group count per athlete.
    athlete.tipo_race_counts[update.tipo] += 1
    athlete.boat_race_counts[update.boat] += 1

    if update.tipo != athlete.primary_tipo:
        modifier = get_tipo_modifier(athlete, update.tipo, create=True)

        modifier.raw_modifier += update.tipo_delta
        modifier.n_direct += 1
        modifier.n_effective += update.tipo_effective_evidence
        modifier.last_race = group.fecha

    if update.boat != athlete.primary_boat:
        modifier = get_boat_modifier(athlete, update.boat, create=True)

        modifier.raw_modifier += update.boat_delta
        modifier.n_direct += 1
        modifier.n_effective += update.boat_effective_evidence
        modifier.last_race = group.fecha


def process_performance_group(athletes, group, k=K_FACTOR):
    result = GroupProcessResult()

    if not group.usable or len(group.entries) < 2:
        result.skip_reason = "fewer_than_two_valid_entries"
        return result

    length_class = normalize_length_class(group.length_class)
    tipo = normalize_context(group.tipo)
    boat = normalize_context(group.embarcacion_tipo)

    if length_class is None:
        result.skip_reason = "unsupported_length_class"
        return result

    if tipo is None or boat is None:
        result.skip_reason = "missing_tipo_or_boat"
        return result

    ensure_group_athletes(athletes=athletes, group=group, tipo=tipo, boat=boat)

    # Freeze all pre-group team ratings.
    team_ratings = [
        get_team_rating(
            athletes=athletes,
            entry=entry,
            length_class=length_class,
            tipo=tipo,
            boat=boat,
        )
        for entry in group.entries
    ]

    team_base_deltas = calculate_team_base_deltas(
        group=group, team_ratings=team_ratings, k=k
    )

    result.evaluation = calculate_pairwise_evaluation(group, team_ratings)

    # Calculate all appearance-level updates
    # before mutating any state.
    pending_lists = {}

    for entry, team_delta in zip(group.entries, team_base_deltas):
        for athlete_id in entry.athlete_ids:
            athlete = athletes[str(athlete_id)]

            pending = calculate_athlete_pending_update(
                athlete=athlete,
                entry=entry,
                team_base_delta=(team_delta),
                length_class=(length_class),
                tipo=tipo,
                boat=boat,
            )

            pending_lists.setdefault(athlete.athlete_id, []).append(pending)

    pending_updates = {
        athlete_id: (average_pending_updates(athlete_id, updates))
        for athlete_id, updates in pending_lists.items()
    }

    # Apply only after every update has
    # been computed from frozen state.
    for athlete_id, pending in pending_updates.items():
        apply_pending_update(athlete=athletes[athlete_id], update=pending, group=group)

    result.processed = True
    result.team_ratings = team_ratings
    result.team_base_deltas = team_base_deltas
    result.athlete_updates = pending_updates

    result.stats["entries_processed"] = len(group.entries)
    result.stats["athletes_processed"] = len(pending_updates)
    result.stats["athletes_with_multiple_appearances"] = sum(
        1 for update in pending_updates.values() if update.appearance_count > 1
    )

    return result


def run_rating_system(groups, k=K_FACTOR, collect_group_results=False):
    athletes = {}
    stats = Counter()
    evaluation = Counter()
    group_results = [] if collect_group_results else None

    for group in groups:
        result = process_performance_group(athletes=athletes, group=group, k=k)

        if collect_group_results:
            group_results.append(result)

        if not result.processed:
            stats["groups_skipped"] += 1
            stats["skip_" + str(result.skip_reason)] += 1
            continue

        stats["groups_processed"] += 1
        stats["entries_processed"] += result.stats["entries_processed"]
        stats["athletes_processed_occurrences"] += result.stats["athletes_processed"]
        stats["athletes_with_multiple_appearances"] += result.stats[
            "athletes_with_multiple_appearances"
        ]

        for key, value in result.evaluation.items():
            evaluation[key] += value

    stats["athletes_created"] = len(athletes)

    return (athletes, stats, evaluation, group_results)


# ============================================================
# SERIALIZATION HELPERS
# ============================================================


def athlete_debug_dict(athlete):
    return {
        "athlete_id": (athlete.athlete_id),
        "primary_tipo": (athlete.primary_tipo),
        "primary_boat": (athlete.primary_boat),
        "length_ratings": {
            length_class: {
                "elo": rating.elo,
                "n_direct": (rating.n_direct),
                "n_effective": (rating.n_effective),
                "last_direct_race": (rating.last_direct_race),
            }
            for length_class, rating in athlete.length_ratings.items()
        },
        "tipo_modifiers": {
            tipo: {
                "raw_modifier": (modifier.raw_modifier),
                "n_direct": (modifier.n_direct),
                "n_effective": (modifier.n_effective),
                "last_race": (modifier.last_race),
            }
            for tipo, modifier in athlete.tipo_modifiers.items()
        },
        "boat_modifiers": {
            boat: {
                "raw_modifier": (modifier.raw_modifier),
                "n_direct": (modifier.n_direct),
                "n_effective": (modifier.n_effective),
                "last_race": (modifier.last_race),
            }
            for boat, modifier in athlete.boat_modifiers.items()
        },
        "tipo_race_counts": dict(athlete.tipo_race_counts),
        "boat_race_counts": dict(athlete.boat_race_counts),
    }
