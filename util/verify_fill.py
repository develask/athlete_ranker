import json
import os
from collections import Counter
from copy import deepcopy

# age range (inclusive) implied by each categoria, used to bound birth year
CATEGORY_AGE_RANGE = {
    'prebenjamin': (6, 7),
    'benjamin': (8, 9),
    'alevin': (10, 11),
    'infantil': (12, 13),
    'cadete': (14, 15),
    'junior': (16, 17),
    'sub23': (18, 22),
    'senior': (23, 35),
    'veterano': (36, 100),
}


def load_data_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
    """
    with open(file_path, encoding='utf-8') as file:
        return json.load(file)


def build_athlete_records(data, category_data=None):
    """
    Group every prueba participation by athlete (lista_palista_id).

    Returns a dict mapping athlete id to a dict with a 'participations' list,
    the shape expected by athlete_profile.
    """
    athletes = {}
    for regata_index, regata in enumerate(data):
        year = regata.get('fecha', '')[:4]
        year = int(year) if year.isdigit() else None
        for prueba_index, prueba in enumerate(regata.get('pruebas', [])):
            categoria = prueba.get('categoria')
            if category_data is not None:
                categoria = category_data[regata_index]['pruebas'][prueba_index].get('categoria')
            participation = {
                'year': year,
                'sexo': prueba.get('sexo'),
                'tipo': prueba.get('tipo'),
                'embarcacion_tipo': prueba.get('embarcacion_tipo'),
                'categoria': categoria,
            }
            for result in prueba.get('results', []):
                for athlete_id in result.get('lista_palista_id', []):
                    if not athlete_id:
                        continue
                    athletes.setdefault(athlete_id, {'participations': []})['participations'].append(participation)
    return athletes

def fuzzy_estimate_birth_year_range(birth_ranges):
    """
    Estimate the birth year range from a list of (lo, hi) tuples.

    Args:
        birth_ranges (list): A list of tuples, where each tuple contains the
            lower and upper bounds of the birth year range for a participation.

    Returns:
        tuple: ``((minimum_year, maximum_year), confidence)``. Confidence is
        1.0 for an exact intersection, 0.75 when every range intersects after
        adding a one-year margin, and 0.25-0.75 for a strong consensus after
        ignoring outliers. Returns ``(None, 0)`` when no reliable estimate can
        be made.
    """
    if not birth_ranges:
        return (None, 0)

    def intersection(ranges):
        min_year = max(lo for lo, _ in ranges)
        max_year = min(hi for _, hi in ranges)
        return (min_year, max_year) if min_year <= max_year else None

    for lo_margin in [0, 1]:
        for hi_margin in [0, 1]:
            adjusted_ranges = [(lo - lo_margin, hi + hi_margin) for (lo, hi) in birth_ranges]
            adjusted_range = intersection(adjusted_ranges)
            if adjusted_range is not None:
                return (adjusted_range, 1.0 - 0.125 * (lo_margin + hi_margin))

    # Find the birth years supported by the largest number of participations.
    first_year = min(lo for lo, _ in birth_ranges)
    last_year = max(hi for _, hi in birth_ranges)
    support_by_year = {
        year: sum(lo <= year <= hi for lo, hi in birth_ranges)
        for year in range(first_year, last_year + 1)
    }
    max_support = max(support_by_year.values())
    support_ratio = max_support / len(birth_ranges)

    if max_support < 2 or support_ratio < 0.8:
        return (None, 0)

    consensus_years = [
        year for year, support in support_by_year.items()
        if support == max_support
    ]
    consensus_range = (min(consensus_years), max(consensus_years))
    confidence = 0.25 + (support_ratio - 0.8) / 0.2 * 0.5
    return (consensus_range, confidence)

def athlete_profile(athlete):
    """
    Generate an athlete profile.

    Args:
        athlete (dict): A dictionary containing athlete information.
        - sex_profile (tuple): A tuple containing the athlete's probability of being female, and the count of how many regatas they have participated in (wihtout counting regatas with unknown gender or mixed gender).
        - preferred_tipo (tuple): The athlete's preferred tipo of regata (mar, aguas_tranquilas). more excatly, the probability of the athlete participating in a regata of tipo "aguas_tranquilas". second element is the count of regatas with known tipo.
        - preferred_embarcacion (tuple): The athlete's preferred embarcacion type (kayak, canoa, dragon). so the first element is a tuple with 3 probabilities (kayak, canoa, dragon) and the second element is the count of regatas with known embarcacion type.
        - estimated_birth_year_range (tuple): The estimated birth year range and confidence of the athlete, based on the regatas they have participated in. The first element is a (minimum year, maximum year) tuple (or None), and the second is confidence. calculated based on the year of the regata and:
            - prebenjamin: <= 7 yo
            - benjamin: 8-9 yo
            - alevin: 10-11 yo
            - infantil: 12-13 yo
            - cadete: 14-15 yo
            - juvenil: 16-17 yo
            - sub23: 18-22 yo
            - senior: 18-45 yo
            - veterano: >= 36 yo
    """
    participations = athlete.get('participations', [])

    sexo_counts = {'femenino': 0, 'masculino': 0}
    tipo_counts = {'aguas_tranquilas': 0, 'mar': 0}
    embarcacion_counts = {'kayak': 0, 'canoa': 0, 'dragon_boat': 0}
    birth_ranges = []

    for p in participations:
        if p['sexo'] in sexo_counts:
            sexo_counts[p['sexo']] += 1
        if p['tipo'] in tipo_counts:
            tipo_counts[p['tipo']] += 1
        if p['embarcacion_tipo'] in embarcacion_counts:
            embarcacion_counts[p['embarcacion_tipo']] += 1

        age_range = CATEGORY_AGE_RANGE.get(p['categoria'])
        if age_range is not None:
            lo = p['year'] - age_range[1]
            hi = p['year'] - age_range[0]
            birth_ranges.append((lo, hi))

    sexo_total = sexo_counts['femenino'] + sexo_counts['masculino']
    sex_profile = (sexo_counts['femenino'] / sexo_total, sexo_total) if sexo_total else (None, 0)

    tipo_total = tipo_counts['aguas_tranquilas'] + tipo_counts['mar']
    preferred_tipo = (tipo_counts['aguas_tranquilas'] / tipo_total, tipo_total) if tipo_total else (None, 0)

    embarcacion_total = sum(embarcacion_counts.values())
    if embarcacion_total:
        preferred_embarcacion = (
            tuple(embarcacion_counts[key] / embarcacion_total for key in ('kayak', 'canoa', 'dragon_boat')),
            embarcacion_total,
        )
    else:
        preferred_embarcacion = (None, 0)

    estimated_birth_year_range = fuzzy_estimate_birth_year_range(birth_ranges)

    return {
        'sex_profile': sex_profile,
        'preferred_tipo': preferred_tipo,
        'preferred_embarcacion': preferred_embarcacion,
        'estimated_birth_year_range': estimated_birth_year_range,
    }


def predict_prueba_sexo(athlete_profiles, embarcacion_num, 
                        th_filter_low=0.3, th_filter_high=0.7,
                        th_no_mixto=0.25):
    """
    Predict a prueba's sexo from a weighted average of the competing
    athletes' sex_profile. A single-paddler boat (embarcacion_num == 1)
    can't be mixto, so the prediction falls back to majority femenino/masculino.
    """
    nb_females = 0
    nb_athletes = 0
    for profile in athlete_profiles:
        if profile['sex_profile'][0] is None:
            continue
        if th_filter_low <= profile['sex_profile'][0] <= th_filter_high:
            continue
        if profile['sex_profile'][0] >= 0.5:
            nb_females += 1
        nb_athletes += 1

    if nb_athletes == 0:
        return None

    female_prob = nb_females / nb_athletes

    if embarcacion_num == 1 or female_prob < th_no_mixto or female_prob > (1 - th_no_mixto):
        return 'femenino' if female_prob >= 0.5 else 'masculino'
    else:
        return 'mixto'


def predict_regata_tipo(athlete_profiles):
    """
    Predict a regata's tipo (shared by all its pruebas) from a weighted
    average of preferred_tipo across every athlete competing in the regata.
    """
    probabilities = []
    for profile in athlete_profiles:
        p = profile['preferred_tipo'][0]
        if p is not None:
            probabilities.append(p)

    if not probabilities:
        return None

    aguas_tranquilas_prob = sum(probabilities) / len(probabilities)

    return 'aguas_tranquilas' if aguas_tranquilas_prob >= 0.75 else 'mar'


def predict_prueba_embarcacion(athlete_profiles, embarcacion_num, 
                               th_kayak=0.75, th_no_canoa=0.25):
    """
    Predict a prueba's embarcacion_tipo as whichever type scores highest in
    the average of the competing athletes' preferred_embarcacion. Boats with
    more than 4 paddlers (embarcacion_num > 4) are always dragon_boat.
    """
    if embarcacion_num > 4:
        return 'dragon_boat'

    labels = ('kayak', 'canoa', 'dragon_boat')
    sums = [0, 0, 0]
    count = 0
    for profile in athlete_profiles:
        probabilities = profile['preferred_embarcacion'][0]
        if probabilities is None:
            continue
        for i in range(len(labels)):
            sums[i] += probabilities[i]
        count += 1

    if count == 0:
        return None

    kayak, canoa, db = [s / count for s in sums]

    if kayak >= th_kayak or canoa < th_no_canoa:
        return 'kayak'
    else:
        return 'canoa'


def predict_prueba_categoria(athlete_profiles, year):
    """
    Predict a prueba's categoria from the confidence-weighted average
    birth-year range of its athletes.
    """
    sum_lo, sum_hi, sum_conf = 0, 0, 0

    for profile in athlete_profiles:
        birth_range, confidence = profile['estimated_birth_year_range']
        if birth_range is None or confidence == 0:
            continue

        birth_min, birth_max = birth_range
        sum_lo += birth_min * confidence
        sum_hi += birth_max * confidence
        sum_conf += confidence

    if sum_conf == 0:
        return None

    avg_birth_min = sum_lo / sum_conf
    avg_birth_max = sum_hi / sum_conf

    avg_birth_year = (avg_birth_min + avg_birth_max) / 2
    estimated_age = year - avg_birth_year

    for category, (age_min, age_max) in CATEGORY_AGE_RANGE.items():
        if age_min <= estimated_age <= age_max:
            return category

    # Handle fractional ages between two ranges and estimates outside the
    # configured age bounds by selecting the nearest category.

    def distance_to_category(category):
        age_min, age_max = CATEGORY_AGE_RANGE[category]
        if estimated_age < age_min:
            return age_min - estimated_age
        if estimated_age > age_max:
            return estimated_age - age_max
        return 0

    return min(CATEGORY_AGE_RANGE, key=distance_to_category)





def fill_missing_values(original_data):
    """
    Iteratively fill missing prueba attributes without modifying the input.

    Original values are preserved. A predicted value is frozen once filled;
    fields for which a predictor returns None remain eligible in later rounds.
    Updates are applied synchronously after each round. Predicted categorias do
    not contribute to later birth-year estimates, avoiding circular feedback.
    """
    filled_data = deepcopy(original_data)
    fields = ('sexo', 'tipo', 'embarcacion_tipo', 'categoria')
    iteration = 0

    while True:
        iteration += 1
        missing_by_field = Counter(
            field
            for regata in filled_data
            for prueba in regata.get('pruebas', [])
            for field in fields
            if prueba.get(field) is None
        )
        total_missing = sum(missing_by_field.values())
        if total_missing == 0:
            print(f"Iteration {iteration}: no missing values remain")
            return filled_data

        athletes = build_athlete_records(filled_data, category_data=original_data)
        profiles = {
            athlete_id: athlete_profile(athlete)
            for athlete_id, athlete in athletes.items()
        }
        pending_updates = []

        for regata in filled_data:
            pruebas = regata.get('pruebas', [])
            year_str = regata.get('fecha', '')[:4]
            year = int(year_str) if year_str.isdigit() else None

            if any(prueba.get('tipo') is None for prueba in pruebas):
                regata_athlete_ids = {
                    athlete_id
                    for prueba in pruebas
                    for result in prueba.get('results', [])
                    for athlete_id in result.get('lista_palista_id', [])
                    if athlete_id
                }
                regata_profiles = [
                    profiles[athlete_id]
                    for athlete_id in regata_athlete_ids
                    if athlete_id in profiles
                ]
                tipo_prediction = predict_regata_tipo(regata_profiles)
                if tipo_prediction is not None:
                    pending_updates.extend(
                        (prueba, 'tipo', tipo_prediction)
                        for prueba in pruebas
                        if prueba.get('tipo') is None
                    )

            for prueba in pruebas:
                athlete_profiles = _prueba_profiles(prueba, profiles)

                if prueba.get('sexo') is None:
                    prediction = predict_prueba_sexo(
                        athlete_profiles, prueba.get('embarcacion_num')
                    )
                    if prediction is not None:
                        pending_updates.append((prueba, 'sexo', prediction))

                if prueba.get('embarcacion_tipo') is None:
                    prediction = predict_prueba_embarcacion(
                        athlete_profiles, prueba.get('embarcacion_num')
                    )
                    if prediction is not None:
                        pending_updates.append((prueba, 'embarcacion_tipo', prediction))

                if prueba.get('categoria') is None and year is not None:
                    prediction = predict_prueba_categoria(athlete_profiles, year)
                    if prediction is not None:
                        pending_updates.append((prueba, 'categoria', prediction))

        filled_by_field = Counter(field for _, field, _ in pending_updates)
        filled_count = len(pending_updates)
        filled_percentage = filled_count / total_missing
        print(
            f"Iteration {iteration}: filled {filled_count}/{total_missing} "
            f"missing values ({filled_percentage:.1%})"
        )
        for field in fields:
            field_missing = missing_by_field[field]
            field_filled = filled_by_field[field]
            field_percentage = field_filled / field_missing if field_missing else 0
            print(
                f"  {field}: {field_filled}/{field_missing} "
                f"({field_percentage:.1%})"
            )

        if not pending_updates:
            return filled_data

        for prueba, field, prediction in pending_updates:
            prueba[field] = prediction



def _evaluation_result(correct, total, abstentions, confusion):
    return (correct / total if total else None, total, abstentions, confusion)


def _prueba_profiles(prueba, profiles):
    athlete_ids = {
        athlete_id
        for result in prueba.get('results', [])
        for athlete_id in result.get('lista_palista_id', [])
        if athlete_id
    }
    return [profiles[athlete_id] for athlete_id in athlete_ids if athlete_id in profiles]


def _evaluate_pruebas(data, profiles, field, predict):
    correct, total, abstentions = 0, 0, 0
    confusion = Counter()
    for regata in data:
        for prueba in regata.get('pruebas', []):
            truth = prueba.get(field)
            if truth is None:
                continue
            prediction = predict(regata, prueba, _prueba_profiles(prueba, profiles))
            if prediction is None:
                abstentions += 1
                continue
            total += 1
            correct += truth == prediction
            confusion[(truth, prediction)] += 1
    return _evaluation_result(correct, total, abstentions, confusion)


def evaluate_sexo(data, profiles):
    """Evaluate the sexo predictor once per prueba."""
    return _evaluate_pruebas(
        data, profiles, 'sexo',
        lambda regata, prueba, athlete_profiles: predict_prueba_sexo(
            athlete_profiles, prueba.get('embarcacion_num')
        ),
    )


def evaluate_categoria(data, profiles):
    """Evaluate the categoria predictor once per prueba."""
    def predict(regata, prueba, athlete_profiles):
        year_str = regata.get('fecha', '')[:4]
        year = int(year_str) if year_str.isdigit() else None
        return predict_prueba_categoria(athlete_profiles, year) if year is not None else None

    return _evaluate_pruebas(data, profiles, 'categoria', predict)


def evaluate_tipo(data, profiles):
    """Evaluate the tipo predictor once per regata."""
    correct, total, abstentions = 0, 0, 0
    confusion = Counter()
    for regata in data:
        truth_counts = Counter(
            prueba['tipo']
            for prueba in regata.get('pruebas', [])
            if prueba.get('tipo')
        )
        if not truth_counts:
            continue
        athlete_ids = {
            athlete_id
            for prueba in regata.get('pruebas', [])
            for result in prueba.get('results', [])
            for athlete_id in result.get('lista_palista_id', [])
            if athlete_id
        }
        regata_profiles = [profiles[athlete_id] for athlete_id in athlete_ids if athlete_id in profiles]
        prediction = predict_regata_tipo(regata_profiles) if regata_profiles else None
        if prediction is None:
            abstentions += 1
            continue
        truth = truth_counts.most_common(1)[0][0]
        total += 1
        correct += truth == prediction
        confusion[(truth, prediction)] += 1
    return _evaluation_result(correct, total, abstentions, confusion)


def evaluate_embarcacion(data, profiles):
    """Evaluate the embarcacion_tipo predictor once per prueba."""
    return _evaluate_pruebas(
        data, profiles, 'embarcacion_tipo',
        lambda regata, prueba, athlete_profiles: predict_prueba_embarcacion(
            athlete_profiles, prueba.get('embarcacion_num')
        ),
    )

if __name__ == "__main__":
    data = load_data_json("../data/processed/output.json")

    print("Filling missing values...")
    filled_data = fill_missing_values(data)
    filled_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'filled_v1.json'
    )
    with open(filled_path, 'w', encoding='utf-8') as file:
        json.dump(filled_data, file, ensure_ascii=False, indent=4)
    print(f"Filled data saved to: {os.path.abspath(filled_path)}")

    athletes = build_athlete_records(data)
    print(f"\nTotal athletes: {len(athletes)}")
    profiles = {}
    for athlete_id, athlete in athletes.items():
        profiles[athlete_id] = athlete_profile(athlete)

    def print_confusion_matrix(field, confusion, label_order=None):
        present = {label for pair in confusion for label in pair}
        if label_order:
            labels = [label for label in label_order if label in present]
            labels += sorted(present - set(label_order))
        else:
            labels = sorted(present)
        row_width = max(len(label) for label in labels)
        col_width = 6
        abbreviations = [label[:col_width].rjust(col_width) for label in labels]
        print(f"\n  Confusion matrix for {field} (rows=truth, cols=predicted):")
        for label, abbreviation in zip(labels, abbreviations):
            if label != abbreviation.strip():
                print(f"    {abbreviation.strip()} = {label}")
        print('  ' + ' ' * row_width + ' | ' + ' | '.join(abbreviations))
        for truth in labels:
            row = [str(confusion[(truth, predicted)]).rjust(col_width) for predicted in labels]
            print(f"  {truth.rjust(row_width)} | " + ' | '.join(row))

    # youngest to oldest, so the categoria confusion matrix reads as an age progression
    categoria_age_order = sorted(CATEGORY_AGE_RANGE, key=lambda name: CATEGORY_AGE_RANGE[name])

    print("\nPredictors evaluation:")
    print("\n\n-----------------------------------------------------\n")

    evaluations = {
        'sexo': evaluate_sexo(data, profiles),
        'tipo': evaluate_tipo(data, profiles),
        'embarcacion_tipo': evaluate_embarcacion(data, profiles),
        'categoria': evaluate_categoria(data, profiles),
    }
    for field, (accuracy, n, abstentions, confusion) in evaluations.items():
        accuracy_str = f"{accuracy:.1%} (n={n})" if accuracy is not None else "n/a (n=0)"
        abstention_rate = abstentions / (n + abstentions) if (n + abstentions) else 0
        print(f"  {field}: {accuracy_str}, abstentions={abstentions} ({abstention_rate:.1%})")
        if confusion:
            label_order = categoria_age_order if field == 'categoria' else None
            print_confusion_matrix(field, confusion, label_order)

        print("\n\n-----------------------------------------------------\n")
