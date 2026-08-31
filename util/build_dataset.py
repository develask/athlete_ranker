"""
End-to-end pipeline: raw CSVs -> single processed dataset.json.

Parses raw CSVs into nested regata/prueba records, derives fields, fills
missing sexo/tipo/embarcacion_tipo/categoria values, and fills missing
length_class values, with no intermediate JSON/CSV files.
"""

import csv
import math
import os
import re
import statistics
import unicodedata
import json
from collections import Counter
from copy import deepcopy

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# CSV extraction and field derivation
# ---------------------------------------------------------------------------

LENGTH_CLASSES = (
    'super_sprint', # <= 500m
    'sprint',       # 501-2000m
    'fondo',        # 2001-8000m, o mar <= 8000m
    'maraton',       # > 8000m
)


def length_class(tipo, distance):
    """Map a distance in metres to its tipo-specific length class, or None if unknown."""
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(distance):
        return None

    if tipo == 'aguas_tranquilas':
        if distance <= 500:
            return LENGTH_CLASSES[0]
        if distance <= 2000:
            return LENGTH_CLASSES[1]
        if distance <= 8000:
            return LENGTH_CLASSES[2]
        return LENGTH_CLASSES[3]
    if tipo == 'mar':
        return LENGTH_CLASSES[2] if distance <= 8000 else LENGTH_CLASSES[3]
    return None


def read_data_csv(file_path):
    """
    Return the CSV rows and header.
    """
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    return data, header


def extract_all_years(years=None, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

    files = os.listdir(data_dir)
    data = []
    header = None
    for file in files:
        if file.endswith('.csv'):
            year = file.split('.')[0]
            if years is None or year in years:
                file_path = os.path.join(data_dir, file)
                year_data, header = read_data_csv(file_path)
                data.extend(year_data)
    if header is None:
        raise ValueError(f'No CSV files found in {data_dir}')
    return data, header


def convert2json(data, header):
    """
    Convert flat CSV rows into nested regata/prueba dictionaries.
    """
    # map header names to indices for safe access
    idx = {name: i for i, name in enumerate(header)}

    def _int(v):
        try:
            return int(v)
        except Exception:
            return None

    regatas = {}

    for row in data:
        # ensure row has at least as many columns as header
        if len(row) < len(header):
            # pad with empty strings
            row = row + [''] * (len(header) - len(row))

        regata_id = row[idx.get('regata_id')]
        prueba_id = row[idx.get('prueba_id')]

        # create regata entry if needed
        if regata_id not in regatas:
            regatas[regata_id] = {
                'regata_id': regata_id,
                'regata_name': row[idx.get('regata_name')],
                'fecha': row[idx.get('fecha')],
                'organizador_id': row[idx.get('organizador_id')],
                'organizador_name': row[idx.get('organizador_name')],
                'regata_liga': row[idx.get('regata_liga')],
                'pruebas': {}
            }

        regata = regatas[regata_id]

        # create prueba entry within regata if needed
        pruebas = regata['pruebas']
        if prueba_id not in pruebas:
            pruebas[prueba_id] = {
                'prueba_id': prueba_id,
                'prueba_name': row[idx.get('prueba_name')],
                'prueba_fase': row[idx.get('prueba_fase')],
                'prueba_serie': row[idx.get('prueba_serie')],
                'results': []
            }

        prueba = pruebas[prueba_id]

        # build result object
        result = {
            'lista_palista_id': row[idx.get('lista_palista_id')].split(' '),
            'calle': row[idx.get('calle')],
            'dorsal': _int(row[idx.get('dorsal')]) if idx.get('dorsal') is not None else None,
            'posicion': _int(row[idx.get('posicion')]) if idx.get('posicion') is not None else None,
            'tiempo': row[idx.get('tiempo')],
            'puntos': _int(row[idx.get('puntos')]) if idx.get('puntos') is not None else None,
            'anotaciones': row[idx.get('anotaciones')]
        }

        prueba['results'].append(result)

    # convert regatas/pruebas dicts into lists for JSON-serializable structure
    out = []
    for reg in regatas.values():
        reg_copy = reg.copy()
        reg_copy['pruebas'] = []
        for p in reg['pruebas'].values():
            reg_copy['pruebas'].append(p)
        out.append(reg_copy)

    return out


def _parse_tiempo(t):
    m = re.match(r'^(\d{2}):(\d{2}):(\d{2}(?:\.\d{1,2})?)$', t or '')
    if not m:
        return None
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _format_tiempo(seconds):
    h = int(seconds // 3600)
    mn = int((seconds % 3600) // 60)
    s = seconds % 60
    return f'{h:02d}:{mn:02d}:{s:05.2f}'


def additional_info(json_data):
    """
        Extract metadata and finish-time statistics for each prueba. The output
        keeps only aguas_tranquilas and mar pruebas, excluding paleo and
        paddleboard boats.

        Fields extracted include:
        - distancia_exacta, tipo, sexo, categoria
        - embarcacion_tipo, embarcacion_num
        - n_resultados, tiempo_min, tiempo_max, tiempo_medio, tiempo_mediana
            (finish-time statistics formatted as HH:MM:SS.ff)
    """

    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

    # common category abbreviation map (order matters: longest/most specific first)
    # juvenil folds into junior, and the prebenjamin/abridor/absoluta variants
    # (typos, abbreviations, age-suffixed forms) all fold into one canonical name
    cat_map = {
        'INFANTIL': 'infantil', 'INF': 'infantil',
        'CADETE': 'cadete', 'CAD': 'cadete',
        'SENIOR': 'senior', 'SEN': 'senior',
        'JUNIOR': 'junior', 'JUN': 'junior', 'JUVENIL': 'junior', 'JUV': 'junior',
        'VETERANO': 'veterano', 'VET': 'veterano',
        'ALEVIN': 'alevin', 'ALE': 'alevin',
        'BENJAMIN': 'benjamin', 'BEN': 'benjamin',
        'PREBENJAMIN': 'prebenjamin', 'PREBE': 'prebenjamin', 'PREB': 'prebenjamin',
        'PBENJA': 'prebenjamin', 'PBENJ': 'prebenjamin', 'PBJ': 'prebenjamin',
        'SUB23': 'sub23', 'SUB-23': 'sub23', 'S23': 'sub23',
    }
    # group 1 = base code, group 2 = optional attached age suffix (e.g. "40-44", "3554", "-A")
    cat_pattern = re.compile(
        r'\b(' + '|'.join(cat_map.keys()) + r'|S\d{2})(-?[A-Z]|\d{2,4}(?:-\d{2,4})?)?\b',
        re.IGNORECASE
    )

    # boat/embarcacion codes, longest/most specific first so e.g. "KL2" wins over "K".
    # crew-size digits may have another code glued right after with no separator
    # (e.g. "DB12OV", "DB12FS"), so only the no-digit form requires a trailing
    # word boundary - the digit form just can't run into more digits.
    embarcacion_pattern = re.compile(
        r'\b(?:\d+\s*[xX]\s*)?(PALEO|SUP|OC|KL|VL|KS|SS|DB|V|K|C)(?:-?(\d{1,2})(?!\d)|\b)'
    )
    known_boat_prefixes = {'PALEO', 'SUP', 'OC', 'KL', 'VL', 'KS', 'SS', 'DB', 'V', 'K', 'C'}
    boat_type_names = {
        'K': 'kayak', 'KL': 'kayak', 'KS': 'kayak', 'SS': 'kayak',
        'C': 'canoa', 'OC': 'canoa',
        'DB': 'dragon_boat',
        'V': 'canoa', 'VL': 'canoa',
        'SUP': 'paddleboard',
        'PALEO': 'paleo',
    }
    DISCARDED_EMBARCACION_TIPOS = {'paleo', 'paddleboard'}
    ALLOWED_TIPOS = {'aguas_tranquilas', 'mar'}

    ignore_words = {
        'FINAL', 'HEAT', 'HEATS', 'SERIE', 'REPECHAJE', 'SPRINTER', 'SPRINT',
        'TRADICIONAL', 'MIXTO', 'MIXTA', 'CROSS', 'INCLUSIVO', 'MARATON',
        'MAR', 'DRAGON', 'PALEO', 'DES', 'A', 'B', 'C',
    }

    # tipo derived from the regata's liga/modalidad, which is far more reliable
    # than guessing from prueba_name (mar/dragon/slalom rarely appear there).
    # sprint/fondo/maraton/dragon are all flatwater and are merged into one
    # aguas_tranquilas tipo - the distinction between them is left to be
    # derived downstream from distancia_exacta and embarcacion_num instead.
    # paracanoe is NOT a tipo of its own - a paracanoe prueba is classified
    # the same as any other (mar/aguas_tranquilas), just not filtered on.
    liga_tipo_map = [
        ('dragon', 'aguas_tranquilas'),
        ('kaiak de mar', 'mar'), ('kayak de mar', 'mar'),
        ('caiac mar', 'mar'), ('itsas kayaka', 'mar'),
        ('aguas bravas', 'slalom'), ('augas bravas', 'slalom'), ('ur biziak', 'slalom'),
        ('slalom', 'slalom'),
        ('descenso', 'descenso'),
    ]

    for reg in json_data:
        liga_norm = _strip_accents((reg.get('regata_liga') or '').lower())
        tipo_from_liga = None
        for key, val in liga_tipo_map:
            if key in liga_norm:
                tipo_from_liga = val
                break

        for prueba in reg.get('pruebas', []):
            name = prueba.get('prueba_name', '') or ''
            low = _strip_accents(name.lower())
            upper = _strip_accents(name.upper())

            # embarcacion_tipo: search for known boat-class codes anywhere in
            # the name (e.g. K1, C2, DB12). The digit next to the prefix is a
            # class/division code (e.g. "DB1" = dragon boat division 1), NOT
            # a reliable crew size - a DB1 race still has ~20 paddlers per
            # boat - so it's used only to locate/mask the span, never as
            # embarcacion_num.
            embarcacion_tipo = None
            emb_span = None
            for m in embarcacion_pattern.finditer(upper):
                prefix = m.group(1)
                if prefix in known_boat_prefixes:
                    embarcacion_tipo = boat_type_names.get(prefix)
                    emb_span = m.span()
                    break

            # embarcacion_num: ALWAYS the mode of paddlers actually seen per
            # boat in this prueba's results (lista_palista_id) - ground truth
            # from the data, never derived from the prueba_name. None if there
            # are no results to estimate from.
            participant_counts = [
                len([pid for pid in r.get('lista_palista_id', []) if pid.strip()])
                for r in prueba.get('results', [])
            ]
            participant_counts = [c for c in participant_counts if c > 0]
            embarcacion_num = Counter(participant_counts).most_common(1)[0][0] if participant_counts else None

            prueba['embarcacion_tipo'] = embarcacion_tipo
            prueba['embarcacion_num'] = embarcacion_num

            # categoria: match known category codes (e.g. VET, VET-A, VET40-44, S23)
            categoria = None
            cat_span = None
            category_matches = list(cat_pattern.finditer(upper))
            cm = next(
                (match for match in category_matches if match.group(1).upper() in {'S23', 'SUB23'}),
                category_matches[0] if category_matches else None,
            )
            if cm:
                token = cm.group(1).upper()
                categoria = cat_map.get(token, 'sub23' if token in {'S23', 'SUB23'} else token.lower())
                cat_span = cm.span()

            # distancia_exacta: last standalone number with 2+ digits, ignoring
            # digits already consumed by the embarcacion/categoria codes above
            masked = list(name)
            for span in (emb_span, cat_span):
                if span:
                    for i in range(*span):
                        masked[i] = ' '
            masked_name = ''.join(masked)

            distancia = None
            for m in re.finditer(r'(?<!\d)(\d{2,6})(?!\d)', masked_name):
                distancia = int(m.group(1))
            # values under 100 are extraction errors (actually veterano age
            # ranges caught by the number regex), not real race distances
            if distancia is not None and distancia < 100:
                distancia = None
            prueba['distancia_exacta'] = distancia

            # tipo: mar/aguas_bravas/slalom/descenso come from the regata's
            # liga/modalidad (or a name keyword fallback for mar);
            # everything else (sprint/fondo/maraton/dragon) is aguas_tranquilas
            if tipo_from_liga:
                tipo = tipo_from_liga
            elif 'mar' in low.split():
                tipo = 'mar'
            else:
                tipo = 'aguas_tranquilas'

            prueba['tipo'] = tipo
            prueba['length_class'] = length_class(tipo, distancia)

            # sexo: look for a standalone H/M/W/MX/X token, else keyword fallback
            sexo = None
            if re.search(r'\bMX\b', upper) or re.search(r'\bX\b', upper):
                sexo = 'mixto'
            elif re.search(r'\bH\b', upper):
                sexo = 'masculino'
            elif re.search(r'\b[MW]\b', upper):
                sexo = 'femenino'
            elif 'mixto' in low or 'mixta' in low:
                sexo = 'mixto'
            elif 'muj' in low or 'fem' in low or 'muller' in low:
                sexo = 'femenino'
            elif 'hombre' in low or 'masc' in low or re.search(r'\bhome\b', low):
                sexo = 'masculino'

            prueba['sexo'] = sexo

            if not cm:
                # fallback: first token that isn't sexo/embarcacion/number/noise word
                for t in _strip_accents(name).split():
                    up = t.upper()
                    if up in ('H', 'M', 'W', 'X', 'MX'):
                        continue
                    if up in ignore_words:
                        continue
                    if re.match(r'^\d+$', t):
                        continue
                    m = embarcacion_pattern.match(up)
                    if m and m.group(1) in known_boat_prefixes:
                        continue
                    categoria = up.split('-')[0].lower()
                    break

            prueba['categoria'] = categoria

            # tiempo stats over valid finish times, kept for manual inspection
            # (output.csv) without carrying every participant's row
            tiempos = [_parse_tiempo(r.get('tiempo', '')) for r in prueba['results']]
            tiempos = [t for t in tiempos if t is not None]
            prueba['n_resultados'] = len(prueba['results'])
            prueba['tiempo_min'] = _format_tiempo(min(tiempos)) if tiempos else None
            prueba['tiempo_max'] = _format_tiempo(max(tiempos)) if tiempos else None
            prueba['tiempo_medio'] = _format_tiempo(statistics.mean(tiempos)) if tiempos else None
            prueba['tiempo_mediana'] = _format_tiempo(statistics.median(tiempos)) if tiempos else None

        reg['pruebas'] = [
            p for p in reg['pruebas']
            if p['tipo'] in ALLOWED_TIPOS
            and p['embarcacion_tipo'] not in DISCARDED_EMBARCACION_TIPOS
        ]

    # Keep only the canonical categories defined in cat_map.
    kept_categories = {
        'infantil', 'cadete', 'senior', 'junior', 'veterano',
        'alevin', 'benjamin', 'prebenjamin', 'sub23',
    }
    for reg in json_data:
        for prueba in reg['pruebas']:
            if prueba['categoria'] not in kept_categories:
                prueba['categoria'] = None

    return [reg for reg in json_data if reg['pruebas']]


# ---------------------------------------------------------------------------
# Filling missing sexo/tipo/embarcacion_tipo/categoria values
# ---------------------------------------------------------------------------

# age range (inclusive) implied by each categoria, used to bound birth year
CATEGORY_AGE_RANGE = {
    'prebenjamin': (5, 7),
    'benjamin': (8, 9),
    'alevin': (10, 11),
    'infantil': (12, 13),
    'cadete': (14, 15),
    'junior': (16, 17),
    'sub23': (18, 22),
    'senior': (20, 35),
    'veterano': (36, 100),
}


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
            - senior: 20-45 yo
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


def _prueba_profiles(prueba, profiles):
    athlete_ids = {
        athlete_id
        for result in prueba.get('results', [])
        for athlete_id in result.get('lista_palista_id', [])
        if athlete_id
    }
    return [profiles[athlete_id] for athlete_id in athlete_ids if athlete_id in profiles]


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


# ---------------------------------------------------------------------------
# Filling missing length_class values
# ---------------------------------------------------------------------------

# shared with train_length_classifier and fill_length_class so prediction rows
# are built identically to training rows
TIME_FEATURES = (
    'best_time',
    'time_q10',
    'time_q25',
    'time_q50',
    'time_q75',
    'time_q90',
    'average_time',
    'time_std',
)
OTHER_NUMERIC_FEATURES = (
    'num_results',
    'embarcacion_num',
)
NUMERIC_FEATURES = TIME_FEATURES + OTHER_NUMERIC_FEATURES
CATEGORICAL_FEATURES = (
    'embarcacion_tipo',
    'tipo',
    'sexo',
    'categoria',
)


def _feature_row(row):
    """Build one model input row from a feature dict, in the trained column order."""
    return [
        *(
            row.get(name) if row.get(name) is not None else np.nan
            for name in NUMERIC_FEATURES
        ),
        *(
            row.get(name) if row.get(name) is not None else np.nan
            for name in CATEGORICAL_FEATURES
        ),
    ]


def extract_features(data):
    """
    Extract one feature row per prueba.

    ``regata_id`` is retained as a grouping column for cross-validation and
    ``distancia_exacta`` is retained as the regression target. Pruebas with a
    missing target are included so the same function can later prepare data
    for prediction. Missing or invalid result times are excluded from the time
    statistics; if none remain, every time statistic is None.

    Returns:
        list[dict]: Feature rows in the same order as the pruebas in ``data``.
    """
    def time_to_seconds(value):
        if value is None:
            return None
        try:
            hours, minutes, seconds = str(value).strip().replace(',', '.').split(':')
            total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(total) or total < 10 or total > 3 * 3600:
            return None
        return total

    def percentile(sorted_values, quantile):
        if not sorted_values:
            return None
        position = (len(sorted_values) - 1) * quantile
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return sorted_values[lower_index]
        weight = position - lower_index
        return (
            sorted_values[lower_index] * (1 - weight)
            + sorted_values[upper_index] * weight
        )

    rows = []
    for regata in data:
        for prueba in regata.get('pruebas', []):
            results = prueba.get('results', [])
            times = sorted(
                seconds
                for result in results
                if (seconds := time_to_seconds(result.get('tiempo'))) is not None
            )

            row = {
                'regata_id': regata.get('regata_id'),
                'prueba_id': prueba.get('prueba_id'),
                'distancia_exacta': prueba.get('distancia_exacta'),
                'best_time': times[0] if times else None,
                'time_q10': percentile(times, 0.10),
                'time_q25': percentile(times, 0.25),
                'time_q50': percentile(times, 0.50),
                'time_q75': percentile(times, 0.75),
                'time_q90': percentile(times, 0.90),
                'average_time': statistics.fmean(times) if times else None,
                'time_std': statistics.pstdev(times) if times else None,
                'num_results': len(results),
                'embarcacion_tipo': prueba.get('embarcacion_tipo'),
                'tipo': prueba.get('tipo'),
                'sexo': prueba.get('sexo'),
                'categoria': prueba.get('categoria'),
                'embarcacion_num': prueba.get('embarcacion_num'),
            }
            rows.append(row)

    return rows


def train_length_classifier(features, n_splits=5):
    """
    Train and evaluate the selected Extra Trees classifier per prueba.

    Features include:
     - the results (times of the athletes in the prueba):
        - the best time of the prueba
        - the q10, q25, q50, q75, and q90 percentiles
        - the average and standard deviation
        - the number of results in the prueba
     - the embarcacion_tipo of the prueba, if available
     - the tipo of the regata, if available
     - the sexo of the prueba, if available
     - the categoria of the prueba, if available
     - the embarcacion_num of the prueba
    ``regata_id`` is used only to keep pruebas from the same regata together
    during cross-validation. ``prueba_id`` is metadata and is not used as a
    predictor. ``distancia_exacta`` is used only to derive one of the six class
    labels. Rows without a valid distance or supported tipo are excluded.

    Args:
        features (list[dict]): Rows returned by :func:`extract_features`.
        n_splits (int): Maximum number of grouped cross-validation folds.

    Returns:
        Pipeline: The Extra Trees classifier refitted on all labeled rows.
    """

    labeled_rows = []
    targets = []
    groups = []
    for row in features:
        try:
            target = float(row.get('distancia_exacta'))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(target) or target <= 0:
            continue
        target_class = length_class(row.get('tipo'), target)
        if target_class is None:
            continue

        labeled_rows.append(_feature_row(row))
        targets.append(target_class)
        groups.append(str(row.get('regata_id')))

    if not labeled_rows:
        raise ValueError(
            'No rows with a valid distancia_exacta and supported tipo were found'
        )

    time_indices = list(range(len(TIME_FEATURES)))
    other_numeric_indices = list(range(len(TIME_FEATURES), len(NUMERIC_FEATURES)))
    categorical_indices = list(
        range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    )
    preprocessor = ColumnTransformer([
        (
            'times',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('log1p', FunctionTransformer(np.log1p)),
                ('scaler', StandardScaler()),
            ]),
            time_indices,
        ),
        (
            'other_numeric',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            other_numeric_indices,
        ),
        (
            'categorical',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder(handle_unknown='ignore')),
            ]),
            categorical_indices,
        ),
    ])
    classifier = ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=1.0,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    model_template = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier),
    ])

    X = np.asarray(labeled_rows, dtype=object)
    y = np.asarray(targets)
    groups = np.asarray(groups)
    fold_count = min(n_splits, len(np.unique(groups)))

    if fold_count < 2:
        raise ValueError('At least two regatas are required for grouped CV')

    consensus_threshold = 0.90

    def apply_regatta_consensus(predictions, validation_groups, threshold):
        adjusted_predictions = predictions.copy()
        eligible_regattas = 0
        changed_predictions = 0
        for regata_id in np.unique(validation_groups):
            mask = validation_groups == regata_id
            regatta_predictions = predictions[mask]
            majority_class, majority_count = Counter(
                regatta_predictions
            ).most_common(1)[0]
            if majority_count / len(regatta_predictions) < threshold:
                continue
            eligible_regattas += 1
            changed_predictions += np.count_nonzero(
                regatta_predictions != majority_class
            )
            adjusted_predictions[mask] = majority_class
        return adjusted_predictions, eligible_regattas, changed_predictions

    raw_truth = []
    consensus_predictions = []
    eligible_regattas = 0
    changed_predictions = 0
    splitter = GroupKFold(n_splits=fold_count)
    for train_indices, validation_indices in splitter.split(X, y, groups):
        model = clone(model_template)
        model.fit(X[train_indices], y[train_indices])
        fold_predictions = model.predict(X[validation_indices])
        raw_truth.extend(y[validation_indices])
        adjusted, eligible, changed = apply_regatta_consensus(
            fold_predictions,
            groups[validation_indices],
            consensus_threshold,
        )
        consensus_predictions.extend(adjusted)
        eligible_regattas += eligible
        changed_predictions += changed

    short_labels = (
        'super_sprint',
        'sprint',
        'fondo',
        'maraton',
    )
    label_width = max(len(label) for label in short_labels)

    def print_metrics(title, predictions):
        accuracy = accuracy_score(raw_truth, predictions)
        macro_f1 = f1_score(
            raw_truth,
            predictions,
            labels=LENGTH_CLASSES,
            average='macro',
            zero_division=0,
        )
        print(
            f"\n{title} (n={len(raw_truth)}): "
            f"accuracy={accuracy:.2%}, macro F1={macro_f1:.4f}"
        )
        matrix = confusion_matrix(
            raw_truth,
            predictions,
            labels=LENGTH_CLASSES,
        )
        print('Confusion matrix (rows=truth, columns=predicted):')
        print(
            ' ' * (label_width + 3)
            + ' '.join(label.rjust(label_width) for label in short_labels)
        )
        for label, row in zip(short_labels, matrix):
            values = ' '.join(str(value).rjust(label_width) for value in row)
            print(f"  {label.rjust(label_width)} {values}")
        return accuracy, macro_f1


    print_metrics(
        f'ExtraTrees after {consensus_threshold:.0%} regatta consensus',
        consensus_predictions,
    )

    final_model = clone(model_template)
    final_model.fit(X, y)
    return final_model


def fill_length_class(data, model):
    """
    Add a ``length_class`` field to every prueba in ``data``.

    Pruebas with a known ``distancia_exacta`` and a supported ``tipo`` get
    their ground-truth length class; every other prueba gets the class
    predicted by ``model`` (trained on all labeled data). Mutates and
    returns ``data``.
    """
    features = extract_features(data)
    pruebas = [prueba for regata in data for prueba in regata.get('pruebas', [])]

    predict_indices = []
    predict_rows = []
    for i, (prueba, row) in enumerate(zip(pruebas, features)):
        known_class = length_class(row.get('tipo'), row.get('distancia_exacta'))
        if known_class is not None:
            prueba['length_class'] = known_class
            continue
        predict_indices.append(i)
        predict_rows.append(_feature_row(row))

    if predict_rows:
        predictions = model.predict(np.asarray(predict_rows, dtype=object))
        for i, predicted_class in zip(predict_indices, predictions):
            pruebas[i]['length_class'] = predicted_class

    return data


def remove_frequently_repeated_athletes(data, min_repeated_pruebas=50):
    """
    Remove athletes duplicated within at least ``min_repeated_pruebas`` pruebas.

    A prueba counts once for an athlete when that athlete ID occurs at least
    twice among its results. A result is discarded only when every athlete in
    its boat is flagged; mixed boats are preserved unchanged. Pruebas with no
    results left are discarded. Mutates and returns ``data``.
    """
    repeated_pruebas_by_athlete = Counter()
    for regata in data:
        for prueba in regata.get('pruebas', []):
            appearances = Counter(
                athlete_id
                for result in prueba.get('results', [])
                for athlete_id in result.get('lista_palista_id', [])
                if athlete_id not in (None, '')
            )
            repeated_pruebas_by_athlete.update(
                athlete_id
                for athlete_id, count in appearances.items()
                if count >= 2
            )

    removed_athletes = {
        athlete_id
        for athlete_id, num_pruebas in repeated_pruebas_by_athlete.items()
        if num_pruebas >= min_repeated_pruebas
    }

    removed_results = 0
    removed_pruebas = 0
    for regata in data:
        retained_pruebas = []
        for prueba in regata.get('pruebas', []):
            retained_results = []
            for result in prueba.get('results', []):
                athlete_ids = [
                    athlete_id
                    for athlete_id in result.get('lista_palista_id', [])
                    if athlete_id not in (None, '')
                ]
                if athlete_ids and all(
                    athlete_id in removed_athletes for athlete_id in athlete_ids
                ):
                    removed_results += 1
                    continue
                retained_results.append(result)

            if not retained_results:
                removed_pruebas += 1
                continue

            prueba['results'] = retained_results
            tiempos = [_parse_tiempo(result.get('tiempo', '')) for result in retained_results]
            tiempos = [tiempo for tiempo in tiempos if tiempo is not None]
            prueba['n_resultados'] = len(retained_results)
            prueba['tiempo_min'] = _format_tiempo(min(tiempos)) if tiempos else None
            prueba['tiempo_max'] = _format_tiempo(max(tiempos)) if tiempos else None
            prueba['tiempo_medio'] = _format_tiempo(statistics.mean(tiempos)) if tiempos else None
            prueba['tiempo_mediana'] = _format_tiempo(statistics.median(tiempos)) if tiempos else None
            retained_pruebas.append(prueba)

        regata['pruebas'] = retained_pruebas

    flagged = sorted(
        (
            (athlete_id, repeated_pruebas_by_athlete[athlete_id])
            for athlete_id in removed_athletes
        ),
        key=lambda item: (-item[1], str(item[0])),
    )
    print(
        f'Flagged {len(removed_athletes)} athletes; removed {removed_results} results '
        f'whose boats contained only flagged athletes, and {removed_pruebas} empty pruebas'
    )
    for athlete_id, num_pruebas in flagged:
        print(f'  Athlete {athlete_id}: repeated in {num_pruebas} pruebas')

    return data


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def build_dataset():
    print("Extracting raw CSV data...")
    data, header = extract_all_years()
    json_data = convert2json(data, header)
    json_data = additional_info(json_data)
    n_pruebas = sum(len(regata['pruebas']) for regata in json_data)
    print(f"Extracted {n_pruebas} pruebas from {len(json_data)} regatas")

    print("\nEvaluating sexo/tipo/embarcacion_tipo/categoria predictors...")
    athletes = build_athlete_records(json_data)
    profiles = {athlete_id: athlete_profile(a) for athlete_id, a in athletes.items()}
    for field, evaluate in (
        ('sexo', evaluate_sexo),
        ('tipo', evaluate_tipo),
        ('embarcacion_tipo', evaluate_embarcacion),
        ('categoria', evaluate_categoria),
    ):
        accuracy, n, abstentions, _ = evaluate(json_data, profiles)
        accuracy_str = f"{accuracy:.1%}" if accuracy is not None else "n/a"
        print(f"  {field}: accuracy={accuracy_str} (n={n}), abstentions={abstentions}")

    print("\nFilling missing sexo/tipo/embarcacion_tipo/categoria values using bootstrap...")
    filled_data = fill_missing_values(json_data)

    print("\nTraining length_class classifier...")
    features = extract_features(filled_data)
    model = train_length_classifier(features, n_splits=10)

    n_missing_length_class = sum(
        1
        for regata in filled_data
        for prueba in regata['pruebas']
        if length_class(prueba.get('tipo'), prueba.get('distancia_exacta')) is None
    )
    print(f"\nFilling {n_missing_length_class} missing length_class values...")
    filled_data = fill_length_class(filled_data, model)

    print("\nRemoving frequently repeated athlete IDs...")
    filled_data = remove_frequently_repeated_athletes(filled_data)

    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'dataset.json'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filled_data, f, ensure_ascii=False, indent=4)
    print(f"\nWrote final dataset to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    build_dataset()
