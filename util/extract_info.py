import csv
import os

def read_data_csv(file_path):
    """
    return a list of lists, where each inner list represents a row in the CSV file.
    it also returns a second list with the header of the CSV file.
    """
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    return data, header

def extract_all_years(years=None):
    files = os.listdir('../data/')
    data = []
    for file in files:
        if file.endswith('.csv'):
            year = file.split('.')[0]
            if years is None or year in years:
                file_path = os.path.join('../data/', file)
                year_data, header = read_data_csv(file_path)
                data.extend(year_data)
    return data, header

def convert2json(data, header):
    """
    Given a list of data rows and the corresponding header, this function 
    reorganizes hierarchicaly the data into a list of dictionaries. each dictionary represents
    a "regata" and contains regata info. then it contains a number of different pruebas with their corresponding info and results. 
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

def additional_info(json_data):
    """
    Extract for each prueba the following information:
    - distancia_exacta: 500, 1000, 2000, 5000, 10000
    - tipo: sprint, fondo, maraton, mar, dragon
    - sexo: masculino, femenino, mixto
    - embarcacion_tipo: kayak, canoa, outrigger, surfski, dragon_boat, vaa, paddleboard, paleo
    - embarcacion_num: number of paddlers per boat (1, 2, 4, 12, ...)
    - categoria: infantil, cadete, senior, etc
    - n_resultados, tiempo_min, tiempo_max, tiempo_medio, tiempo_mediana:
      finish-time stats over the prueba's results (HH:MM:SS.ff)

    Only pruebas whose tipo is fondo/sprint/maraton/mar/dragon are kept; the
    rest (aguas bravas, slalom, descenso, paracanoe) are discarded.
    """
    import re
    import unicodedata
    import statistics
    from collections import Counter

    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

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
        'PROMESA': 'promesa',
        'OPEN': 'open',
        'ABSOLUTA': 'absoluta', 'ABSOLUTO': 'absoluta', 'ABSOLUT': 'absoluta', 'ABS': 'absoluta',
        'ABRIDORES': 'abridor', 'ABRIDOR': 'abridor', 'ABRIODORES': 'abridor', 'ABR': 'abridor',
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
        ('aguas bravas', 'aguas_bravas'), ('augas bravas', 'aguas_bravas'), ('ur biziak', 'aguas_bravas'),
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
            cm = cat_pattern.search(upper)
            if cm:
                token = cm.group(1).upper()
                categoria = cat_map.get(token, 'sub23' if re.match(r'S\d{2}', token) else token.lower())
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
            # liga/modalidad (or a name keyword fallback for mar/dragon);
            # everything else (sprint/fondo/maraton/dragon) is aguas_tranquilas
            if tipo_from_liga:
                tipo = tipo_from_liga
            elif 'dragon' in low:
                tipo = 'aguas_tranquilas'
            elif 'mar' in low.split():
                tipo = 'mar'
            else:
                tipo = 'aguas_tranquilas'

            prueba['tipo'] = tipo

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

    # blank out categorias rarer than prebenjamin (i.e. open and below) - too
    # sparse/noisy to be meaningful (typos, one-off codes, misc classifications)
    cat_counts = Counter(
        p['categoria'] for reg in json_data for p in reg['pruebas'] if p['categoria']
    )
    threshold = cat_counts.get('prebenjamin', 0)
    for reg in json_data:
        for prueba in reg['pruebas']:
            if prueba['categoria'] and cat_counts[prueba['categoria']] < threshold:
                prueba['categoria'] = None

    return [reg for reg in json_data if reg['pruebas']]

def write_csv(json_data, file_path):
    """
    Flatten json_data to one row per prueba (regata + prueba info, no
    per-participant rows) for manual inspection of the extracted fields.
    """
    fieldnames = [
        'fecha', 'organizador_id', 'organizador_name',
        'regata_id', 'regata_name', 'regata_liga',
        'prueba_id', 'prueba_name', 'prueba_fase', 'prueba_serie',
        'distancia_exacta', 'tipo', 'sexo', 'categoria',
        'embarcacion_tipo', 'embarcacion_num',
        'n_resultados', 'tiempo_min', 'tiempo_max', 'tiempo_medio', 'tiempo_mediana',
    ]
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for reg in json_data:
            for prueba in reg['pruebas']:
                row = {
                    'fecha': reg['fecha'],
                    'organizador_id': reg['organizador_id'],
                    'organizador_name': reg['organizador_name'],
                    'regata_id': reg['regata_id'],
                    'regata_name': reg['regata_name'],
                    'regata_liga': reg['regata_liga'],
                }
                row.update({k: prueba.get(k) for k in fieldnames if k not in row})
                writer.writerow(row)

if __name__ == "__main__":
    # Example usage
    data, header = extract_all_years()
    json_data = convert2json(data, header)
    json_data = additional_info(json_data)

    with open('output.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(json_data, f, indent=4)

    write_csv(json_data, 'output.csv')