import math
import os
import statistics
import json
from collections import Counter

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


LENGTH_CLASSES = (
    'aguas_tranquilas_0_500',
    'aguas_tranquilas_501_2000',
    'aguas_tranquilas_2001_8000',
    'aguas_tranquilas_over_8000',
    'mar_0_8000',
    'mar_over_8000',
)


def length_class(tipo, distance):
    """Map a distance in metres to its tipo-specific length class."""
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
        return LENGTH_CLASSES[4] if distance <= 8000 else LENGTH_CLASSES[5]
    return None


def load_data_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
    """
    with open(file_path, encoding='utf-8') as file:
        return json.load(file)

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

    time_features = (
        'best_time',
        'time_q10',
        'time_q25',
        'time_q50',
        'time_q75',
        'time_q90',
        'average_time',
        'time_std',
    )
    other_numeric_features = (
        'num_results',
        'embarcacion_num',
    )
    numeric_features = time_features + other_numeric_features
    categorical_features = (
        'embarcacion_tipo',
        'tipo',
        'sexo',
        'categoria',
    )

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

        labeled_rows.append([
            *(
                row.get(name) if row.get(name) is not None else np.nan
                for name in numeric_features
            ),
            *(
                row.get(name) if row.get(name) is not None else np.nan
                for name in categorical_features
            ),
        ])
        targets.append(target_class)
        groups.append(str(row.get('regata_id')))

    if not labeled_rows:
        raise ValueError(
            'No rows with a valid distancia_exacta and supported tipo were found'
        )

    time_indices = list(range(len(time_features)))
    other_numeric_indices = list(range(len(time_features), len(numeric_features)))
    categorical_indices = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
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

    consensus_thresholds = (0.90,)

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
    raw_predictions = []
    consensus_predictions = {threshold: [] for threshold in consensus_thresholds}
    eligible_regattas = Counter()
    changed_predictions = Counter()
    splitter = GroupKFold(n_splits=fold_count)
    for train_indices, validation_indices in splitter.split(X, y, groups):
        model = clone(model_template)
        model.fit(X[train_indices], y[train_indices])
        fold_predictions = model.predict(X[validation_indices])
        raw_truth.extend(y[validation_indices])
        raw_predictions.extend(fold_predictions)
        for threshold in consensus_thresholds:
            adjusted, eligible, changed = apply_regatta_consensus(
                fold_predictions,
                groups[validation_indices],
                threshold,
            )
            consensus_predictions[threshold].extend(adjusted)
            eligible_regattas[threshold] += eligible
            changed_predictions[threshold] += changed

    short_labels = (
        'AT<=500',
        'AT501-2000',
        'AT2001-8000',
        'AT>8000',
        'MAR<=8000',
        'MAR>8000',
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
        f'ExtraTrees grouped {fold_count}-fold CV',
        raw_predictions,
    )
    for threshold in consensus_thresholds:
        print(
            f"Regatta consensus threshold {threshold:.0%}: "
            f"eligible regattas={eligible_regattas[threshold]}, "
            f"changed pruebas={changed_predictions[threshold]}"
        )
        print_metrics(
            f'ExtraTrees after {threshold:.0%} regatta consensus',
            consensus_predictions[threshold],
        )

    final_model = clone(model_template)
    final_model.fit(X, y)
    return final_model

if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'filled_v1.json'
    )
    data = load_data_json(data_path)
    features = extract_features(data)
    print(f"Extracted {len(features)} feature rows.\n")

    model = train_length_classifier(features, n_splits=10)
