"""Reproducible record analysis and stop-level accident-label benchmarking."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://catalog.data.gov/dataset/traffic-violations'
PREPARED = ROOT / 'data' / 'traffic_prepared.csv'
NUMERIC_FEATURES = ['hour', 'weekday', 'month', 'vehicle_age']
CATEGORICAL_FEATURES = ['subagency', 'vehicle_type', 'alcohol', 'work_zone', 'commercial_vehicle']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SOURCE_COLUMNS = ['SeqID', 'Date Of Stop', 'Time Of Stop', 'Accident', 'SubAgency',
                  'VehicleType', 'Year', 'Alcohol', 'Work Zone', 'Commercial Vehicle',
                  'Violation Type', 'Charge', 'Contributed To Accident']


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(raw_path: Path | None = None) -> dict:
    """Read the raw CSV in chunks; retain unknown features rather than delete rows."""
    raw_path = Path(raw_path) if raw_path else ROOT / 'Traffic_Violations.csv'
    if not raw_path.exists():
        raise FileNotFoundError(f'Download the CSV from {SOURCE_URL} and save it as {raw_path}. See data/README.md.')
    chunks, seen = [], set()
    counts = {'raw_rows': 0, 'exact_duplicates_removed': 0, 'invalid_target_rows': 0,
              'missing_stop_id_rows': 0, 'invalid_date_rows': 0, 'invalid_vehicle_year_rows': 0}
    for raw in pd.read_csv(raw_path, dtype='string', chunksize=100_000):
        missing = set(SOURCE_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f'Source schema changed; missing required columns: {sorted(missing)}')
        counts['raw_rows'] += len(raw)
        # Hash every original column, not just the retained features, for exact deduplication.
        hashes = pd.util.hash_pandas_object(raw, index=False)
        duplicate = hashes.duplicated() | hashes.isin(seen)
        seen.update(hashes.tolist())
        counts['exact_duplicates_removed'] += int(duplicate.sum())
        frame = raw.loc[~duplicate, SOURCE_COLUMNS].copy()
        accident = frame['Accident'].str.strip().str.lower().map({'yes': 1, 'no': 0})
        counts['invalid_target_rows'] += int(accident.isna().sum())
        frame = frame.loc[accident.notna()].copy()
        frame['accident'] = accident.loc[frame.index].astype('int8')
        stop_id = frame['SeqID'].str.strip()
        missing_id = stop_id.isna() | stop_id.eq('')
        counts['missing_stop_id_rows'] += int(missing_id.sum())
        frame = frame.loc[~missing_id].copy()
        dates = pd.to_datetime(frame['Date Of Stop'], format='mixed', errors='coerce')
        counts['invalid_date_rows'] += int(dates.isna().sum())
        frame['stop_id'] = frame['SeqID'].str.strip()
        frame['stop_date'] = dates.dt.strftime('%Y-%m-%d')
        times = pd.to_datetime(frame['Time Of Stop'], format='%H:%M:%S', errors='coerce')
        frame['hour'] = times.dt.hour
        frame['weekday'] = dates.dt.dayofweek
        frame['month'] = dates.dt.month
        vehicle_year = pd.to_numeric(frame['Year'], errors='coerce')
        invalid_year = vehicle_year.notna() & ((vehicle_year < 1900) | (vehicle_year > dates.dt.year + 1))
        counts['invalid_vehicle_year_rows'] += int(invalid_year.sum())
        frame['vehicle_age'] = dates.dt.year - vehicle_year.mask(invalid_year)
        mapping = {'SubAgency': 'subagency', 'VehicleType': 'vehicle_type', 'Alcohol': 'alcohol',
                   'Work Zone': 'work_zone', 'Commercial Vehicle': 'commercial_vehicle',
                   'Violation Type': 'violation_type', 'Charge': 'charge'}
        for old, new in mapping.items():
            frame[new] = frame[old].str.strip().str.upper().fillna('UNKNOWN').replace('', 'UNKNOWN')
        frame['contributed_to_accident'] = frame['Contributed To Accident'].str.strip().str.lower().map(
            {'true': 1, 'false': 0, 'yes': 1, 'no': 0})
        chunks.append(frame[['stop_id', 'stop_date', 'accident'] + FEATURES +
                            ['violation_type', 'charge', 'contributed_to_accident']])
    data = pd.concat(chunks, ignore_index=True)
    PREPARED.parent.mkdir(exist_ok=True)
    data.to_csv(PREPARED, index=False)
    counts.update({'prepared_records': len(data), 'distinct_stop_ids': int(data.stop_id.nunique()),
                   'accident_records': int(data.accident.sum()), 'accident_record_rate': float(data.accident.mean()),
                   'date_min': str(data.stop_date.min()), 'date_max': str(data.stop_date.max()),
                   'input_file': raw_path.name, 'input_bytes': raw_path.stat().st_size,
                   'source_url': SOURCE_URL, 'grain': 'violation record; multiple records may share a stop ID'})
    with raw_path.open('rb') as handle:
        counts['input_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    save_json(ROOT / 'reports' / 'data_audit.json', counts)
    return counts


def load_prepared() -> pd.DataFrame:
    if not PREPARED.exists():
        raise FileNotFoundError('Run python scripts/run_analysis.py --prepare-only first. See data/README.md.')
    return pd.read_csv(PREPARED, dtype={'stop_id': 'string'}, parse_dates=['stop_date'])


def group_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    result = data.groupby(column, dropna=False).agg(records=('accident', 'size'),
                                                  accident_records=('accident', 'sum')).reset_index()
    result['accident_rate_pct'] = 100 * result.accident_records / result.records
    return result


def plot_eda(data: pd.DataFrame):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    hourly = group_summary(data.dropna(subset=['hour']), 'hour')
    types = group_summary(data, 'violation_type').sort_values('records', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    axes[0].bar(hourly.hour, hourly.accident_rate_pct, color='#2166ac')
    axes[0].set(xlabel='Hour of stop', ylabel='Records marked Accident = Yes (%)', xticks=range(0, 24, 3),
                title='Recorded accident association by hour')
    axes[1].barh(types.violation_type, types.records / 1000, color='#287d8e')
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Violation records (thousands)', title='Recorded disposition types')
    fig.suptitle(f'Montgomery County | {len(data):,} prepared violation records', fontsize=14)
    fig.text(0.01, -0.05, 'Multiple records may describe one stop. Counts reflect enforcement records, not driving exposure.', fontsize=10)
    (ROOT / 'assets').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'assets' / 'traffic-overview.png', dpi=160, bbox_inches='tight')
    return fig


def make_model_sample(data: pd.DataFrame, max_stops: int = 200_000, seed: int = 42):
    """One consistent-labeled record per stop; sample stops before chronological splitting."""
    targets = data.groupby('stop_id').accident.nunique()
    consistent_ids = targets[targets == 1].index
    eligible = data[data.stop_id.isin(consistent_ids)].dropna(subset=['stop_date'])
    stops = eligible.drop_duplicates('stop_id').copy()
    if len(stops) > max_stops:
        stops = stops.sample(max_stops, random_state=seed)
    stops = stops.sort_values(['stop_date', 'stop_id']).reset_index(drop=True)
    # Keep an entire calendar day in one partition.
    train_end = stops.stop_date.quantile(0.6).normalize()
    val_end = stops.stop_date.quantile(0.8).normalize()
    partitions = {'train': stops[stops.stop_date <= train_end].copy(),
                  'validation': stops[(stops.stop_date > train_end) & (stops.stop_date <= val_end)].copy(),
                  'test': stops[stops.stop_date > val_end].copy()}
    sets = {name: set(frame.stop_id) for name, frame in partitions.items()}
    assert not (sets['train'] & sets['test'] or sets['train'] & sets['validation'] or sets['test'] & sets['validation'])
    for frame in partitions.values():
        if frame.accident.nunique() != 2:
            raise ValueError('Every chronological partition needs both classes; increase max_stops or inspect the data.')
    meta = {'sampling_unit': 'one stop ID with a consistent recorded Accident label',
            'max_stops': max_stops, 'sampled_stops': len(stops), 'seed': seed,
            'conflicting_target_stop_ids_excluded': int((targets > 1).sum()),
            'first_record_policy': 'first source record supplies features; charge and violation disposition are excluded',
            'split': 'chronological 60/20/20 using full calendar days',
            'train_end': str(train_end.date()), 'validation_end': str(val_end.date()),
            'features': FEATURES,
            'excluded_features': ['Race', 'Gender', 'Personal Injury', 'Fatal', 'Property Damage',
                                  'Contributed To Accident', 'Description', 'Charge', 'Violation Type', 'SeqID'],
            'partitions': {name: {'stops': len(frame), 'positive_stops': int(frame.accident.sum()),
                                 'positive_rate': float(frame.accident.mean()),
                                 'date_min': str(frame.stop_date.min().date()),
                                 'date_max': str(frame.stop_date.max().date())} for name, frame in partitions.items()}}
    return partitions, meta


def classification_metrics(y, scores, threshold=0.5):
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'accuracy': float(accuracy_score(y, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'accident_precision': float(precision_score(y, predicted, zero_division=0)),
            'accident_recall': float(recall_score(y, predicted, zero_division=0)),
            'accident_f1': float(f1_score(y, predicted, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def benchmark(data: pd.DataFrame, max_stops: int = 200_000, seed: int = 42):
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import sklearn
    partitions, metadata = make_model_sample(data, max_stops, seed)
    train, val, test = (partitions[k] for k in ['train', 'validation', 'test'])
    estimators = {
        'Majority baseline': DummyClassifier(strategy='most_frequent'),
        'Single-tree benchmark': RandomForestClassifier(n_estimators=1, max_depth=14, min_samples_leaf=20, random_state=seed, n_jobs=2),
        'Balanced logistic regression': LogisticRegression(class_weight='balanced', max_iter=600, random_state=seed),
        'Balanced random forest': RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=30,
                                                       class_weight='balanced_subsample', max_samples=0.8,
                                                       random_state=seed, n_jobs=2),
        'Balanced histogram boosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                                                     learning_rate=0.06, l2_regularization=10,
                                                                     class_weight='balanced', random_state=seed),
    }
    fitted, validation_rows, test_rows = {}, [], []
    for name, estimator in estimators.items():
        print(f'Training {name}...', flush=True)
        preprocess = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC_FEATURES),
            ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30, sparse_output=False), CATEGORICAL_FEATURES),
        ])
        pipeline = Pipeline([('preprocess', preprocess), ('model', estimator)])
        pipeline.fit(train[FEATURES], train.accident)
        scores = pipeline.predict_proba(val[FEATURES])[:, 1]
        validation_rows.append({'model': name, **classification_metrics(val.accident, scores)})
        fitted[name] = pipeline
    validation = pd.DataFrame(validation_rows)
    # All selection happens before test predictions are evaluated.
    selected = validation[validation.model != 'Majority baseline'].sort_values('average_precision', ascending=False).iloc[0].model
    selected_val_scores = fitted[selected].predict_proba(val[FEATURES])[:, 1]
    precision, recall, thresholds = precision_recall_curve(val.accident, selected_val_scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f1)])
    score_arrays = {}
    for name, pipeline in fitted.items():
        scores = pipeline.predict_proba(test[FEATURES])[:, 1]
        test_rows.append({'model': name, **classification_metrics(test.accident, scores)})
        score_arrays[name] = scores
    tuned_name = f'{selected} (validation threshold)'
    test_rows.append({'model': tuned_name, **classification_metrics(test.accident, score_arrays[selected], threshold)})
    score_arrays[tuned_name] = score_arrays[selected]
    test_results = pd.DataFrame(test_rows)
    metadata.update({'selected_model': selected, 'selection_metric': 'validation average precision',
                     'selected_threshold': threshold, 'threshold_rule': 'maximize accident-class F1 on validation only',
                     'test_used_for_selection': False, 'python_version': platform.python_version(),
                     'sklearn_version': sklearn.__version__, 'pandas_version': pd.__version__,
                     'data_audit': json.loads((ROOT / 'reports/data_audit.json').read_text()),
                     'model_parameters': {name: est.get_params() for name, est in estimators.items()}})
    (ROOT / 'reports').mkdir(exist_ok=True)
    validation.to_csv(ROOT / 'reports/validation_metrics.csv', index=False)
    test_results.to_csv(ROOT / 'reports/test_metrics.csv', index=False)
    save_json(ROOT / 'reports/model_run.json', metadata)
    return test_results, validation, metadata, test.accident.to_numpy(), score_arrays


def plot_evaluation(results, metadata, y, scores):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    selected = metadata['selected_model']
    tuned = results.iloc[-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    for name in ['Majority baseline', selected]:
        fpr, tpr, _ = roc_curve(y, scores[name])
        precision, recall, _ = precision_recall_curve(y, scores[name])
        axes[0].plot(fpr, tpr, label=name, linewidth=2)
        if name == 'Majority baseline':
            axes[1].axhline(float(np.mean(y)), label=f'Constant-score baseline (AP={np.mean(y):.3f})',
                            linewidth=2, linestyle='--')
        else:
            axes[1].step(recall, precision, where='post', label=name, linewidth=2)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Untouched chronological test: ROC')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Accident recall', ylabel='Accident precision', title='Precision-recall trade-off')
    axes[1].legend(fontsize=8)
    matrix = np.array([[tuned.tn, tuned.fp], [tuned.fn, tuned.tp]], dtype=int)
    axes[2].imshow(matrix, cmap='Blues')
    for row in range(2):
        for col in range(2):
            axes[2].text(col, row, f'{matrix[row,col]:,}', ha='center', va='center',
                         color='white' if matrix[row,col] > matrix.max()/2 else 'black')
    axes[2].set(xticks=[0,1], yticks=[0,1], xticklabels=['No accident','Accident'],
                yticklabels=['No accident','Accident'], xlabel='Predicted label', ylabel='Recorded label',
                title='Selected model: validation-set threshold')
    fig.suptitle(f'{selected} | {len(y):,} held-out stops', fontsize=14)
    fig.savefig(ROOT / 'assets/model-evaluation.png', dpi=150, bbox_inches='tight')
    return fig
