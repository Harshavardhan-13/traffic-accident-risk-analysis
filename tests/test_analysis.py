"""Regression checks for the data and evaluation failures found in the coursework."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from src import traffic_analysis as analysis


class AnalysisChecks(unittest.TestCase):
    def test_cleaning_preserves_unknowns_and_deduplicates_full_records(self):
        base = {name: 'UNKNOWN' for name in analysis.SOURCE_COLUMNS}
        base.update({'SeqID': 'stop-a', 'Date Of Stop': '01/01/2020', 'Time Of Stop': '12:30:00',
                     'Accident': 'No', 'Year': '2010', 'Description': 'first charge'})
        changed_charge = {**base, 'Description': 'different charge'}
        unknown_features = {**base, 'SeqID': 'stop-b', 'Year': '2099', 'Work Zone': None}
        invalid_target = {**base, 'SeqID': 'stop-c', 'Accident': 'unclear'}
        missing_id = {**base, 'SeqID': None}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'source.csv'
            pd.DataFrame([base, base, changed_charge, unknown_features, invalid_target, missing_id]).to_csv(raw, index=False)
            with patch.object(analysis, 'ROOT', root), patch.object(analysis, 'PREPARED', root / 'data/prepared.csv'):
                audit = analysis.prepare(raw)
                result = analysis.load_prepared()
            self.assertEqual(audit['raw_rows'], 6)
            self.assertEqual(audit['exact_duplicates_removed'], 1)
            self.assertEqual(audit['invalid_target_rows'], 1)
            self.assertEqual(audit['missing_stop_id_rows'], 1)
            self.assertEqual(len(result), 3)
            self.assertTrue(pd.isna(result.loc[result.stop_id == 'stop-b', 'vehicle_age'].iloc[0]))
            self.assertEqual(result.loc[result.stop_id == 'stop-b', 'work_zone'].iloc[0], 'UNKNOWN')

    def test_stop_ids_do_not_cross_chronological_partitions(self):
        dates = pd.date_range('2020-01-01', periods=240)
        frame = pd.DataFrame({'stop_id': [f'stop-{i}' for i in range(240)],
                              'stop_date': dates, 'accident': [int(i % 10 == 0) for i in range(240)]})
        repeated = frame.iloc[[0]].copy()
        conflict = frame.iloc[[1]].copy()
        conflict['accident'] = 1
        data = pd.concat([frame, repeated, conflict], ignore_index=True)
        parts, metadata = analysis.make_model_sample(data, max_stops=500)
        self.assertEqual(metadata['conflicting_target_stop_ids_excluded'], 1)
        all_ids = [set(part.stop_id) for part in parts.values()]
        self.assertFalse(all_ids[0] & all_ids[1] or all_ids[0] & all_ids[2] or all_ids[1] & all_ids[2])
        self.assertLess(parts['train'].stop_date.max(), parts['validation'].stop_date.min())
        self.assertLess(parts['validation'].stop_date.max(), parts['test'].stop_date.min())

    def test_majority_accuracy_does_not_imply_accident_detection(self):
        target = np.array([0] * 98 + [1] * 2)
        result = analysis.classification_metrics(target, np.zeros(100))
        self.assertEqual(result['accuracy'], 0.98)
        self.assertEqual(result['accident_recall'], 0)
        self.assertEqual(result['roc_auc'], 0.5)
        self.assertEqual(result['average_precision'], 0.02)


if __name__ == '__main__':
    unittest.main()
