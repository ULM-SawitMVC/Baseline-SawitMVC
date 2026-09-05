"""Validate exact paired inference against SciPy and the persisted audit."""
import unittest
import json

import numpy as np
import pandas as pd
from scipy.stats import permutation_test

import a10_revision_audit as audit
import revcommon as rc


class PairedInferenceTests(unittest.TestCase):
    def test_exact_null_matches_enumerated_paired_swaps(self):
        # Includes within-tree class dependence, ties, opposite signs and
        # different weights: independent class resampling would be incorrect.
        for delta in ([1, -2, 3, 0, -1], [0, 0, 0], [1, 1, 1, 1], [4, -4, 2]):
            delta = np.array(delta)
            a = {"ok": np.array([[1] * max(d, 0) + [0] * (4-max(d, 0)) for d in delta])}
            b = {"ok": np.array([[1] * max(-d, 0) + [0] * (4-max(-d, 0)) for d in delta])}
            expected = permutation_test((delta,), np.sum, permutation_type="samples",
                n_resamples=np.inf, alternative="two-sided").pvalue
            self.assertAlmostEqual(audit.exact_class_test(a, b), expected)
            self.assertAlmostEqual(audit.exact_class_test(a, b), audit.exact_class_test(b, a))

    def test_holm_keeps_original_order_and_is_monotone(self):
        np.testing.assert_allclose(audit.holm([.03, .001, .04]), [.06, .003, .06])
        np.testing.assert_allclose(audit.holm([.9, .8]), [1, 1])

    def test_complete_comparison_families(self):
        table = pd.read_csv(rc.OUT_DIR / "a10_exact_paired_tests.csv")
        self.assertEqual(table.groupby("family").size().to_dict(),
            {"condition_gap": 3, "fixed_counters": 10, "fixed_features": 7, "gt_counters": 10})
        fixed = table[table.family == "fixed_counters"]
        self.assertTrue((fixed.class_p_exact_holm > .05).all())

    def test_cv_pairs_share_targets_and_cover_every_tree_once_per_repeat(self):
        table = pd.read_csv(rc.OUT_DIR / "a10_matched_cv_predictions.csv")
        index = ["pool", "repeat", "fold", "tree_id"]
        truth = ["true_" + c for c in rc.CLASSES]
        gt = table[table.condition == "gt"].set_index(index)[truth].sort_index()
        fixed = table[table.condition == "fixed"].set_index(index)[truth].sort_index()
        pd.testing.assert_frame_equal(gt, fixed)
        self.assertFalse(table.duplicated(["pool", "repeat", "condition", "tree_id"]).any())
        for pool, n in [("nontraining237", 237), ("testonly141", 141)]:
            self.assertTrue((table[table.pool == pool].groupby(["repeat", "condition"]).size() == n).all())

    def test_validation_selection_ignores_test_ranking(self):
        candidates = pd.read_csv(rc.OUT_DIR / 'a11_selection_candidates.csv')
        chosen = json.loads((rc.OUT_DIR / 'a11_validation_selected.json').read_text())
        for condition in ['gt', 'fixed']:
            part = candidates[candidates.condition == condition]
            self.assertEqual(len(part), 40)
            winner = part.sort_values(['val_macro','val_mae','n_dim','model','features'],
                ascending=[False,True,True,True,True]).iloc[0]
            self.assertEqual((winner.model,winner.features),
                (chosen[condition]['model'],chosen[condition]['features']))
            # Selection remains identical even if test metrics are reversed.
            altered = part.copy()
            altered['test_macro'] = 1-altered['test_macro']
            other = altered.sort_values(['val_macro','val_mae','n_dim','model','features'],
                ascending=[False,True,True,True,True]).iloc[0]
            self.assertEqual((winner.model,winner.features),(other.model,other.features))

    def test_selected_prediction_metrics_match_report(self):
        selected = pd.read_csv(rc.OUT_DIR / 'a11_selected_predictions.csv')
        report = json.loads((rc.OUT_DIR / 'a11_validation_selected.json').read_text())
        for condition in ['gt','fixed']:
            rows = selected[selected.condition == condition]
            self.assertEqual(len(rows), 141)
            score = rc.score(rows[['true_'+c for c in rc.CLASSES]].to_numpy(),
                             rows[['pred_'+c for c in rc.CLASSES]].to_numpy())
            for key in ['macro','joint','mae','total_mae']:
                self.assertAlmostEqual(score[key], report[condition]['test_'+key])


if __name__ == "__main__":
    unittest.main()
