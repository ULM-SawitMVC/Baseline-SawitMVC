"""Choose model/features on validation, then report held-out test performance.

The deterministic selection rule is descending validation Class +/-1, then
ascending validation macro MAE, feature dimension, and lexical model name.
No training fit uses validation or test labels. All candidates use the released
model builders; selection is a new sensitivity check, not a fresh unseen test.
Run: python experiments/revision/a11_validation_selection.py
"""
from itertools import product

import numpy as np
import pandas as pd

import revcommon as rc
from a10_revision_audit import comparison


def main():
    rows, predictions, chosen_stats, selected = [], [], {}, {}
    for condition, source in [('gt', rc.GT_DIR), ('fixed', rc.PRED_DIR)]:
        df, y, ids, splits, _ = rc.load_dataset(source)
        features = rc.feature_sets(df)
        tr, va, te = splits == 'train', splits == 'val', splits == 'test'
        models = rc.model_builders()
        candidates = []
        for feature, name in product(features, models):
            x = df[features[feature]].to_numpy(float)
            model = models[name]()
            model.fit(x[tr], y[tr])
            pv = rc.round_counts(model.predict(x[va]))
            pt = rc.round_counts(model.predict(x[te]))
            mv, mt = rc.score(y[va], pv), rc.score(y[te], pt)
            row = dict(condition=condition, features=feature, model=name,
                       n_dim=x.shape[1], **{'val_'+k: v for k,v in mv.items()},
                       **{'test_'+k: v for k,v in mt.items()})
            rows.append(row)
            candidates.append(((-mv['macro'], mv['mae'], x.shape[1], name, feature), row, pt))
            print(condition, feature, name, f"val={mv['macro']*100:.2f}", flush=True)
        _, winner, pred = min(candidates, key=lambda v: v[0])
        selected[condition] = winner
        chosen_stats[condition] = rc.per_tree_stats(y[te], pred)
        for tid, truth, predicted in zip(np.array(ids)[te], y[te], pred):
            predictions.append(dict(condition=condition, tree_id=tid,
                **{f'true_{c}': int(v) for c,v in zip(rc.CLASSES, truth)},
                **{f'pred_{c}': int(v) for c,v in zip(rc.CLASSES, predicted)}))
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rc.OUT_DIR/'a11_selection_candidates.csv', index=False)
    pd.DataFrame(predictions).to_csv(rc.OUT_DIR/'a11_selected_predictions.csv', index=False)
    selected['paired_test'] = comparison(chosen_stats['gt'], chosen_stats['fixed'])
    selected['selection_rule'] = 'val Class +/-1 descending, val macro MAE ascending, dimension ascending, model/features lexical'
    selected['caveat'] = 'Same historical test data; no claim of a newly collected independent test. Original model builders retained.'
    rc.dump(selected, rc.OUT_DIR/'a11_validation_selected.json')
    print(selected, flush=True)


if __name__ == '__main__':
    main()
