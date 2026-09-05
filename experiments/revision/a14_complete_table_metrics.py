"""Complete missing Table III/VI entries from existing annotation/detector caches.

No detector inference, test-set tuning, or changes to historical artifacts.
The global divisor is fitted separately per input condition using train trees.
Threshold recall uses the actual a9 low-confidence cache filtered before matching.
Run: python experiments/revision/a14_complete_table_metrics.py
"""
from collections import Counter
from functools import lru_cache
import json

import numpy as np
import pandas as pd

import revcommon as rc
from a2_error_decomposition import load_json, match_tree
from a4_detector_sensitivity import old_split_map
from a9_low_threshold_sweep import LOW_DIR, THRESHOLDS


@lru_cache(maxsize=None)
def read(path):
    return load_json(path)


def annotations(tree):
    return [a for side in tree['images'].values() for a in side['annotations']
            if a['class_name'] in rc.CLASSES]


def filter_tree(tree, threshold):
    return dict(tree, images={key:dict(side, annotations=[a for a in side['annotations']
        if a.get('conf',1.) >= threshold]) for key,side in tree['images'].items()})


def appearance_metrics(source, ids, threshold=None):
    confusion = np.zeros((5,5),dtype=int)
    n_detections = 0
    for tid in ids:
        gt, pred = read(rc.GT_DIR/f'{tid}.json'), read(source/f'{tid}.json')
        assert set(gt['images']) == set(pred['images'])
        if threshold is not None:
            pred = filter_tree(pred,threshold)
        n_detections += len(annotations(pred))
        pairs, fps, fns = match_tree(gt,pred)
        for _,_,g,p,_ in pairs:
            confusion[rc.CLASSES.index(g),rc.CLASSES.index(p)] += 1
        for _,p,_ in fps:
            confusion[4,rc.CLASSES.index(p)] += 1
        for _,_,g in fns:
            confusion[rc.CLASSES.index(g),4] += 1
    assert n_detections == confusion[:,:4].sum()
    denominators = confusion[:4,:].sum(axis=1)
    assert (denominators > 0).all()
    recalls = np.diag(confusion[:4,:4])/denominators
    return dict(detections=n_detections,gt_appearances=int(denominators.sum()),
        recall_macro=float(recalls.mean()),
        **{f'recall_{c}':float(v) for c,v in zip(rc.CLASSES,recalls)},
        confusion=confusion.tolist())


def global_divisors(manifest):
    output, predictions = {}, []
    ids = sorted(manifest.index)
    train = np.array([manifest.loc[t,'new_split']=='train' for t in ids])
    test = np.array([manifest.loc[t,'new_split']=='test' for t in ids])
    assert train.sum()==716 and test.sum()==141
    truth = np.array([[read(rc.GT_DIR/f'{t}.json')['summary']['by_class'].get(c,0)
                       for c in rc.CLASSES] for t in ids],dtype=int)
    for condition,source in [('gt',rc.GT_DIR),('fixed',rc.PRED_DIR)]:
        counts = [Counter(a['class_name'] for a in annotations(read(source/f'{t}.json'))) for t in ids]
        naive = np.array([[v[c] for c in rc.CLASSES] for v in counts])
        k = float(naive[train].sum()/truth[train].sum())
        pred = rc.round_counts(naive[test]/k)
        stats = rc.per_tree_stats(truth[test],pred)
        metrics = rc.metrics_from_stats(stats)
        output[condition] = dict(k=k,train_appearances=int(naive[train].sum()),
            train_unique_bunches=int(truth[train].sum()),metrics=metrics,
            ci=rc.bootstrap_ci(stats,['macro','joint','mae']))
        for tid,y,p in zip(np.array(ids)[test],truth[test],pred):
            predictions.append(dict(condition=condition,tree_id=tid,
                **{f'true_{c}':int(v) for c,v in zip(rc.CLASSES,y)},
                **{f'pred_{c}':int(v) for c,v in zip(rc.CLASSES,p)}))
    historical=json.loads((rc.OUT_DIR/'a7_heuristics_ci.json').read_text())
    assert np.isclose(output['gt']['k'],historical['k_global'],rtol=0,atol=1e-12)
    for key,value in historical['global divisor']['point'].items():
        assert np.isclose(output['gt']['metrics'][key],value,rtol=0,atol=1e-12),key
    pd.DataFrame(predictions).to_csv(rc.OUT_DIR/'a14_global_divisor_predictions.csv',index=False)
    return output


def main():
    manifest=pd.read_csv(rc.GT_DIR.parent/'split_manifest.csv').set_index('tree_id')
    test=sorted(manifest.index[manifest.new_split=='test'])
    old=old_split_map()
    common=sorted(t for t in manifest.index if manifest.loc[t,'new_split'] in ['val','test']
                  and old[t] in ['val','test'])
    assert len(common)==64
    capacity=pd.read_csv(rc.OUT_DIR/'a4_detector_capacity.csv')
    rows=[]
    for label,source in [('YOLO26n (old protocol)',rc.ARCHIVE_PRED/'y26n_per_tree'),
                         ('YOLO26s (old protocol)',rc.ARCHIVE_PRED/'y26s_per_tree'),
                         ('YOLO26m (old protocol)',rc.ARCHIVE_PRED/'y26m_per_tree'),
                         ('YOLO26m (y26mv2, released)',rc.PRED_DIR)]:
        metrics=appearance_metrics(source,common)
        previous=capacity.loc[(capacity.detector==label)&(capacity.features=='F0')&(capacity.model=='Ridge')].iloc[0]
        assert np.isclose(metrics['recall_macro'],previous.recall_macro,rtol=0,atol=1e-12)
        rows.append(dict(block='checkpoint',setting=label,n_trees=len(common),**metrics))
    for block,ids in [('family',test),('checkpoint',common)]:
        n=sum(len(annotations(read(rc.GT_DIR/f'{t}.json'))) for t in ids)
        rows.append(dict(block=block,setting='GT detections',n_trees=len(ids),
                         detections=n,gt_appearances=n,recall_macro=1.0))
    sweep=pd.read_csv(rc.OUT_DIR/'a9_threshold_sweep_full.csv')
    for threshold in THRESHOLDS:
        metrics=appearance_metrics(LOW_DIR,test,threshold)
        previous=sweep.loc[(sweep.threshold==threshold)&(sweep.features=='F_all')&(sweep.model=='Ridge')].iloc[0]
        assert metrics['detections']==int(previous.test_detections)
        assert metrics['gt_appearances']==2612
        rows.append(dict(block='threshold',setting=float(threshold),n_trees=len(test),**metrics))
        print('threshold',threshold,'detections',metrics['detections'],'recall',metrics['recall_macro'],flush=True)
    report=dict(global_divisor=global_divisors(manifest),appearance_metrics=rows,
        protocol='Divisor = training appearances / training unique bunches, fitted separately per input condition. GT detection counts are annotation appearances. Recall = macro class-correct appearance recall, greedy class-agnostic IoU>=0.5 matching after filtering the actual threshold cache.')
    (rc.OUT_DIR/'a14_completed_table_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in row.items() if k!='confusion'} for row in rows]).to_csv(
        rc.OUT_DIR/'a14_appearance_metrics.csv',index=False)
    print('Divisor results:',json.dumps(report['global_divisor'],indent=2),flush=True)


if __name__=='__main__':
    main()
