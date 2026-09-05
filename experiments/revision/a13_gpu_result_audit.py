"""Audit committed a12 GPU artifacts without retraining or overwriting them.

Run: python experiments/revision/a13_gpu_result_audit.py
Recomputes Ridge+F0 fits, all metrics, paired inference, and appearance matching.
Loads the author's trusted checkpoint to verify hash and training metadata.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

import revcommon as rc
import a12_second_detector as a12


def main():
    output = rc.OUT_DIR
    provenance = json.loads((output/'a12_provenance.json').read_text())
    weights = rc.ROOT/'models/yolo/yolo11m_revision_306f9394.pt'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == provenance['weights_sha256']
    predictions = rc.ROOT/'predictions'/f'yolo11m_revision_{digest[:12]}_per_tree'
    manifest = pd.read_csv(rc.GT_DIR.parent/'split_manifest.csv').set_index('tree_id')
    assert {p.stem for p in predictions.glob('*.json')} == set(manifest.index)
    counts = {s:dict(trees=0,images=0,detections=0) for s in ['train','val','test']}
    for tid, row in manifest.iterrows():
        pred = json.loads((predictions/f'{tid}.json').read_text())
        gt = json.loads((rc.GT_DIR/f'{tid}.json').read_text(encoding='utf-8-sig'))
        assert pred['tree_name'] == tid and pred['split'] == row.new_split
        assert set(pred['images']) == set(gt['images'])
        assert pred['inference_settings'] == dict(weights_sha256=digest,conf=.25,imgsz=640)
        item = counts[row.new_split]
        item['trees'] += 1
        item['images'] += len(pred['images'])
        for side in pred['images'].values():
            for ann in side['annotations']:
                assert ann['class_name'] in rc.CLASSES and .25 <= ann['conf'] <= 1
                assert np.isfinite(ann['bbox_yolo']).all()
                item['detections'] += 1
    assert [counts[s]['trees'] for s in counts] == [716,96,141]
    assert [counts[s]['images'] for s in counts] == [3000,404,588]
    # Save reproduced a12 outputs separately, retaining the committed GPU results.
    rc.OUT_DIR = rc.ROOT/'.cache/a13-reproduced'
    rc.OUT_DIR.mkdir(parents=True,exist_ok=True)
    a12.evaluate(predictions)
    original = json.loads((output/'a12_second_detector_metrics.json').read_text())
    rebuilt = json.loads((rc.OUT_DIR/'a12_second_detector_metrics.json').read_text())
    for condition, metrics in original.items():
        for key, value in metrics.items():
            assert np.isclose(rebuilt[condition][key],value,rtol=0,atol=1e-12), (condition,key)
    for file in ['a12_count_predictions.csv','a12_yolo11_confusion.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(output/file),pd.read_csv(rc.OUT_DIR/file))
    checkpoint = torch.load(weights,map_location='cpu',weights_only=False)
    assert list(checkpoint['model'].names.values()) == rc.CLASSES
    args = yaml.safe_load((output/'a12_train_args.yaml').read_text())
    for key,value in dict(epochs=60,batch=32,imgsz=640,seed=42,workers=12,patience=60).items():
        assert args[key] == value
        assert checkpoint['train_args'][key] == value
    curve = pd.read_csv(output/'a12_train_curve.csv')
    curve.columns = curve.columns.str.strip()
    assert curve.epoch.tolist() == list(range(1,61))
    best = curve.loc[curve['metrics/mAP50-95(B)'].idxmax()]
    assert np.isclose(best['metrics/mAP50-95(B)'],checkpoint['train_metrics']['fitness'])
    c = pd.read_csv(output/'a12_yolo11_confusion.csv',index_col=0).to_numpy()
    gt_n = int(c[:4,:].sum())
    detector_n = int(c[:,:4].sum())
    confusion = dict(appearances=gt_n,detections=detector_n,
        correct=int(np.trace(c[:4,:4])),missed=int(c[:4,4].sum()),
        wrong_class=int(c[:4,:4].sum()-np.trace(c[:4,:4])),unmatched_detections=int(c[4,:4].sum()),
        pooled_correct_class_recall=float(np.trace(c[:4,:4])/gt_n),
        macro_correct_class_recall=float(np.mean(np.diag(c[:4,:4])/c[:4,:].sum(axis=1))))
    assert detector_n == counts['test']['detections']
    report = dict(weights_sha256=digest,cache_counts=counts,
        recomputation='All committed metrics, paired tests, predictions and confusion reproduced.',
        training=dict(epochs=60,workers=12,ultralytics=checkpoint['version'],
            parameter_count=sum(p.numel() for p in checkpoint['model'].parameters()),
            duration_seconds=float(curve.time.iloc[-1]),
            best_epoch_inferred_from_curve=int(best.epoch),best=best.to_dict(),last=curve.iloc[-1].to_dict(),
            note='Stripped checkpoint stores epoch=-1; best epoch inferred from matching validation fitness, not checkpoint epoch.'),
        confusion=confusion,metrics=original,
        provenance_note='Original a12_provenance.json omits workers; archived args and checkpoint both record workers=12. GPU absolute paths are historical.',
        local_versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,torch=torch.__version__))
    (output/'a13_gpu_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['cache_counts','recomputation','confusion','provenance_note']},indent=2))


if __name__=='__main__':
    main()
