"""Prepare/train/evaluate YOLO11m for reviewer 4 on the official tree split.

Prepare only (no model download/training):
  python experiments/revision/a12_second_detector.py --prepare-only
Full run on a CUDA machine:
  python experiments/revision/a12_second_detector.py --device 0
Evaluate an already trained four-class checkpoint:
  python experiments/revision/a12_second_detector.py --weights path/to/best.pt --device 0

Training: 60 epochs, batch 32, imgsz 640, patience 60, seed 42. The selected
best.pt uses detector validation, not test. Ridge+F0 is fixed before evaluation.
The script does not imply a matched architecture-only causal comparison: the
architecture-specific Ultralytics defaults are saved with the training run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import revcommon as rc
from a10_revision_audit import comparison
from a2_error_decomposition import load_json, match_tree


def prepare(data_dir):
    manifest = pd.read_csv(rc.GT_DIR.parent/'split_manifest.csv').set_index('tree_id')
    assert manifest.new_split.value_counts().to_dict() == {'train':716,'test':141,'val':96}
    grouped = defaultdict(list)
    for path in sorted((data_dir/'images').rglob('*.jpg')):
        tid, side = path.stem.rsplit('_',1)
        if tid in manifest.index:
            parts = list(path.parts)
            pos = parts.index('images')
            label = Path(*parts[:pos], 'labels', *parts[pos+1:]).with_suffix('.txt')
            if not label.exists():
                raise FileNotFoundError(label)
            grouped[tid].append((int(side),path.resolve()))
    assert set(grouped)==set(manifest.index), 'Missing tree images'
    assert sum(map(len,grouped.values()))==3992, 'Unexpected image count'
    for tid, sides in grouped.items():
        assert len({i for i,_ in sides})==len(sides), f'Duplicate side: {tid}'
    config_dir = rc.ROOT/'.cache/reviewer-yolo11-data'
    config_dir.mkdir(parents=True,exist_ok=True)
    config = {'path':str(data_dir.resolve()),'names':{i:c for i,c in enumerate(rc.CLASSES)}}
    for split in ['train','val','test']:
        paths = [str(path).replace('\\','/') for tid,sides in grouped.items()
                 if manifest.loc[tid,'new_split']==split for _,path in sides]
        listing = config_dir/(split+'.txt')
        listing.write_text('\n'.join(paths)+'\n',encoding='utf-8')
        config[split] = str(listing.resolve())
        print(split, int((manifest.new_split==split).sum()),'trees',len(paths),'images',flush=True)
    import yaml
    yaml_path=config_dir/'data.yaml'
    yaml_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    return yaml_path,grouped,manifest


def evaluate(source):
    stats, metrics, pred_rows, reference_ids = {},{},[],None
    for name,path in [('YOLO11m',source),('YOLO26m',rc.PRED_DIR),('GT',rc.GT_DIR)]:
        df,y,ids,splits,_ = rc.load_dataset(path)
        if reference_ids is None:
            reference_ids=ids
        assert ids==reference_ids, 'Tree alignment differs'
        x=df[rc.feature_sets(df)['F0']].to_numpy(float)
        model=rc.model_builders()['Ridge']()
        model.fit(x[splits=='train'],y[splits=='train'])
        te=splits=='test'
        pred=rc.round_counts(model.predict(x[te]))
        stats[name]=rc.per_tree_stats(y[te],pred)
        metrics[name]=rc.metrics_from_stats(stats[name])
        for tid,t,p in zip(np.array(ids)[te],y[te],pred):
            pred_rows.append(dict(detector=name,tree_id=tid,
                **{f'true_{c}':int(v) for c,v in zip(rc.CLASSES,t)},
                **{f'pred_{c}':int(v) for c,v in zip(rc.CLASSES,p)}))
    metrics['paired_YOLO11_vs_YOLO26']=comparison(stats['YOLO11m'],stats['YOLO26m'])
    metrics['paired_GT_vs_YOLO11']=comparison(stats['GT'],stats['YOLO11m'])
    confusion=np.zeros((5,5),dtype=int)
    manifest=pd.read_csv(rc.GT_DIR.parent/'split_manifest.csv')
    for tid in manifest.loc[manifest.new_split=='test','tree_id']:
        pairs,fp,fn=match_tree(load_json(rc.GT_DIR/f'{tid}.json'),load_json(source/f'{tid}.json'))
        for _,_,g,p,_ in pairs:
            confusion[rc.CLASSES.index(g),rc.CLASSES.index(p)]+=1
        for _,p,_ in fp:
            confusion[4,rc.CLASSES.index(p)]+=1
        for _,_,g in fn:
            confusion[rc.CLASSES.index(g),4]+=1
    pd.DataFrame(confusion,index=rc.CLASSES+['background'],columns=rc.CLASSES+['background']).to_csv(
        rc.OUT_DIR/'a12_yolo11_confusion.csv')
    pd.DataFrame(pred_rows).to_csv(rc.OUT_DIR/'a12_count_predictions.csv',index=False)
    rc.dump(metrics,rc.OUT_DIR/'a12_second_detector_metrics.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=rc.ROOT/'SawitMVC-YOLO')
    p.add_argument('--device',default='0',help='CUDA device index or cpu')
    p.add_argument('--weights',type=Path,help='Already fine-tuned four-class checkpoint')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    yaml_path,grouped,manifest=prepare(args.data.resolve())
    if args.prepare_only:
        print('Data validated; no detector was trained or evaluated.')
        return
    import torch
    from ultralytics import YOLO
    if args.device!='cpu' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Run this package on GPU, or explicitly select --device cpu for a slow run.')
    weights=args.weights
    if weights is None:
        model=YOLO('yolo11m.pt')
        model.train(data=str(yaml_path),epochs=60,batch=32,imgsz=640,patience=60,
                    seed=42,deterministic=True,device=args.device,workers=0,
                    project=str(rc.ROOT/'runs/reviewer-yolo11'),name='train',exist_ok=False)
        weights=Path(model.trainer.best)
    weights=weights.resolve()
    sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    model=YOLO(str(weights))
    assert [model.names[i] for i in range(len(model.names))]==rc.CLASSES, 'Not a trained B1--B4 checkpoint'
    output=rc.ROOT/'predictions'/f'yolo11m_revision_{sha[:12]}_per_tree'
    output.mkdir(parents=True,exist_ok=True)
    settings={'weights_sha256':sha,'conf':.25,'imgsz':640}
    for tid,sides in grouped.items():
        target=output/f'{tid}.json'
        if target.exists():
            assert json.loads(target.read_text())['inference_settings']==settings
            continue
        images={}
        for side,path in sides:
            r=model.predict(str(path),conf=.25,imgsz=640,device=args.device,verbose=False)[0]
            boxes=r.boxes
            annotations=[dict(box_index=k,class_name=rc.CLASSES[int(c)],bbox_yolo=box,conf=float(conf))
                for k,(c,box,conf) in enumerate(zip(boxes.cls.cpu().tolist(),boxes.xywhn.cpu().tolist(),boxes.conf.cpu().tolist()))]
            images[f'side_{side}']=dict(side_index=side-1,annotations=annotations)
        target.write_text(json.dumps(dict(tree_name=tid,split=manifest.loc[tid,'new_split'],
            inference_settings=settings,images=images)),encoding='utf-8')
        print('Inferred',tid,flush=True)
    rc.OUT_DIR.mkdir(parents=True,exist_ok=True)
    rc.dump(dict(settings,weights=str(weights),predictions=str(output)),rc.OUT_DIR/'a12_provenance.json')
    evaluate(output)


if __name__=='__main__':
    main()
