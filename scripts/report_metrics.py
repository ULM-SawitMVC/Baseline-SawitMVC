"""
Print evaluation metrics for a given detector and split.

Usage:
    python scripts/report_metrics.py [name] [split]

Examples:
    python scripts/report_metrics.py y26mv2 test
    python scripts/report_metrics.py y26mv2 val
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["B1", "B2", "B3", "B4"]
NAME  = sys.argv[1] if len(sys.argv) > 1 else "y26mv2"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"

results_dir = ROOT / "results" / "e2e_per_tree"
counters = ["svm", "lr", "rf", "m01"]

print(f"\n{'='*70}")
print(f"  Metrics: {NAME}  |  Split: {SPLIT}")
print(f"{'='*70}")

for counter in counters:
    mpath = results_dir / f"{NAME}_{counter}" / "metrics.json"
    if not mpath.exists():
        print(f"\n[{counter.upper()}] not found: {mpath}")
        continue
    m = json.loads(mpath.read_text())[SPLIT]
    print(f"\n[{counter.upper()}]  n_trees={m['n_trees']}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*39}")
    for c in CLASSES:
        print(f"  {'MAE_'+c:<28} {m[f'MAE_{c}']:>10.4f}")
    print(f"  {'macro_class_mae':<28} {m['macro_class_mae']:>10.4f}")
    print(f"  {'total_count_mae':<28} {m['total_count_mae']:>10.4f}")
    print(f"  {'exact_profile_acc':<28} {m['exact_profile_acc']*100:>9.2f}%")
    print(f"  {'total_pm1_acc':<28} {m['total_pm1_acc']*100:>9.2f}%")
    print(f"  {'macro_acc_pm1 (Acc+/-1)':<28} {m['macro_acc_pm1']*100:>9.2f}%")
    print(f"\n  Per-class Acc+/-1:")
    for c in CLASSES:
        print(f"    {'acc_pm1_'+c:<26} {m[f'acc_pm1_{c}']*100:>8.2f}%")
    print(f"\n  Per-class Bias (mean error):")
    for c in CLASSES:
        bias = m[f'bias_{c}']
        direction = "over" if bias > 0 else "under"
        print(f"    {'bias_'+c:<26} {bias:>+10.4f}  ({direction}count)")
