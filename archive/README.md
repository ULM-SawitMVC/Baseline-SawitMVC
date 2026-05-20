# Archive

This folder contains **old experiments, superseded detectors, and historical
artifacts** that are kept only for reference.

Nothing under `archive/` is part of the latest baseline release surface.

## What belongs here

- legacy detector outputs (`y26n`, `y26s`, `y26m`)
- per-image experimental pipelines and their results
- historical docs that describe superseded experiments
- old model weights and training logs that are no longer the canonical baseline

## What does not belong here

The latest baseline remains in the repository root:

- `models/yolo/y26mv2.pt`
- `predictions/y26mv2_per_tree/`
- `results/e2e_per_tree/y26mv2_*/`
- `results/e2e_upper_bound/`
- `results/experiments/`

If a file is needed to reproduce the current README headline numbers, it should
stay outside `archive/`.
