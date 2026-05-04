# phase1_restart

Single-night Phase-1 push targeting Kaggle ≥ 0.7627. Foundation-model fine-tune (CellSAM + MEDIAR), parallel on Modal, with Phase-2 re-run before tonight's 23:55 EST deadline.

See [docs/2026-05-04-phase1-restart-design.md](docs/2026-05-04-phase1-restart-design.md) for the design and [docs/2026-05-04-phase1-restart-plan.md](docs/2026-05-04-phase1-restart-plan.md) for the implementation plan.

## Run order tonight

1. `PYTHONPATH=. python phase1_restart/scripts/smoke.py` — coord sanity gate
2. `PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model cellsam --split val`
3. `PYTHONPATH=. python -m phase1_restart.scripts.zero_shot --model mediar --split val`
4. `python -m phase1_restart.scripts.fine_tune --model cellsam --config phase1_restart/configs/cellsam.yaml` (Modal, detached)
5. `python -m phase1_restart.scripts.fine_tune --model mediar --config phase1_restart/configs/mediar.yaml` (Modal, detached)
6. … see plan for full sequence (Blocks A through G).

## Folder name

Underscored (`phase1_restart`) instead of hyphenated (like sibling `phase2-restart/`) because Python module names cannot contain dashes — and this folder is imported as a package.
