# Thesis Reproducible Package

This folder is a self-contained reproducibility bundle for the thesis experiments:

- Sensor selection optimization
- Temporal ablations
- Encoder variants (GraphSAGE vs RouteNet-lite)
- Calibration for nowcast
- Stability & ensembles
- Cross-trace generalization

## Folder layout

- `src/` : Python source code (training, evaluation, plotting)
- `scripts/` : one-click drivers (FAST and FULL) + per-experiment scripts
- `data/npz/` : NPZ datasets and NPZ artifacts
- `outputs/` :
  - `outputs/models/` : pretrained checkpoints (used by FAST mode)
  - `outputs/figures/` : figures (PDF+PNG)
  - `outputs/tables/` : LaTeX tables
  - `outputs/summaries/` : CSV/JSON summaries
  - `outputs/logs/` : logs produced by FULL mode
- `docs/` : documentation and a short results overview
- `latex/` : thesis-ready LaTeX snippet(s)

## Environment

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate thesis-repro
```

Pip (alternative):
```bash
pip install -r requirements.txt
```

PyTorch installation can be CPU or GPU depending on your machine.

## Reproduce results

FAST mode (minutes): regenerates plots/tables from the included summaries and cached artifacts.
```bash
bash scripts/reproduce_all_fast.sh
```

FULL mode (hours): retrains and re-evaluates everything from scratch.
```bash
bash scripts/reproduce_all_full.sh
```

## Notes

- Datasets are expected under `data/npz/`. The scripts reference these paths explicitly.
- If you run into path issues, execute scripts from the package root directory.
