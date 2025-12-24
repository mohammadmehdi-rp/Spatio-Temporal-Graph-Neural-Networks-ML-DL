# Results Overview Outputs

This folder adds *presentation-ready* artifacts that summarize all experiments and highlight the **proposed configuration**.

## Proposed configuration
- Encoder: GraphSAGE
- Sensors: k=10 setup (main configuration used in later experiments)
- Temporal features: Full lags (Lag1–3)
- Reported in two ways:
  1) **Best single run** (useful for showing peak performance)
  2) **Mean ± Std** over multiple seeds/runs (more scientific and reproducible)

## Files
- `outputs/summaries/results_overview.csv` — all numbers in one CSV.
- `outputs/tables/proposal_vs_alternatives_nowcast.tex` — thesis table (nowcast).
- `outputs/tables/proposal_vs_alternatives_lead1.tex` — thesis table (Lead-1).
- `outputs/figures/overview_nowcast_micro.*` — overview bar plot (micro RMSE).
- `outputs/figures/overview_nowcast_macro.*` — overview bar plot (macro RMSE).
- `outputs/figures/overview_lead1_micro.*` — overview bar plot (Lead-1 micro RMSE).
- `outputs/figures/overview_calibration_idlefp.*` — calibration effect on idle false positives.

Note: the LaTeX tables use `booktabs`, `float`, and (for multirow labels) `multirow`.


## Added: Nowcast calibration & baseline artifacts

- `outputs/summaries/nowcast_baselines.csv`
- `outputs/tables/nowcast_calibration_baselines.tex`
- `outputs/figures/nowcast_calibration_*.(png|pdf)`

These files summarize the calibration effect (idle false positives vs busy RMSE) and a few simple reference baselines.
