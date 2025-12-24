# LaTeX snippets

- `packaging_reproducibility.tex`: drop-in section for your thesis.
- The `outputs/tables/`*.tex` files are LaTeX tables generated from the experiments.

Required packages in your thesis preamble:

```latex
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{multirow}
```

Typical usage:

```latex
\input{outputs/tables/proposal_vs_alternatives_nowcast}
\input{outputs/tables/proposal_vs_alternatives_lead1}
\input{outputs/tables/calibration_summary}
```

Figures are in `outputs/figures/` as both PDF and PNG.

Additional thesis-ready sections (drop-in):

```latex
\input{latex/sections/temporal_ablations}
\input{latex/sections/encoder_variants}
\input{latex/sections/calibration_nowcast}
\input{latex/sections/stability_ensembles}
\input{latex/sections/cross_trace_generalization}
```
