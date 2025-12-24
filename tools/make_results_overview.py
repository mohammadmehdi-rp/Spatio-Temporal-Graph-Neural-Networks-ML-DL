#!/usr/bin/env python3
"""Generate overview plots + LaTeX tables from outputs/summaries/results_overview.csv.

Designed for the thesis packaging: FAST mode regenerates the key comparison
figures/tables without retraining.

Input CSV format:
method,task,metric,value,std,kind,notes

Only rows with numeric 'value' are used.
"""

import argparse
import os
import math
import pandas as pd
import numpy as np

# Ensure matplotlib uses a writable cache directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt


def _to_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def make_barplot(df, title, ylabel, out_png, out_pdf, order=None):
    """Simple bar plot with optional error bars."""
    if order is None:
        order = list(df["method"].unique())

    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method")

    x = np.arange(len(df))
    y = df["value"].to_numpy(dtype=float)
    yerr = df["std"].to_numpy(dtype=float)
    yerr = np.where(np.isnan(yerr), 0.0, yerr)

    plt.figure(figsize=(10, 4.2))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, df["method"], rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("_", "\\_")
             .replace("%", "\\%")
             .replace("&", "\\&")
             .replace("#", "\\#")
             .replace("$", "\\$")
             .replace("{", "\\{")
             .replace("}", "\\}"))


def write_tex_table_nowcast(df_micro, df_macro, out_tex):
    """Proposal vs alternatives (nowcast): micro+macro."""
    # Prefer rows that are clearly the key comparison (remove stability/cross-trace/calibration)
    def keep(m):
        m = str(m)
        bad = ["Stability", "Cross-trace", "Calibration"]
        return not any(b in m for b in bad)

    dfm = df_micro[df_micro["method"].apply(keep)].copy()
    dfa = df_macro[df_macro["method"].apply(keep)].copy()

    # align methods
    methods = list(dict.fromkeys(list(dfm["method"]) + list(dfa["method"])))

    micro_map = {r.method: (r.value, r.std, r.kind) for r in dfm.itertuples(index=False)}
    macro_map = {r.method: (r.value, r.std, r.kind) for r in dfa.itertuples(index=False)}

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Proposal vs. alternatives on nowcasting (test split). Mean$\pm$std rows aggregate multiple seeds/runs; 'best' denotes the strongest single run. Lower RMSE is better.}")
    lines.append(r"\label{tab:proposal-vs-alternatives-nowcast}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Micro RMSE & Macro RMSE \\")
    lines.append(r"\midrule")

    for m in methods:
        mv, ms, mk = micro_map.get(m, (np.nan, np.nan, ""))
        av, astd, ak = macro_map.get(m, (np.nan, np.nan, ""))

        def fmt(v, s, kind):
            if np.isnan(v):
                return "--"
            if kind and "mean" in kind and not np.isnan(s):
                return f"{v:.3f} $\\pm$ {s:.3f}"
            return f"{v:.3f}"

        lines.append(f"{tex_escape(m)} & {fmt(mv, ms, mk)} & {fmt(av, astd, ak)} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_tex_table_lead1(df, out_tex):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Proposal vs. alternatives on Lead-1 forecasting (test split). Lower RMSE is better.}")
    lines.append(r"\label{tab:proposal-vs-alternatives-lead1}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Micro RMSE \\")
    lines.append(r"\midrule")

    for r in df.itertuples(index=False):
        v, s, kind = r.value, r.std, str(r.kind)
        if np.isnan(v):
            continue
        if ("mean" in kind) and (not np.isnan(s)):
            val = f"{v:.3f} $\\pm$ {s:.3f}"
        else:
            val = f"{v:.3f}"
        lines.append(f"{tex_escape(str(r.method))} & {val} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True, help="outputs/ directory")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["value"] = df["value"].apply(_to_float)
    df["std"] = df["std"].apply(_to_float)

    out_fig = os.path.join(args.out_dir, "figures")
    out_tab = os.path.join(args.out_dir, "tables")
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_tab, exist_ok=True)

    # NOWCAST micro/macro overview
    now_micro = df[(df.task == "nowcast") & (df.metric == "micro_RMSE") & (~df.value.isna())]
    now_macro = df[(df.task == "nowcast") & (df.metric == "macro_RMSE") & (~df.value.isna())]

    # For the figure, focus on the key comparison set (proposal vs ablations vs encoder vs random)
    fig_methods = [
        "Proposed (best single run)",
        "Proposed (GraphSAGE, k=10, full lags)",
        "Lag1 only",
        "No lags",
        "Lag1+Lag2",
        "Random sensors (k=10)",
        "RouteNet-lite (same setup)",
    ]

    now_micro_fig = now_micro[now_micro["method"].isin(fig_methods)].copy()
    now_macro_fig = now_macro[now_macro["method"].isin(fig_methods)].copy()

    if len(now_micro_fig) > 0:
        make_barplot(
            now_micro_fig,
            title="Nowcast comparison (micro RMSE)",
            ylabel="Test micro RMSE",
            out_png=os.path.join(out_fig, "overview_nowcast_micro.png"),
            out_pdf=os.path.join(out_fig, "overview_nowcast_micro.pdf"),
            order=fig_methods,
        )

    if len(now_macro_fig) > 0:
        make_barplot(
            now_macro_fig,
            title="Nowcast comparison (macro RMSE)",
            ylabel="Test macro RMSE",
            out_png=os.path.join(out_fig, "overview_nowcast_macro.png"),
            out_pdf=os.path.join(out_fig, "overview_nowcast_macro.pdf"),
            order=fig_methods,
        )

    # LEAD1 micro overview
    lead_micro = df[(df.task == "lead1") & (df.metric == "micro_RMSE") & (~df.value.isna())].copy()
    lead_methods = [
        "Proposed (best single run)",
        "Proposed (GraphSAGE, k=10, full lags)",
        "RouteNet-lite (same setup)",
    ]
    lead_micro_fig = lead_micro[lead_micro["method"].isin(lead_methods)].copy()
    if len(lead_micro_fig) > 0:
        make_barplot(
            lead_micro_fig,
            title="Lead-1 comparison (micro RMSE)",
            ylabel="Test micro RMSE",
            out_png=os.path.join(out_fig, "overview_lead1_micro.png"),
            out_pdf=os.path.join(out_fig, "overview_lead1_micro.pdf"),
            order=lead_methods,
        )

    # Calibration overview (idle FP)
    cal_fp = df[(df.task == "nowcast") & (df.metric == "idle_FP_rate") & (~df.value.isna())].copy()
    if len(cal_fp) > 0:
        make_barplot(
            cal_fp,
            title="Calibration effect (idle false-positive rate)",
            ylabel="Idle FP rate",
            out_png=os.path.join(out_fig, "overview_calibration_idlefp.png"),
            out_pdf=os.path.join(out_fig, "overview_calibration_idlefp.pdf"),
        )

    # LaTeX tables
    write_tex_table_nowcast(now_micro, now_macro, os.path.join(out_tab, "proposal_vs_alternatives_nowcast.tex"))
    write_tex_table_lead1(lead_micro, os.path.join(out_tab, "proposal_vs_alternatives_lead1.tex"))

    print(f"[OK] Wrote plots to: {out_fig}")
    print(f"[OK] Wrote tables to: {out_tab}")


if __name__ == "__main__":
    main()
