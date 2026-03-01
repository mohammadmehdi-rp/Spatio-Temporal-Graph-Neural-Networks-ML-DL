#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="outputs/summaries/nowcast_baselines.csv")
    ap.add_argument("--out_tex", default="outputs/tables/nowcast_calibration_baselines.tex")
    ap.add_argument("--out_prefix", default="outputs/figures/nowcast_calibration")
    args=ap.parse_args()

    df=pd.read_csv(args.in_csv)
    # Write LaTeX table
    df2=df.copy()
    for c in ["global_RMSE","idle_RMSE","busy_RMSE"]:
        df2[c]=df2[c].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
    df2["idle_FP_rate"]=df2["idle_FP_rate"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
    df2["idle_mean_pred"]=df2["idle_mean_pred"].map(lambda x: f"{x:.1f}" if np.isfinite(x) else "")
    cols=["method","global_RMSE","idle_RMSE","busy_RMSE","idle_FP_rate","idle_mean_pred"]
    tex=df2[cols].to_latex(index=False, escape=False, column_format="lrrrrr",
                           caption="Nowcast calibration and simple baselines on the test split (busy threshold 50 pkts).",
                           label="tab:nowcast_calib_baselines")
    with open(args.out_tex,"w",encoding="utf-8") as f:
        f.write(tex)

    methods=df["method"].tolist()
    x=np.arange(len(methods))

    def barplot(values, ylabel, suffix):
        plt.figure()
        plt.bar(x, values)
        plt.xticks(x, methods, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_{suffix}.png")
        plt.savefig(f"{args.out_prefix}_{suffix}.pdf")
        plt.close()

    barplot(df["global_RMSE"].to_numpy(), "RMSE (global)", "global_rmse")
    barplot(df["busy_RMSE"].to_numpy(), "RMSE (busy only)", "busy_rmse")
    barplot(df["idle_FP_rate"].to_numpy(), "Idle false positive rate", "idle_fp")

    print("[OK] wrote:", args.out_tex, "and plots with prefix", args.out_prefix)

if __name__=="__main__":
    main()
