#!/usr/bin/env python3
"""
sparse_sensors_sweep.py
Sweep sparse-sensor fractions (default: 10%, 20%, 30%) and report Naive, Ridge ORCL, Ridge MASK.

What it does:
  • Loads processed.csv (from process_data.py)
  • Picks sensors for each fraction: always include --include, fill the rest by highest label variance
  • Masks features on non-sensors (labels are never masked)
  • Uses one consistent time split for all fractions (80/20 on busy timestamps)
  • Evaluates:
      - Naive(t-1) RMSE (micro & macro)
      - Ridge ORCL  (all features)
      - Ridge MASK  (only sensor features; others NaN)
  • Writes: sparse_results.csv and per-fraction sensors files: sensors_frac_10.txt, etc.
"""

import argparse, math, numpy as np, pandas as pd
from numpy.linalg import inv

def rmse(a, b):
    e = (a - b).dropna()
    return float(np.sqrt((e**2).mean())) if len(e) else float("nan")

def macro_rmse(df, ycol, yhat):
    vals=[]
    for _, g in df[[ycol,yhat,"iface"]].dropna().groupby("iface"):
        if len(g)>=2:
            vals.append(rmse(g[ycol], g[yhat]))
    return float(np.nanmean(vals)) if vals else float("nan")

def add_lags_by_iface(df, cols, k):
    for i in range(1, k+1):
        for c in cols:
            df[f"{c}_lag{i}"] = df.groupby("iface")[c].shift(i)
    return df

def fit_ridge(df, feats, ycol, tr_mask, te_mask, l2=50.0):
    cols = feats + [ycol, "iface"]
    dtr = df.loc[tr_mask, cols].dropna()
    dte = df.loc[te_mask, cols].dropna()
    if dtr.empty or dte.empty:
        return dict(micro=float("nan"), macro=float("nan"), tr=0, te=0)

    Xtr = dtr[feats].to_numpy(float); ytr = dtr[ycol].to_numpy(float)
    Xte = dte[feats].to_numpy(float); yte = dte[ycol].to_numpy(float)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0],1))])
    Xte = np.hstack([Xte, np.ones((Xte.shape[0],1))])

    w = inv(Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1])) @ (Xtr.T @ ytr)
    yhat = Xte @ w

    micro = rmse(pd.Series(yte), pd.Series(yhat))
    tmp = dte.copy(); tmp["yhat"] = yhat
    macro = macro_rmse(tmp, ycol, "yhat")
    return dict(micro=micro, macro=macro, tr=len(dtr), te=len(dte))

def pick_sensors(df, label, include_list, count):
    by_std = df.groupby("iface")[label].std().sort_values(ascending=False)
    sensors=[]
    for m in include_list:
        if m in by_std.index and m not in sensors:
            sensors.append(m)
    for iface in by_std.index:
        if len(sensors) >= count: break
        if iface not in sensors:
            sensors.append(iface)
    return sensors[:count]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--fractions", default="0.1,0.2,0.3", help="comma-separated sensor fractions")
    ap.add_argument("--include", default="s2-eth2,s2-eth1,s1-eth1,s1-eth2", help="must-include sensors")
    ap.add_argument("--lags", type=int, default=10, help="feature lags for Ridge")
    ap.add_argument("--l2", type=float, default=50.0, help="Ridge L2")
    ap.add_argument("--out", default="sparse_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    if df.empty:
        raise SystemExit("processed.csv is empty or not found.")

    # label
    label = None
    for c in ["backlog_pkts","backlog_bytes","qlen_pkts"]:
        if c in df.columns: label = c; break
    if label is None:
        raise SystemExit("No label column (backlog_pkts/backlog_bytes/qlen_pkts) in processed.csv")

    # base feature columns (per-second counters + optional throughput)
    base_feats = [c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns:
        base_feats.append("throughput_Mbps")

    # add lags once, reuse
    if args.lags > 0:
        df_lag = add_lags_by_iface(df.copy(), base_feats, args.lags)
        lag_feats = [f"{c}_lag{i}" for i in range(1, args.lags+1) for c in base_feats]
    else:
        df_lag = df.copy()
        lag_feats = base_feats

    # consistent split time based on busy timestamps
    busy = df_lag[df_lag[label] > 0]
    uts = np.array(sorted((busy["timestamp"] if len(busy) else df_lag["timestamp"]).unique()))
    split_t = uts[int(0.8*len(uts))] if len(uts)>1 else df_lag["timestamp"].max()
    tr = df_lag["timestamp"] <= split_t
    te = df_lag["timestamp"] >  split_t

    # Oracle ridge once (reference, not sparse)
    ridge_orcl = fit_ridge(df_lag, lag_feats, label, tr, te, l2=args.l2)

    # Naive baseline (independent of sensors)
    d_naive = df_lag.copy()
    d_naive["naive"] = d_naive.groupby("iface")[label].shift(1)
    naive_micro = rmse(d_naive.loc[te,label], d_naive.loc[te,"naive"])
    naive_macro = macro_rmse(d_naive.loc[te], label, "naive")

    include_list = [x.strip() for x in args.include.split(",") if x.strip()]
    ifaces = df["iface"].dropna().unique().tolist()
    N = len(ifaces)

    rows=[]
    for frac_str in args.fractions.split(","):
        frac = float(frac_str)
        k = max(1, int(round(frac * N)))
        sensors = pick_sensors(df, label, include_list, k)

        # build masked copy for this fraction
        dfm = df_lag.copy()
        feat_cols_now = [c for c in dfm.columns if c.endswith("_per_s")] + (["throughput_Mbps"] if "throughput_Mbps" in dfm.columns else [])
        # mask base feats and their lags for non-sensors
        mask_idx = ~dfm["iface"].isin(sensors)
        for c in feat_cols_now:
            dfm.loc[mask_idx, c] = np.nan
        for i in range(1, args.lags+1):
            for c in feat_cols_now:
                col = f"{c}_lag{i}"
                if col in dfm.columns:
                    dfm.loc[mask_idx, col] = np.nan

        ridge_mask = fit_ridge(dfm, lag_feats, label, tr, te, l2=args.l2)

        rows.append({
            "fraction": frac,
            "sensors_count": len(sensors),
            "sensors": ",".join(sensors),
            "split_time": split_t,
            "naive_micro": naive_micro,
            "naive_macro": naive_macro,
            "ridge_orcl_micro": ridge_orcl["micro"],
            "ridge_orcl_macro": ridge_orcl["macro"],
            "ridge_mask_micro": ridge_mask["micro"],
            "ridge_mask_macro": ridge_mask["macro"],
            "train_rows": ridge_mask["tr"],
            "test_rows": ridge_mask["te"],
            "lags": args.lags,
            "l2": args.l2
        })

        # also dump the sensors list for this fraction
        txt_name = f"sensors_frac_{int(round(frac*100))}.txt"
        with open(txt_name,"w") as f: f.write("\n".join(sensors))
        print(f"[{int(frac*100)}%] sensors → {txt_name}: {sensors}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    # pretty print
    show = out[["fraction","sensors_count","ridge_mask_micro","ridge_mask_macro","naive_micro","ridge_orcl_micro","lags","l2"]]
    print("\n=== Sparse sensor sweep (RMSE) ===")
    print(show.to_string(index=False))
    print(f"\nWrote {args.out} with details (including sensor lists).")

if __name__ == "__main__":
    main()
