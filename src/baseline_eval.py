# Scores naive, moving average, and Ridge on processed.csv (oracle) and processed_masked.csv (masked)

import argparse, pandas as pd, numpy as np
from numpy.linalg import inv

def rmse(a, b):
    e = (a - b).dropna()
    return float(np.sqrt((e**2).mean())) if len(e) else float("nan")

def macro_rmse(df, ycol, yhat):
    vals = []
    for _, g in df[[ycol,yhat,"iface"]].dropna().groupby("iface"):
        if len(g) >= 2:
            vals.append(rmse(g[ycol], g[yhat]))
    return float(np.nanmean(vals)) if vals else float("nan")

def add_lags(df, cols, k):
    for i in range(1, k+1):
        for c in cols:
            df[f"{c}_lag{i}"] = df.groupby("iface")[c].shift(i)
    return df

def ridge(df, feats, ycol, tr_mask, te_mask, l2=1.0):
    dtr = df.loc[tr_mask, feats+[ycol]].dropna()
    dte = df.loc[te_mask, feats+[ycol]].dropna()
    if dtr.empty or dte.empty:
        return (float("nan"), float("nan"), 0, 0)
    Xtr = dtr[feats].to_numpy(float); ytr = dtr[ycol].to_numpy(float)
    Xte = dte[feats].to_numpy(float); yte = dte[ycol].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    w = inv(Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1])) @ (Xtr.T @ ytr)
    yhat = Xte @ w
    micro = float(np.sqrt(np.mean((yte - yhat)**2))) if len(yte) else float("nan")
    tmp = dte.copy(); tmp["yhat"] = yhat
    macro = macro_rmse(tmp.assign(**{ycol: tmp[ycol], "iface": tmp.index.map(lambda i: tmp.iloc[i - tmp.index.min()]["iface"])}) if False else tmp.join(df[["iface"]], how="left"), ycol, "yhat")
    # simpler: reuse groups by iface from dte
    tmp2 = dte.copy(); tmp2["yhat"] = yhat
    macro = macro_rmse(tmp2.join(df[["iface"]], how="left"), ycol, "yhat") if np.isnan(macro) else macro
    return (micro, macro, len(dtr), len(dte))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--masked", default="processed_masked.csv")
    ap.add_argument("--lags", type=int, default=0, help="number of lag steps to add to features (per iface)")
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    df_o = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    df_m = pd.read_csv(args.masked, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    label = "backlog_pkts" if "backlog_pkts" in df_o.columns else ("backlog_bytes" if "backlog_bytes" in df_o.columns else "qlen_pkts")

    base_feats = [c for c in df_o.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df_o.columns:
        base_feats.append("throughput_Mbps")

    if args.lags > 0:
        df_o = add_lags(df_o, base_feats, args.lags)
        df_m = add_lags(df_m, base_feats, args.lags)
        feats = [f"{c}_lag{i}" for i in range(1, args.lags+1) for c in base_feats]
    else:
        feats = base_feats

    # naive & MA5 on oracle
    d = df_o.copy()
    d["naive"] = d.groupby("iface")[label].shift(1)
    d["ma5"] = d.groupby("iface")[label].rolling(5).mean().reset_index(level=0, drop=True)

    # split: prefer busy period (>0), else global 80/20 by timestamp
    busy = d[d[label] > 0]
    uts = np.array(sorted(busy["timestamp"].unique())) if len(busy) else np.array(sorted(d["timestamp"].unique()))
    split_t = uts[int(0.8*len(uts))] if len(uts) > 1 else d["timestamp"].max()
    tr_o = d["timestamp"] <= split_t
    te_o = d["timestamp"] >  split_t

    naive_micro = rmse(d.loc[te_o, label], d.loc[te_o, "naive"])
    naive_macro = macro_rmse(d.loc[te_o], label, "naive")
    ma5_micro   = rmse(d.loc[te_o, label], d.loc[te_o, "ma5"])
    ma5_macro   = macro_rmse(d.loc[te_o], label, "ma5")

    # ridge oracle
    ro_micro, ro_macro, ro_tr, ro_te = ridge(d, feats, label, tr_o, te_o, l2=args.l2)

    # ridge masked
    dm = df_m.copy()
    tr_m = dm["timestamp"] <= split_t
    te_m = dm["timestamp"] >  split_t
    rm_micro, rm_macro, rm_tr, rm_te = ridge(dm, feats, label, tr_m, te_m, l2=args.l2)

    print(f"Label: {label}")
    print(f"Split time: {split_t}")
    print(f"Naive(t-1)  → micro {naive_micro:.3f} | macro {naive_macro:.3f}")
    print(f"MovAvg(5)   → micro {ma5_micro:.3f} | macro {ma5_macro:.3f}")
    print(f"Ridge ORCL  → micro {ro_micro:.3f} | macro {ro_macro:.3f} | train {ro_tr} | test {ro_te}")
    print(f"Ridge MASK  → micro {rm_micro:.3f} | macro {rm_macro:.3f} | train {rm_tr} | test {rm_te}")

if __name__ == "__main__":
    main()
