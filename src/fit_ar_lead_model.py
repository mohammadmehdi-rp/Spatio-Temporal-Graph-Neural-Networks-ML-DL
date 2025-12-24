#!/usr/bin/env python3
"""
fit_ar_lead_model.py — train a pooled AR(lags) Ridge for lead-1 backlog prediction and save it.

Example:
  python3 fit_ar_lead_model.py processed.csv --label backlog_pkts --lags 3 --l2 50 --out ar_lead_model.json
"""
import argparse, json, math
import numpy as np
import pandas as pd
from numpy.linalg import inv

def rmse(a,b):
    e = (a-b).dropna()
    return float(np.sqrt((e**2).mean())) if len(e) else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed")
    ap.add_argument("--label", default=None, choices=["backlog_pkts","backlog_bytes","qlen_pkts"])
    ap.add_argument("--lags", type=int, default=3, help="AR lags (at 5 Hz, 3 ≈ 0.6 s history)")
    ap.add_argument("--l2", type=float, default=50.0, help="ridge L2")
    ap.add_argument("--out", default="ar_lead_model.json")
    args = ap.parse_args()

    df = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    lab = args.label or ("backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts"))

    # target = next sample
    df["target"] = df.groupby("iface")[lab].shift(-1)

    # add AR lags per iface
    for i in range(1, args.lags+1):
        df[f"{lab}_lag{i}"] = df.groupby("iface")[lab].shift(i)

    feats = [f"{lab}_lag{i}" for i in range(1, args.lags+1)]
    df = df.dropna(subset=feats+["target"]).sort_values(["timestamp","iface"])

    # time split on busy if possible
    busy = df[df["target"] > 0]
    uts = np.array(sorted((busy["timestamp"] if len(busy) else df["timestamp"]).unique()))
    split_t = uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
    tr = df["timestamp"] <= split_t
    te = df["timestamp"] >  split_t

    Xtr = df.loc[tr, feats].to_numpy(float); ytr = df.loc[tr, "target"].to_numpy(float)
    Xte = df.loc[te, feats].to_numpy(float); yte = df.loc[te, "target"].to_numpy(float)

    if not len(ytr) or not len(yte):
        raise SystemExit("Not enough train/test rows; collect longer or reduce lags.")

    # standardize + intercept
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])

    w = inv(Xtr.T @ Xtr + args.l2 * np.eye(Xtr.shape[1])) @ (Xtr.T @ ytr)
    yhat = Xte @ w
    score = rmse(pd.Series(yte), pd.Series(yhat))

    # save model
    model = {
        "label": lab,
        "lags": int(args.lags),
        "l2": float(args.l2),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": w[:-1].tolist(),
        "intercept": float(w[-1]),
        "split_time": split_t.isoformat(),
        "train_rows": int(tr.sum()),
        "test_rows": int(te.sum()),
        "rmse_test": score,
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved {args.out}")
    print(f"Lead-1 RMSE (test): {score:.3f} | lags={args.lags} | L2={args.l2} | split={split_t}")

if __name__ == "__main__":
    main()
