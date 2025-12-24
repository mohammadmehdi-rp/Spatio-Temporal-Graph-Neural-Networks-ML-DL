#!/usr/bin/env python3
"""
predict_ar_lead.py — load ar_lead_model.json and emit next-sample predictions per iface.

Examples:
  python3 predict_ar_lead.py processed.csv ar_lead_model.json
  python3 predict_ar_lead.py processed.csv ar_lead_model.json --out preds.csv
"""
import argparse, json, sys, numpy as np, pandas as pd
from datetime import timedelta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed")
    ap.add_argument("model_json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    M = json.load(open(args.model_json))
    lab = M["label"]; L = int(M["lags"])
    mu = np.array(M["mu"]); sd = np.array(M["sd"])
    coef = np.array(M["coef"]); b0 = float(M["intercept"])

    df = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["iface","timestamp"])

    # build lags per iface and take the LAST available row per iface
    for i in range(1, L+1):
        df[f"{lab}_lag{i}"] = df.groupby("iface")[lab].shift(i)
    feats = [f"{lab}_lag{i}" for i in range(1, L+1)]
    last = (df.dropna(subset=feats)
              .groupby("iface", as_index=False)
              .tail(1)
              .copy())

    if last.empty:
        print("No iface has sufficient history for prediction.", file=sys.stderr)
        sys.exit(1)

    # estimate typical dt per iface to propose a next timestamp
    dts = (df.sort_values(["iface","timestamp"])
             .groupby("iface")["timestamp"]
             .apply(lambda s: s.diff().dropna().median()))
    dt_default = pd.to_timedelta(dts.median()) if len(dts) else pd.to_timedelta("0.2s")

    X = last[feats].to_numpy(float)
    X = (X - mu) / sd
    yhat = X @ coef + b0

    out = pd.DataFrame({
        "iface": last["iface"].values,
        "last_timestamp": last["timestamp"].dt.tz_convert(None) if last["timestamp"].dt.tz is not None else last["timestamp"],
        "pred_next_backlog": yhat,
        "label": lab,
    })

    # add suggested next timestamp
    # try per-iface dt; fall back to global median
    next_ts = []
    for iface, ts in zip(out["iface"], out["last_timestamp"]):
        dt = dts.get(iface, dt_default)
        if pd.isna(dt): dt = dt_default
        next_ts.append(ts + dt)
    out["pred_for_timestamp"] = next_ts

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {args.out} ({len(out)} rows)")
    else:
        print(out.to_string(index=False))

if __name__ == "__main__":
    main()
