#!/usr/bin/env python3
"""
Per-interface 1D Kalman Filter (random-walk) baseline:
  x_{t+1} = x_t + w,  z_t = x_t + v
Use previous filtered state to predict next (lead-1). Tune Q/R globally.
Reports micro/macro RMSE on test split (80/20 busy).
"""
import argparse, numpy as np, pandas as pd, math

def rmse(a,b): e=(a-b).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else float("nan")
def macro_rmse(df, y, yhat):
    vals=[]; 
    for _,g in df[[y,yhat,"iface"]].dropna().groupby("iface"):
        if len(g)>=2: vals.append(rmse(g[y], g[yhat]))
    return float(np.nanmean(vals)) if vals else float("nan")

def kf_lead1_series(z, Q, R):
    """Return one-step-ahead predictions for series z (pd.Series)."""
    x=None; P=None
    preds=[]
    for val in z:
        if x is None:
            x=float(val) if pd.notna(val) else 0.0
            P=1.0
            preds.append(np.nan)  # no prior to predict first step
            continue
        # predict next (for t): x_prior = x; P_prior = P+Q
        x_prior = x; P_prior = P + Q
        # update with current z_t
        if pd.notna(val):
            K = P_prior / (P_prior + R)
            x = x_prior + K*(float(val) - x_prior)
            P = (1.0 - K)*P_prior
        else:
            x = x_prior; P = P_prior
        # the lead-1 prediction is x itself (for next step)
        preds.append(x)
    return pd.Series(preds, index=z.index)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--label", default=None, choices=["backlog_pkts","backlog_bytes","qlen_pkts"])
    ap.add_argument("--Q", type=float, default=50.0, help="process noise")
    ap.add_argument("--R", type=float, default=200.0, help="measurement noise")
    args=ap.parse_args()

    df=pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["iface","timestamp"])
    lab=args.label or ("backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts"))

    # split time (busy-aware)
    busy=df[df[lab]>0]; uts=np.array(sorted((busy["timestamp"] if len(busy) else df["timestamp"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
    tr=df["timestamp"]<=split_t; te=df["timestamp"]>split_t

    # compute lead-1 predictions on test using filter fitted sequentially on all data
    preds=[]; truth=[]; ifaces=[]
    for iface, g in df.groupby("iface"):
        g=g.sort_values("timestamp")
        # run KF over whole series to produce lead-1 preds
        y=g[lab]
        yhat_lead = kf_lead1_series(y, Q=args.Q, R=args.R)
        # evaluate only on test timestamps (exclude the very first where pred is NaN)
        mask = te & (df["iface"]==iface)
        yh = yhat_lead.loc[g.index[mask.loc[g.index]]]
        yt = y.loc[g.index[mask.loc[g.index]]]
        preds.append(yh.values); truth.append(yt.values); ifaces.extend([iface]*len(yh))
    preds=np.concatenate([p for p in preds if len(p)])
    truth=np.concatenate([t for t in truth if len(t)])
    micro = rmse(pd.Series(truth), pd.Series(preds))

    # macro
    out = pd.DataFrame({"iface":ifaces, "y":truth, "yhat":preds})
    macro = macro_rmse(out.rename(columns={"y":lab}), lab, "yhat")

    print(f"Split time: {split_t}")
    print(f"KF lead-1 (Q={args.Q}, R={args.R}) → micro {micro:.3f} | macro {macro:.3f}")

if __name__=="__main__":
    main()
