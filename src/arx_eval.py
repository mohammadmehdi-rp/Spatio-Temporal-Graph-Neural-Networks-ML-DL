
#!/usr/bin/env python3
# ARX: predict backlog using per-port label lags (AR) + top-K global sensor features with lags.
import argparse, pandas as pd, numpy as np
from numpy.linalg import inv
def rmse(a,b): e=(a-b).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else float("nan")
def add_lags(df, cols, k, by=None):
    if by is None:
        for i in range(1,k+1):
            for c in cols: df[f"{c}_lag{i}"]=df[c].shift(i)
        return df
    # per-group lags (e.g., per iface)
    for i in range(1,k+1):
        for c in cols: df[f"{c}_lag{i}"]=df.groupby(by)[c].shift(i)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("processed")                 # processed.csv
    ap.add_argument("sensors_txt")              # sensors.txt
    ap.add_argument("--lags_y", type=int, default=10)     # AR label lags
    ap.add_argument("--lags_g", type=int, default=5)      # global feature lags
    ap.add_argument("--topk",   type=int, default=40)     # select top-K global feats by train corr
    ap.add_argument("--l2",     type=float, default=20000.0)
    ap.add_argument("--align",  default="S", help="time bucket (default 1s)")
    ap.add_argument("--subset_ifaces", default="", help="comma list to evaluate only a subset")
    args=ap.parse_args()

    df=pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    with open(args.sensors_txt) as f: sensors=[x.strip() for x in f if x.strip()]
    if not sensors: raise SystemExit("sensors.txt is empty")

    lab="backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
    rate_feats=[c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns: rate_feats.append("throughput_Mbps")

    # align to strict grid
    df["ts"]=df["timestamp"].dt.floor(args.align)
    ts_all=pd.date_range(df["ts"].min(), df["ts"].max(), freq=args.align)

    # GLOBAL sensor features (rates + sensor backlog), aligned + lagged
    parts=[]
    for s in sensors:
        sub=df[df["iface"]==s][["ts"]+rate_feats].drop_duplicates("ts").set_index("ts").reindex(ts_all).sort_index()
        sub=sub.fillna(method="ffill", limit=1)
        sub.columns=[f"{s}__{c}" for c in rate_feats]
        parts.append(sub)
    G_rate=pd.concat(parts, axis=1)

    partsY=[]
    for s in sensors:
        ys=df[df["iface"]==s][["ts",lab]].drop_duplicates("ts").set_index("ts").reindex(ts_all).sort_index()
        ys=ys.fillna(method="ffill", limit=1)
        ys.columns=[f"{s}__Y"]
        partsY.append(ys)
    G_y=pd.concat(partsY, axis=1)

    G=pd.concat([G_rate, G_y], axis=1); G=G.reset_index().rename(columns={"index":"ts"})

    # add lags to GLOBAL features
    gcols=[c for c in G.columns if c!="ts"]
    for i in range(1, args.lags_g+1):
        for c in list(gcols): G[f"{c}_lag{i}"]=G[c].shift(i)

    # BASE per-iface label + AR lags (target = current lab)
    base=df[["ts","iface",lab]].drop_duplicates(["ts","iface"]).sort_values(["iface","ts"]).copy()
    base=add_lags(base, [lab], args.lags_y, by="iface").dropna(subset=[f"{lab}_lag{args.lags_y}"])

    # merge: AR lags + GLOBAL (same ts)
    M=base.merge(G, on="ts", how="inner").dropna().sort_values(["ts","iface"])

    # optional subset
    subs=[s.strip() for s in args.subset_ifaces.split(",") if s.strip()]
    if subs: M=M[M["iface"].isin(subs)].copy()
    if M.empty: raise SystemExit("No rows after subset/merge; widen time or remove subset.")

    # split on busy timestamps if possible
    busy=M[M[lab]>0]; uts=np.array(sorted((busy["ts"] if len(busy) else M["ts"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else M["ts"].max()
    tr=M["ts"]<=split_t; te=M["ts"]>split_t

    # feature selection on GLOBAL feats by correlation with label (train only)
    ar_feats=[f"{lab}_lag{i}" for i in range(1, args.lags_y+1)]
    glb_feats=[c for c in M.columns if "__" in c]  # all global cols (with lags)
    dtr=M.loc[tr, [lab]+glb_feats].copy()
    corr=dtr.corr(numeric_only=True)[lab].abs().sort_values(ascending=False)
    keep=[c for c in corr.index if c in glb_feats][:args.topk]

    feats=ar_feats + keep
    cols=feats+[lab,"iface"]
    dtr=M.loc[tr, cols].dropna(); dte=M.loc[te, cols].dropna()
    if dtr.empty or dte.empty: raise SystemExit("Not enough train/test rows; adjust lags/topk or collect longer.")

    Xtr=dtr[feats].to_numpy(float); ytr=dtr[lab].to_numpy(float)
    Xte=dte[feats].to_numpy(float); yte=dte[lab].to_numpy(float)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.hstack([Xtr, np.ones((Xtr.shape[0],1))]); Xte=np.hstack([Xte, np.ones((Xte.shape[0],1))])
    w=inv(Xtr.T@Xtr + args.l2*np.eye(Xtr.shape[1])) @ (Xtr.T@ytr)
    yhat=Xte@w

    micro=rmse(pd.Series(yte), pd.Series(yhat))
    # macro RMSE
    dte2=dte.copy(); dte2["yhat"]=yhat
    vals=[]
    for _,g in dte2[[lab,"yhat","iface"]].dropna().groupby("iface"):
        if len(g)>=2: vals.append(rmse(g[lab], g["yhat"]))
    macro=float(np.nanmean(vals)) if vals else float("nan")

    print(f"ARX → micro {micro:.3f} | macro {macro:.3f} | train {len(dtr)} | test {len(dte)} | ARlags {args.lags_y} | Glags {args.lags_g} | topK {args.topk} | L2 {args.l2} | split {split_t}")

if __name__=="__main__": main()
