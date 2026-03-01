#!/usr/bin/env python3
import argparse, numpy as np, csv

def rmse(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def metrics(yhat, y, thr):
    yhat=np.asarray(yhat,float); y=np.asarray(y,float)
    idle = y < thr
    busy = y >= thr
    out = {}
    out["global_RMSE"] = rmse(yhat.reshape(-1), y.reshape(-1))
    out["idle_RMSE"]   = rmse(yhat[idle], y[idle]) if np.any(idle) else float("nan")
    out["busy_RMSE"]   = rmse(yhat[busy], y[busy]) if np.any(busy) else float("nan")
    out["idle_FP_rate"]= float(np.mean(yhat[idle] >= thr)) if np.any(idle) else float("nan")
    out["idle_mean_pred"]= float(np.mean(yhat[idle])) if np.any(idle) else float("nan")
    out["n_idle"]= int(np.sum(idle))
    out["n_busy"]= int(np.sum(busy))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="dataset_sparse_*.npz")
    ap.add_argument("--calib_npz", default=None, help="calibration artifact with y_test/pred_* arrays (optional)")
    ap.add_argument("--busy_thr", type=float, default=50.0)
    ap.add_argument("--out_csv", default="outputs/summaries/nowcast_baselines.csv")
    args=ap.parse_args()

    Z=np.load(args.npz, allow_pickle=True)
    X=Z["X"].astype(np.float32)
    Y=Z["Y"].astype(np.float32)
    test=Z["test_idx"].astype(int)
    feat=[n.decode() if isinstance(n,bytes) else str(n) for n in Z["feat_names"]]
    mu=Z.get("norm_mu", None)
    sd=Z.get("norm_sd", None)

    def denorm(ch, arr):
        # arr shape [Ttest,N]
        if mu is None or sd is None: return arr
        return arr*sd[ch] + mu[ch]

    y_true = Y[test]

    rows=[]
    # baseline: always zero
    rows.append(("Zero", metrics(np.zeros_like(y_true), y_true, args.busy_thr)))

    # baseline: copy observed sensor_backlog (if present)
    if "sensor_backlog" in feat:
        ch=feat.index("sensor_backlog")
        yhat = denorm(ch, X[test,:,ch])
        rows.append(("Copy sensor_backlog", metrics(yhat, y_true, args.busy_thr)))

    # baseline: ar1_next feature (if present)
    if "ar1_next" in feat:
        ch=feat.index("ar1_next")
        yhat = denorm(ch, X[test,:,ch])
        rows.append(("AR1_next feature", metrics(yhat, y_true, args.busy_thr)))

    # optionally: include calibrated GNN predictions if provided
    if args.calib_npz:
        C=np.load(args.calib_npz, allow_pickle=True)
        yy=C["y_test"]
        for key,name in [("pred_raw","GNN (raw)"),("pred_scale","GNN (scale)"),("pred_soft","GNN (soft+scale)")]:
            if key in C.files:
                rows.append((name, metrics(C[key], yy, float(C.get("busy_thr", args.busy_thr)))))

    # write CSV
    out_path = args.out_csv
    header=["method","global_RMSE","idle_RMSE","busy_RMSE","idle_FP_rate","idle_mean_pred","n_idle","n_busy"]
    with open(out_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for m,d in rows:
            w.writerow([m,d["global_RMSE"],d["idle_RMSE"],d["busy_RMSE"],d["idle_FP_rate"],d["idle_mean_pred"],d["n_idle"],d["n_busy"]])
    print(f"[OK] wrote {out_path} ({len(rows)} rows)")

if __name__=="__main__":
    main()
