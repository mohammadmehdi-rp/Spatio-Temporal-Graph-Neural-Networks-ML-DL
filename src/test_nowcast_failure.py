#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, NowcastHead  # SAME as training (correct) :contentReference[oaicite:0]{index=0}

def load_npz(p):
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}

def find_idx(names, key):
    names=[n.decode() if isinstance(n,bytes) else str(n) for n in names]
    for i,n in enumerate(names):
        if n==key: return i
    raise KeyError(key)

def rmse(a,b):
    return float(torch.sqrt(torch.mean((a-b)**2)).item())

# =============================== Fault Testing =============================== #

def test_with_failure(data_npz, model_path, n_trials=5):
    Z = load_npz(data_npz)

    X = torch.from_numpy(Z["X"])         # [T,N,F]
    Y = torch.from_numpy(Z["Y"])         # [T,N]
    edges = torch.from_numpy(Z["edges"]).long()
    test_idx = torch.from_numpy(Z["test_idx"]).long()
    feat_names = Z["feat_names"]
    N = X.shape[1]

    # feature channels we zero out if a sensor fails (same channels used in training)
    ch_back = [find_idx(feat_names,k) for k in 
               ["sensor_backlog","sensor_backlog_lag1","sensor_backlog_lag2","sensor_backlog_lag3"]]
    ch_is_sensor = find_idx(feat_names,"is_sensor")

    # Build model as in training :contentReference[oaicite:1]{index=1}
    enc = GNNEncoder(X.shape[2], hid=96, layers=3, kind="sage", dropout=0.20)
    head = NowcastHead(hid=96)

    state = torch.load(model_path, map_location="cpu")
    enc.load_state_dict(state["enc"])
    head.load_state_dict(state["head"])
    enc.eval(); head.eval()

    # Determine which nodes are sensors (from is_sensor feature)
    baseline_pred=[]; baseline_true=[]
    with torch.no_grad():
        for t in test_idx.tolist():
            baseline_pred.append(head(enc(X[t],edges)))
            baseline_true.append(Y[t])
    baseline_pred=torch.stack(baseline_pred); baseline_true=torch.stack(baseline_true)
    base_rmse = rmse(baseline_pred.flatten(), baseline_true.flatten())

    sensor_nodes = [i for i in range(N) if X[0,i,ch_is_sensor]>0.5]

    failures=[]
    for trial in range(n_trials):
        import random
        bad = random.choice(sensor_nodes)

        X_fault = X.clone()
        # zero backlog features for FAILED SENSOR NODE
        X_fault[:,bad,ch_back]=0.0

        preds=[]; trues=[]
        with torch.no_grad():
            for t in test_idx.tolist():
                preds.append(head(enc(X_fault[t],edges)))
                trues.append(Y[t])
        preds=torch.stack(preds); trues=torch.stack(trues)

        fail_rmse = rmse(preds.flatten(), trues.flatten())
        failures.append(fail_rmse)
        print(f"[trial {trial}] failed={bad} → RMSE={fail_rmse:.3f}")

    print("\n=== Final Summary ===")
    print(f"Normal RMSE      = {base_rmse:.3f}")
    print(f"Failure RMSE mean= {np.mean(failures):.3f}  std={np.std(failures):.3f}\n")
    return base_rmse, failures

# =============================== CLI =============================== #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_trials", type=int, default=5)
    args=ap.parse_args()

    test_with_failure(args.npz, args.model, args.n_trials)
