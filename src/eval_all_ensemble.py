#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, TCNHead, GRUHead, NowcastHead

def load_npz(p):
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}

def rmse(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--task", choices=["lead1", "nowcast"], default="lead1")
    args = ap.parse_args()

    Z = load_npz(args.data)
    X = torch.from_numpy(Z["X"])          # [T,N,F]
    Y = torch.from_numpy(Z["Y"])          # [T,N]
    edges = torch.from_numpy(Z["edges"]).long()
    test_idx = torch.from_numpy(Z["test_idx"]).long()
    Fin, N = X.shape[2], X.shape[1]

    preds = []
    if args.task == "lead1":
        # Build predictions y_{t+1} for t in test_idx ∩ [K, T-2]
        valid_sets = []
        for path in args.ckpts:
            ck = torch.load(path, map_location="cpu"); m = ck["meta"]
            K = int(m.get("K", 20))
            enc = GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
            head = TCNHead(hid=m["hid"], K=K) if m.get("temporal", "tcn") == "tcn" else GRUHead(hid=m["hid"])
            enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
            enc.eval(); head.eval()

            valid = [t for t in range(K, X.shape[0] - 1) if t in set(test_idx.tolist())]
            yh = []
            with torch.no_grad():
                for t in valid:
                    H = [enc(X[s], edges) for s in range(t - K + 1, t + 1)]   # [K,N,H]
                    pred = head(torch.stack(H, 0))                             # [N]
                    yh.append(pred)
            preds.append(torch.stack(yh))   # [Ttest,N]
            valid_sets.append(valid)

        # Ensure all runs used the same valid positions (should match if K is same; otherwise intersect)
        base = set(valid_sets[0])
        for v in valid_sets[1:]:
            base &= set(v)
        valid = sorted(list(base))
        if not valid:
            print("Lead-1 ENSEMBLE all-frames RMSE: N/A (no valid test positions)")
            return

        # Re-generate aligned predictions for the intersected valid indices
        aligned = []
        for i, path in enumerate(args.ckpts):
            ck = torch.load(path, map_location="cpu"); m = ck["meta"]
            K = int(m.get("K", 20))
            enc = GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
            head = TCNHead(hid=m["hid"], K=K) if m.get("temporal", "tcn") == "tcn" else GRUHead(hid=m["hid"])
            enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
            enc.eval(); head.eval()
            yh = []
            with torch.no_grad():
                for t in valid:
                    H = [enc(X[s], edges) for s in range(t - K + 1, t + 1)]
                    yh.append(head(torch.stack(H, 0)))
            aligned.append(torch.stack(yh))  # [Tvalid,N]

        P = torch.mean(torch.stack(aligned), 0)     # [Tvalid,N]
        Ytrue = torch.stack([Y[t + 1] for t in valid])  # [Tvalid,N]
        score = rmse(P, Ytrue)
        print(f"Lead-1 ENSEMBLE all-frames RMSE: {score:.3f}")

    else:
        # NOWCAST: predict y_t at each t in test_idx, compare on all nodes
        preds_now = []
        for path in args.ckpts:
            ck = torch.load(path, map_location="cpu"); m = ck.get("meta", {})
            enc = GNNEncoder(Fin, hid=m.get("hid", 96), layers=m.get("layers", 3),
                             kind=m.get("encoder", "sage"), dropout=m.get("dropout", 0.2))
            head = NowcastHead(hid=m.get("hid", 96))
            enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
            enc.eval(); head.eval()
            yh = []
            with torch.no_grad():
                for t in test_idx.tolist():
                    pred = head(enc(X[t], edges))   # [N]
                    yh.append(pred)
            preds_now.append(torch.stack(yh))  # [Ttest,N]
        P = torch.mean(torch.stack(preds_now), 0)        # [Ttest,N]
        Ytrue = Y[test_idx]                               # [Ttest,N]
        score = rmse(P, Ytrue)
        print(f"Nowcast ENSEMBLE all-frames RMSE: {score:.3f}")

if __name__ == "__main__":
    main()
