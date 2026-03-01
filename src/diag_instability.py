#!/usr/bin/env python3
import numpy as np, argparse, json

def load(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_sparse", default="dataset_sparse_v4.npz")
    ap.add_argument("--teacher_preds", default="")
    ap.add_argument("--lead1_ckpt", default="")  # optional: to compute per-port errors, skip if you like
    args=ap.parse_args()

    S=load(args.data_sparse)
    nodes=[n.decode() if isinstance(n,bytes) else str(n) for n in S["nodes"]]
    Y=S["Y"]               # [T,N] raw pkts
    is_sensor=S["is_sensor"].astype(bool)
    tr,va,te=S["train_idx"],S["val_idx"],S["test_idx"]

    def busy_counts(idx):
        yy=Y[idx]  # [len, N]
        return (yy>0).sum(0)              # per-port count

    bc_train=busy_counts(tr)
    bc_val=busy_counts(va)
    bc_test=busy_counts(te)

    uncovered=[nodes[i] for i in range(len(nodes)) if bc_train[i]==0 and bc_test[i]>0]
    top_test = np.argsort(-bc_test)[:10]
    report={
      "sensors": [nodes[i] for i,v in enumerate(is_sensor) if v],
      "train_busy_total": int((Y[tr]>0).sum()),
      "val_busy_total": int((Y[va]>0).sum()),
      "test_busy_total": int((Y[te]>0).sum()),
      "test_top_ports": [{"iface":nodes[i], "test_busy": int(bc_test[i]), "train_busy": int(bc_train[i]), "is_sensor": bool(is_sensor[i])} for i in top_test],
      "uncovered_test_ports": uncovered
    }
    print(json.dumps(report, indent=2))

    # Optional: teacher vs truth on test (busy-only) to see an upper bound for *these splits*
    if args.teacher_preds:
        T=np.load(args.teacher_preds,allow_pickle=True)["Yhat"]
        yy=Y[te]; tt=T[te]; mask=(yy>0)
        if mask.any():
            rmse=np.sqrt(np.mean((tt[mask]-yy[mask])**2))
            print(f"Teacher→TEST busy-only RMSE: {rmse:.3f}")
        else:
            print("Teacher→TEST busy-only RMSE: N/A (no busy)")

if __name__=="__main__":
    main()
