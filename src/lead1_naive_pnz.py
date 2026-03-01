#!/usr/bin/env python3
# Lead-1 baseline: Naive lead  →  predict y_{t+1} = y_t
import argparse, numpy as np

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2)))
def macro_rmse(yhat, yref):
    N=yref.shape[1]; vals=[]
    for j in range(N): vals.append(rmse(yhat[:,j], yref[:,j]))
    return float(np.nanmean(vals))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    Y=Z["Y"]; test=Z["test_idx"]
    T,N=Y.shape
    t_valid=[t for t in test.tolist() if t+1<T]
    Yref=np.stack([Y[t+1] for t in t_valid])
    Yhat=np.stack([Y[t]   for t in t_valid])

    mic=rmse(Yhat, Yref); mac=macro_rmse(Yhat,Yref)
    print(f"Naive lead → all-frames micro {mic:.3f} | macro {mac:.3f} | test {len(t_valid)} | ports {N}")

if __name__=="__main__": main()
