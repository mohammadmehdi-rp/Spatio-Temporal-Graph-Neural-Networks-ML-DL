#!/usr/bin/env python3
import argparse, numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args=ap.parse_args()
    Z=np.load(args.data, allow_pickle=True)
    Y=Z["Y"]; test=Z["test_idx"]
    Yt=Y[test]  # [Ttest, N]
    print("test frames:", int(len(test)))
    print("label stats on test (pooled): mean=%.3f std=%.3f min=%.3f p50=%.3f p90=%.3f max=%.3f"
          % (Yt.mean(), Yt.std(), Yt.min(), np.median(Yt), np.quantile(Yt,0.9), Yt.max()))
    nz = Yt[Yt>0]
    if nz.size:
        print("nonzero-only: mean=%.3f std=%.3f min=%.3f p50=%.3f p90=%.3f max=%.3f"
              % (nz.mean(), nz.std(), nz.min(), np.median(nz), np.quantile(nz,0.9), nz.max()))
    else:
        print("nonzero-only: none")
if __name__=="__main__":
    main()
