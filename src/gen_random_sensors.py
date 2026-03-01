import numpy as np
import argparse
import os
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_full", default="dataset_full_v4.npz")
    ap.add_argument("--k_list", default="4,6,8,10")
    ap.add_argument("--n_sets", type=int, default=5)
    ap.add_argument("--out_dir", default="random_sensors")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    Z = np.load(args.npz_full, allow_pickle=True)
    nodes = Z["nodes"]
    N = nodes.shape[0]
    k_list = [int(x) for x in args.k_list.split(",") if x]

    os.makedirs(args.out_dir, exist_ok=True)

    for k in k_list:
        for s in range(args.n_sets):
            idx = rng.choice(N, size=k, replace=False)
            names = nodes[idx]
            fname = os.path.join(args.out_dir, f"sensors_rand_k{k}_seed{s}.txt")
            with open(fname, "w") as f:
                for n in names:
                    f.write(str(n) + "\n")
            print(f"[OK] {fname}: {names}")

if __name__ == "__main__":
    main()
