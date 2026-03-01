import os
import subprocess

# 1. Create the directories (equivalent to mkdir -p)
os.makedirs("random_npz", exist_ok=True)
os.makedirs("random_logs", exist_ok=True)

# 2. Define your loops
k_values = [4, 6, 8, 10]
s_values = [0, 1, 2, 3, 4]

for k in k_values:
    for s in s_values:
        # Define filenames using f-strings
        sens_file = f"random_sensors/sensors_rand_k{k}_seed{s}.txt"
        npz_file  = f"random_npz/dataset_sparse_rand_k{k}_seed{s}.npz"
        log_file  = f"random_logs/nowcast_rand_k{k}_seed{s}.log"
        out_pt    = f"nowcast_rand_k{k}_seed{s}.pt"

        print(f"=== k={k}, seed={s} ===")

        # --- Step 1: Build Sparse Dataset ---
        # Note: We use "python" instead of "python3" for Windows standard
        subprocess.run([
            "python3", "build_sparse_dataset.py",
            "--npz_full", "dataset_full_v4.npz",
            "--sensors_txt", sens_file,
            "--out", npz_file
        ])

        # --- Step 2: Train Nowcast ---
        # We open the log file to save the output (equivalent to > log)
        with open(log_file, "w") as f:
            subprocess.run([
                "python3", "train_nowcast_sparse.py",
                "--data", npz_file,
                "--encoder", "sage",
                "--epochs", "80",
                "--lr", "2e-4",
                "--out", out_pt
            ], stdout=f, stderr=subprocess.STDOUT) 
            # stderr=subprocess.STDOUT captures errors into the log too
