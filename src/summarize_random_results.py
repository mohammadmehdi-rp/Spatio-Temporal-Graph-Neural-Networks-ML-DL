import re
import numpy as np
import os

log_dir = "random_logs"
k_list = [4, 6, 8, 10]
n_seeds = 5

pattern = re.compile(r"TEST nowcast micro ([0-9.]+) \| macro ([0-9.]+)")

for k in k_list:
    micro_vals = []
    macro_vals = []
    for s in range(n_seeds):
        log_path = os.path.join(log_dir, f"nowcast_rand_k{k}_seed{s}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] missing log: {log_path}")
            continue
        with open(log_path, "r") as f:
            text = f.read()
        matches = pattern.findall(text)
        if not matches:
            print(f"[WARN] no TEST line in {log_path}")
            continue
        micro, macro = matches[-1]  # last TEST line
        micro_vals.append(float(micro))
        macro_vals.append(float(macro))

    if micro_vals:
        micro_vals = np.array(micro_vals)
        macro_vals = np.array(macro_vals)
        print(f"\n=== k={k} sensors, {len(micro_vals)} runs ===")
        print(f"micro RMSE: mean={micro_vals.mean():.3f}  std={micro_vals.std():.3f}")
        print(f"macro RMSE: mean={macro_vals.mean():.3f}  std={macro_vals.std():.3f}")
    else:
        print(f"\n=== k={k}: no valid logs found ===")
