import re, glob, os
import numpy as np

pat = re.compile(r"TEST nowcast micro ([0-9.]+) \| macro ([0-9.]+)")

# Map file prefixes to pretty names
groups = {
    "nowcast_10_nolags": "No lags",
    "nowcast_10_lag1": "Lag1 only",
    "nowcast_10_lag12": "Lag1+Lag2",
    "nowcast_10_full": "Full (Lag1-3)",
}

logs = glob.glob("temporal_seed_logs/*.log")

data = {k: {"micro": [], "macro": []} for k in groups}

for lp in logs:
    base = os.path.basename(lp).replace(".log","")
    txt = open(lp, "r", encoding="utf-8", errors="ignore").read()
    m = pat.findall(txt)
    if not m:
        print("[WARN] no TEST line in", lp)
        continue
    micro, macro = map(float, m[-1])

    for prefix in groups:
        if base.startswith(prefix):
            data[prefix]["micro"].append(micro)
            data[prefix]["macro"].append(macro)
            break

print("=== Temporal ablations (3 seeds) ===")
print("setting, micro_mean, micro_std, macro_mean, macro_std, n")
for prefix, label in groups.items():
    micro = np.array(data[prefix]["micro"], dtype=float)
    macro = np.array(data[prefix]["macro"], dtype=float)
    if len(micro) == 0:
        print(f"{label}, NA, NA, NA, NA, 0")
        continue
    print(f"{label}, {micro.mean():.3f}, {micro.std():.3f}, {macro.mean():.3f}, {macro.std():.3f}, {len(micro)}")
