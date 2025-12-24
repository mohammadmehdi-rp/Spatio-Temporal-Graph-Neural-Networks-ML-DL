import re, glob, os
import numpy as np

# Extract setting + seed from filename safely
name_re = re.compile(r"^nowcast_10_(full|nolags|lag1|lag12)_seed(\d+)\.log$")

# Extract metrics from the log content
pat = re.compile(r"TEST nowcast micro ([0-9.]+) \| macro ([0-9.]+)")

pretty = {
    "nolags": "No lags",
    "lag1": "Lag1 only",
    "lag12": "Lag1+Lag2",
    "full": "Full (Lag1-3)",
}

data = {k: {"micro": [], "macro": []} for k in pretty}

logs = glob.glob("temporal_seed_logs/*.log")
unmatched = []

for lp in logs:
    base = os.path.basename(lp)
    mname = name_re.match(base)
    if not mname:
        unmatched.append(base)
        continue
    key = mname.group(1)

    txt = open(lp, "r", encoding="utf-8", errors="ignore").read()
    m = pat.findall(txt)
    if not m:
        print("[WARN] no TEST line in", base)
        continue
    micro, macro = map(float, m[-1])
    data[key]["micro"].append(micro)
    data[key]["macro"].append(macro)

if unmatched:
    print("[INFO] Unmatched log files (ignored):")
    for u in sorted(unmatched):
        print(" ", u)

print("=== Temporal ablations (seeds summary) ===")
print("setting, micro_mean, micro_std, macro_mean, macro_std, n")
order = ["nolags", "lag1", "lag12", "full"]
for key in order:
    micro = np.array(data[key]["micro"], dtype=float)
    macro = np.array(data[key]["macro"], dtype=float)
    if len(micro) == 0:
        print(f"{pretty[key]}, NA, NA, NA, NA, 0")
        continue
    print(f"{pretty[key]}, {micro.mean():.3f}, {micro.std():.3f}, {macro.mean():.3f}, {macro.std():.3f}, {len(micro)}")
