import re, glob, os
import numpy as np

now_pat = re.compile(r"TEST nowcast micro ([0-9.]+) \| macro ([0-9.]+)")
lead_pat = re.compile(r"TEST lead-1 micro RMSE:\s*([0-9.]+)")

def collect(pattern, files, mode):
    out = {"sage": [], "routenet": []}
    for f in files:
        base = os.path.basename(f)
        enc = "sage" if "_sage_" in base else ("routenet" if "_routenet_" in base else None)
        if enc is None:
            continue
        txt = open(f, "r", encoding="utf-8", errors="ignore").read()
        m = pattern.findall(txt)
        if not m:
            print("[WARN] no match in", f)
            continue
        if mode == "nowcast":
            micro, macro = map(float, m[-1])
            out[enc].append((micro, macro))
        else:
            micro = float(m[-1])
            out[enc].append(micro)
    return out

now_files = sorted(glob.glob("enc_logs/nowcast_*.log"))
lead_files = sorted(glob.glob("enc_logs/lead1_*.log"))

now = collect(now_pat, now_files, "nowcast")
lead = collect(lead_pat, lead_files, "lead1")

print("=== Encoder variants: NOWCAST (k=10) ===")
for enc in ["sage","routenet"]:
    arr = np.array(now[enc], dtype=float)  # shape [n,2]
    if len(arr)==0:
        print(enc, "NA")
        continue
    micro = arr[:,0]; macro = arr[:,1]
    print(f"{enc}: micro {micro.mean():.3f} ± {micro.std():.3f} | macro {macro.mean():.3f} ± {macro.std():.3f} | n={len(arr)}")

print("\n=== Encoder variants: LEAD-1 (k=10) ===")
for enc in ["sage","routenet"]:
    arr = np.array(lead[enc], dtype=float)
    if len(arr)==0:
        print(enc, "NA")
        continue
    print(f"{enc}: micro {arr.mean():.3f} ± {arr.std():.3f} | n={len(arr)}")
