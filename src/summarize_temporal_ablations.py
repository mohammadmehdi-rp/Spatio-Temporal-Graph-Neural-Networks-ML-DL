import re, glob, os, numpy as np

pat = re.compile(r"TEST nowcast micro ([0-9.]+) \| macro ([0-9.]+)")
logs = sorted(glob.glob("temporal_logs/*.log"))

rows = []
for p in logs:
    txt = open(p, "r").read()
    m = pat.findall(txt)
    if not m:
        print("[WARN] no TEST line in", p)
        continue
    micro, macro = m[-1]
    tag = os.path.basename(p).replace(".log","")
    rows.append((tag, float(micro), float(macro)))

print("tag,micro,macro")
for r in rows:
    print(f"{r[0]},{r[1]:.3f},{r[2]:.3f}")
