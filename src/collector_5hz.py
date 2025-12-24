#!/usr/bin/env python3
"""
collector_5hz.py — poll switch ports at ~5 Hz and append rows to CSV.

Usage:
  python3 collector_5hz.py data.csv
  python3 collector_5hz.py data.csv --freq 5 --duration 120
  python3 collector_5hz.py data.csv --pattern '^s[0-9]+-eth[0-9]+$'
"""
import argparse, csv, os, re, subprocess, sys, time
from datetime import datetime, timezone

STAT_KEYS = ["rx_bytes","tx_bytes","rx_packets","tx_packets","rx_dropped","tx_dropped"]

def list_ifaces(pattern):
    pat = re.compile(pattern)
    out = subprocess.check_output(["ip","-o","link"], text=True)
    names = []
    for line in out.splitlines():
        # e.g., "7: s2-eth3@if23: <BROADCAST,...>"
        parts = line.split(": ", 2)
        if len(parts) >= 2:
            name = parts[1].split("@",1)[0]
            if pat.match(name):
                names.append(name)
    return sorted(set(names))

def read_sys_stat(iface, key):
    p = f"/sys/class/net/{iface}/statistics/{key}"
    try:
        with open(p,"r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None

def read_counters(iface):
    return {k: read_sys_stat(iface, k) for k in STAT_KEYS}

def read_tc_backlog(iface):
    try:
        out = subprocess.check_output(["tc","-s","qdisc","show","dev",iface], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return {"backlog_bytes": None, "backlog_pkts": None}
    # find first "backlog Xb Yp"
    m = re.search(r"backlog\s+(\d+)b\s+(\d+)p", out)
    if not m:
        return {"backlog_bytes": 0, "backlog_pkts": 0}
    return {"backlog_bytes": int(m.group(1)), "backlog_pkts": int(m.group(2))}

def ensure_header(path, fieldnames):
    exists = os.path.exists(path)
    if not exists or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", help="output CSV (appends)")
    ap.add_argument("--freq", type=float, default=5.0, help="samples per second")
    ap.add_argument("--pattern", default=r"^s[0-9]+-eth[0-9]+$", help="iface name regex")
    ap.add_argument("--ifaces", default="", help="comma list to pin interfaces (optional)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    args = ap.parse_args()

    ifaces = [x.strip() for x in args.ifaces.split(",") if x.strip()] or list_ifaces(args.pattern)
    if not ifaces:
        print("No interfaces matched.", file=sys.stderr); sys.exit(1)

    fields = ["timestamp","iface"] + STAT_KEYS + ["backlog_bytes","backlog_pkts"]
    ensure_header(args.out_csv, fields)

    period = 1.0/args.freq if args.freq > 0 else 0.2
    t0 = time.perf_counter()
    n = 0
    try:
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            while True:
                now = datetime.now(timezone.utc).isoformat()
                for iface in ifaces:
                    row = {"timestamp": now, "iface": iface}
                    row.update(read_counters(iface))
                    row.update(read_tc_backlog(iface))
                    w.writerow(row)
                f.flush()
                n += 1
                if args.duration and (time.perf_counter() - t0) >= args.duration:
                    break
                # simple pacing
                t_next = t0 + n*period
                dt = t_next - time.perf_counter()
                if dt > 0: time.sleep(dt)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
