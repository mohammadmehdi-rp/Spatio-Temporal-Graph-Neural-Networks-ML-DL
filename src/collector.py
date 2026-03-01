import csv, os, re, subprocess, sys, time
from datetime import datetime

IFACE_PATTERNS = [re.compile(r"^s\d+-eth\d+$")]  # collect switch ports
STAT_KEYS = [
    "rx_bytes","tx_bytes","rx_packets","tx_packets","rx_dropped","tx_dropped"
]

BACKLOG_RE = re.compile(r"backlog\s+(\d+)b\s+(\d+)p")
QLEN_RE = re.compile(r"qlen\s+(\d+)")

STAT_PATH = "/sys/class/net/{iface}/statistics/{key}"

def list_ifaces():
    ifaces = []
    for name in os.listdir("/sys/class/net"):
        if any(p.match(name) for p in IFACE_PATTERNS):
            ifaces.append(name)
    return sorted(ifaces)

def read_stat(iface, key):
    with open(STAT_PATH.format(iface=iface, key=key), "r") as f:
        return int(f.read().strip())

def read_qdisc(iface):
    try:
        out = subprocess.check_output(["tc", "-s", "qdisc", "show", "dev", iface], text=True)
    except subprocess.CalledProcessError:
        return {"backlog_bytes": None, "backlog_pkts": None, "qlen_pkts": None}
    backlog_bytes = backlog_pkts = qlen = None
    m = BACKLOG_RE.search(out)
    if m:
        backlog_bytes = int(m.group(1))
        backlog_pkts = int(m.group(2))
    m2 = QLEN_RE.search(out)
    if m2:
        qlen = int(m2.group(1))
    return {"backlog_bytes": backlog_bytes, "backlog_pkts": backlog_pkts, "qlen_pkts": qlen}

def main(out_csv="data.csv", interval=1.0):
    ifaces = list_ifaces()
    if not ifaces:
        print("No switch interfaces found (pattern s*-eth*). Is the topology running?", file=sys.stderr)
        sys.exit(1)

    fields = [
        "timestamp","iface",
        *STAT_KEYS,
        "backlog_bytes","backlog_pkts","qlen_pkts"
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        while True:
            ts = datetime.utcnow().isoformat()
            for iface in ifaces:
                row = {"timestamp": ts, "iface": iface}
                for k in STAT_KEYS:
                    try:
                        row[k] = read_stat(iface, k)
                    except Exception:
                        row[k] = None
                row.update(read_qdisc(iface))
                w.writerow(row)
            f.flush()
            time.sleep(interval)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    main(out)
