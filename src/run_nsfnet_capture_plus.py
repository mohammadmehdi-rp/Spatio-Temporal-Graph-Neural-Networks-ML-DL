#!/usr/bin/env python3
"""run_nsfnet_capture_plus.py

End-to-end capture for NSFNET *multi-target* NDT experiments:

  - Queue backlog (packets)
  - Link throughput (Mbps)
  - Link utilization (ratio)
  - End-to-end latency (RTT ms) via ping (optional)

Compared to src/run_dumbbell_capture_plus.py, this script:
  1) Builds the 14-node, 21-link NSFNET topology.
  2) Applies distance-based propagation delays on core links.
  3) Installs static shortest-path forwarding rules (by km) to avoid L2 loops
     and make paths deterministic.

Outputs (in --outdir)
--------------------
  data.csv
  latency.csv                (if --ping)
  latency_multi.csv          (if --multi_ping)
  processed_plus.csv
  links.txt                  (core links only)
  dataset_multi.npz

Notes
-----
* Requires sudo/root (Mininet/ComNetsEmu).
* Uses the same collector_5hz/process_data_plus/gnn_prep_multitarget pipeline.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from comnetsemu.net import Containernet
from mininet.node import Controller
from mininet.clean import cleanup

from simple_nsfnet import build_nsfnet, install_static_shortest_path_flows, populate_static_arp


def require_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("[ERROR] Mininet/ComNetsEmu must run as root.")
        print("        Re-run with: sudo -E python3 src/run_nsfnet_capture_plus.py ...")
        sys.exit(1)


def docker_image_exists(img: str) -> bool:
    try:
        subprocess.run(["docker", "image", "inspect", img], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def sh(cmd, check=True):
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def start_iperf_server(host, port=5201):
    host.cmd(f"pkill -f 'iperf3 -s -p {port}' || true")
    host.cmd(f"iperf3 -s -D -p {port} >/tmp/iperf3_server_{port}.log 2>&1")
    time.sleep(0.2)


def run_udp_client(src, dst_ip: str, port: int, mbps: float, duration_s: float, tag: str) -> None:
    log = f"/tmp/iperf3_{tag}_{src.name}_to_{dst_ip}_{port}.log"
    src.cmd(
        " ".join(
            [
                f"pkill -f \"iperf3 -c {dst_ip} -p {port}\" || true;",
                f"iperf3 -c {dst_ip} -u -b {mbps}M -l 1200 -t {duration_s} -p {port} --forceflush > {log} 2>&1 &",
            ]
        )
    )


def start_ping_trace(src_host, dst_ip: str, interval_s: float, out_path_in_container: str) -> float:
    """Start an infinite ping in the background and return the local (python) start epoch.

    We intentionally *avoid* iputils' -D timestamp flag because some images ship BusyBox ping
    (or older iputils) that either don't support -D or emit a different format.

    Timestamps are reconstructed during parsing from (start_time + (seq-1)*interval).
    """

    src_host.cmd(f"pkill -f 'ping .* {dst_ip}' || true")
    # Truncate any prior log
    src_host.cmd(f": > {out_path_in_container}")
    # -n: numeric output, -O: report missed replies (optional; safe to ignore if unsupported)
    src_host.cmd(f"ping -n -O -i {interval_s:.3f} {dst_ip} > {out_path_in_container} 2>&1 &")
    t0 = time.time()
    time.sleep(0.2)
    return t0


def stop_ping_trace(src_host, dst_ip: str) -> None:
    src_host.cmd(f"pkill -f 'ping .* {dst_ip}' || true")


def parse_ping_log(ping_text: str, *, t0_epoch: float, interval_s: float) -> pd.DataFrame:
    """Parse ping output into a dataframe with UTC timestamps.

    Supported formats:
      * iputils ping with -D: [epoch] ... icmp_seq=... time=... ms
      * iputils/busybox without -D: ... icmp_seq=... time=... ms
      * busybox alternative: ... seq=... time=... ms

    If no explicit timestamp is present, we reconstruct it using icmp_seq/seq and the
    known ping interval.
    """

    import re

    rows = []

    # 1) If the line includes an epoch timestamp in square brackets (iputils -D)
    pat_ts = re.compile(r"^\[(?P<ts>[0-9]+(?:\.[0-9]+)?)\].*?time[=<]\s*(?P<rtt>[0-9.]+)\s*ms", re.M)
    for m in pat_ts.finditer(ping_text):
        rows.append((float(m.group('ts')), float(m.group('rtt'))))

    # 2) Otherwise, parse seq + RTT and reconstruct timestamps
    if not rows:
        pat_seq = re.compile(
            r"(?P<seq>(?:icmp_)?seq|seq)=(?P<n>[0-9]+).*?time[=<]\s*(?P<rtt>[0-9.]+)\s*ms"
        )
        # Try iputils form first (icmp_seq), then busybox (seq)
        pat_iputils = re.compile(r"icmp_seq=(?P<n>[0-9]+).*?time[=<]\s*(?P<rtt>[0-9.]+)\s*ms")
        pat_busybox = re.compile(r"\bseq=(?P<n>[0-9]+).*?time[=<]\s*(?P<rtt>[0-9.]+)\s*ms")

        for line in ping_text.splitlines():
            m = pat_iputils.search(line) or pat_busybox.search(line)
            if not m:
                continue
            seq = int(m.group('n'))
            rtt = float(m.group('rtt'))
            ts = t0_epoch + max(seq - 1, 0) * interval_s
            rows.append((ts, rtt))

    if not rows:
        return pd.DataFrame(columns=['timestamp', 'rtt_ms'])

    df = pd.DataFrame(rows, columns=['timestamp', 'rtt_ms'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    return df


def parse_probe_list(s: str, n_hosts: int = 14) -> list[tuple[int, int]]:
    """Parse probes like '0-13,1-12,2-11' into list[(src,dst)]."""
    out: list[tuple[int, int]] = []
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            raise ValueError(f"Invalid probe '{part}' (expected a-b)")
        a, b = part.split('-', 1)
        src, dst = int(a), int(b)
        if not (0 <= src < n_hosts and 0 <= dst < n_hosts) or src == dst:
            raise ValueError(f"Invalid probe '{part}' (src/dst out of range or equal)")
        out.append((src, dst))
    # keep deterministic order
    return sorted(set(out))


def default_probes() -> list[tuple[int, int]]:
    """A compact probe set with good geographic coverage on NSFNET."""
    return [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]


def choose_flows(n_hosts: int, k: int, seed: int) -> list[tuple[int, int]]:
    """Pick k distinct (src,dst) pairs, biased to longer distances by using far-apart indices."""
    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    # seed with some long-ish pairs
    for a, b in [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9)]:
        if a < n_hosts and b < n_hosts and a != b:
            pairs.add((a, b))
    # fill remaining randomly
    while len(pairs) < k:
        a = rng.randrange(0, n_hosts)
        b = rng.randrange(0, n_hosts)
        if a != b:
            pairs.add((a, b))
    return sorted(pairs)


def main() -> None:
    require_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs/nsfnet_seed1_plus", help="output directory")
    ap.add_argument("--seed", type=int, default=1, help="random seed for flow selection")
    ap.add_argument(
        "--img",
        default=os.environ.get("COMNETSEMU_DIMAGE", "ndt/host:focal-nettools"),
        help="Docker image for hosts",
    )

    # Telemetry
    ap.add_argument("--freq", type=float, default=5.0, help="telemetry sampling frequency (Hz)")

    # Link parameters
    ap.add_argument("--bw_access", type=float, default=1000.0)
    ap.add_argument("--delay_access", default="1ms")
    ap.add_argument("--bw_core", type=float, default=2.0)
    ap.add_argument("--km_per_ms", type=float, default=200.0)
    ap.add_argument("--delay_scale", type=float, default=1.0)

    # Traffic schedule
    ap.add_argument("--k_flows", type=int, default=8, help="number of simultaneous UDP flows")
    ap.add_argument("--under_mbps", type=float, default=6.0, help="total offered load in UNDER phase (Mbps)")
    ap.add_argument("--over_mbps", type=float, default=18.0, help="total offered load in OVER phase (Mbps)")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--t_under", type=float, default=30.0)
    ap.add_argument("--t_over", type=float, default=60.0)
    ap.add_argument("--t_idle", type=float, default=20.0)
    ap.add_argument("--bursts", type=int, default=4)
    ap.add_argument("--t_on", type=float, default=10.0)
    ap.add_argument("--t_off", type=float, default=10.0)
    ap.add_argument("--final_over", type=float, default=20.0)

    # Ping RTT
    ap.add_argument("--ping", action="store_true", help="Enable ping RTT collection (single OD pair)")
    ap.add_argument("--ping_src", type=int, default=0)
    ap.add_argument("--ping_dst", type=int, default=13)
    ap.add_argument("--ping_interval", type=float, default=0.2)

    ap.add_argument("--multi_ping", action="store_true", help="Collect RTT for multiple OD probes in parallel")
    ap.add_argument(
        "--probes",
        default="",
        help="Comma-separated probe list like '0-13,1-12,2-11'. If empty, uses a default set.",
    )

    # Dataset build
    ap.add_argument("--sensor_frac", type=float, default=0.25)
    ap.add_argument("--include_sensors", default="s0-eth2,s13-eth2", help="comma list of must-include sensors")
    ap.add_argument("--mask_rates_non_sensors", action="store_true")

    args = ap.parse_args()

    if not docker_image_exists(args.img):
        print(f"[ERROR] Docker image not found locally: {args.img}")
        sys.exit(2)

    outdir = Path(args.outdir)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    data_csv = outdir / "data.csv"
    latency_csv = outdir / "latency.csv"
    latency_multi_csv = outdir / "latency_multi.csv"
    processed_csv = outdir / "processed_plus.csv"
    links_txt = outdir / "links.txt"
    dataset_npz = outdir / "dataset_multi.npz"

    # ---- Start network
    cleanup()
    net = Containernet(controller=Controller)
    net.addController("c0")

    hosts, switches, port_map, links_lines = build_nsfnet(
        net,
        img=args.img,
        bw_access=args.bw_access,
        delay_access=args.delay_access,
        bw_core=args.bw_core,
        km_per_ms=args.km_per_ms,
        delay_scale=args.delay_scale,
    )

    # Write core links file deterministically
    links_txt.write_text("\n".join(links_lines) + "\n")

    net.start()

    # Static ARP + static L2 shortest-path routing
    populate_static_arp(hosts)
    host_macs = {i: hosts[i].MAC() for i in range(len(hosts))}
    install_static_shortest_path_flows(switches, host_macs, port_map, n_nodes=14)

    # ---- Flow selection and iperf servers
    flows = choose_flows(n_hosts=14, k=max(1, args.k_flows), seed=args.seed)
    # Assign ports per destination to allow multiple concurrent servers
    dst_to_port: dict[int, int] = {}
    for idx, (_s, d) in enumerate(flows):
        if d not in dst_to_port:
            dst_to_port[d] = 5201 + len(dst_to_port)
            start_iperf_server(hosts[d], port=dst_to_port[d])

    # ---- Start telemetry collector
    repo_root = Path(__file__).resolve().parents[1]
    collector = repo_root / "src" / "collector_5hz.py"
    proc = repo_root / "src" / "process_data_plus.py"
    prep = repo_root / "src" / "gnn_prep_multitarget.py"

    total_dur = args.warmup + args.t_under + args.t_over + args.t_idle + args.bursts * (args.t_on + args.t_off) + args.final_over
    coll_cmd = [sys.executable, str(collector), str(data_csv), "--freq", str(args.freq), "--pattern", r"^s[0-9]+-eth[0-9]+$", "--duration", str(total_dur)]
    coll_p = subprocess.Popen(coll_cmd)

    # ---- Optional ping trace(s)
    ping_jobs = []
    if args.multi_ping:
        # Multi-probe RTT collection (parallel pings)
        try:
            probes = parse_probe_list(str(args.probes), n_hosts=14) if str(args.probes).strip() else default_probes()
        except Exception as e:
            raise SystemExit(f"Invalid --probes: {e}")

        # Ensure the single-probe pair is included if --ping is also enabled.
        if args.ping:
            probes = sorted(set(probes + [(int(args.ping_src), int(args.ping_dst))]))

        for (s, d) in probes:
            src_h = hosts[int(s)]
            dst_ip = hosts[int(d)].IP()
            log_path = f"/tmp/ping_{s}_{d}.log"
            t0 = start_ping_trace(src_h, dst_ip, args.ping_interval, log_path)
            ping_jobs.append({"src": int(s), "dst": int(d), "dst_ip": str(dst_ip), "log": log_path, "t0": float(t0)})

    # Backward-compatible single-probe RTT (also emitted as latency.csv)
    ping_log_in_container = "/tmp/ping_rtt.log"
    ping_t0 = None
    if args.ping and not args.multi_ping:
        src = hosts[int(args.ping_src)]
        dst_ip = hosts[int(args.ping_dst)].IP()
        ping_t0 = start_ping_trace(src, dst_ip, args.ping_interval, ping_log_in_container)

    # ---- Traffic schedule
    def run_phase(total_mbps: float, duration_s: float, tag: str) -> None:
        if duration_s <= 0:
            return
        per_flow = max(float(total_mbps) / max(len(flows), 1), 0.1)
        for (s, d) in flows:
            run_udp_client(hosts[s], hosts[d].IP(), dst_to_port[d], per_flow, duration_s, tag)
        time.sleep(duration_s + 0.5)

    time.sleep(args.warmup)
    run_phase(args.under_mbps, args.t_under, "under")
    run_phase(args.over_mbps, args.t_over, "over")
    time.sleep(args.t_idle)

    for b in range(args.bursts):
        run_phase(args.over_mbps, args.t_on, f"burst{b}_on")
        time.sleep(args.t_off)

    run_phase(args.over_mbps, args.final_over, "final_over")

    # ---- Stop collector and ping
    coll_p.wait()
    # Stop iperf servers
    for d, p in dst_to_port.items():
        hosts[d].cmd(f"pkill -f 'iperf3 -s -p {p}' || true")

    if args.multi_ping:
        # Stop and parse all ping jobs
        frames = []
        for job in ping_jobs:
            s, d = int(job["src"]), int(job["dst"])
            src_h = hosts[s]
            dst_ip = job["dst_ip"]
            stop_ping_trace(src_h, dst_ip)
            txt = src_h.cmd(f"cat {job['log']} || true")
            (outdir / 'logs' / f"ping_raw_{s}_{d}.txt").write_text(txt)
            dfp = parse_ping_log(txt, t0_epoch=float(job["t0"]), interval_s=args.ping_interval)
            if len(dfp) > 0:
                dfp["src"] = s
                dfp["dst"] = d
                dfp["probe"] = f"{s}-{d}"
                frames.append(dfp)
        if frames:
            latm = pd.concat(frames, axis=0, ignore_index=True).sort_values(["probe", "timestamp"])
            latm.to_csv(latency_multi_csv, index=False)
            print(f"[INFO] latency_multi.csv rows={len(latm)} -> {latency_multi_csv}")
        else:
            print(f"[WARN] multi_ping enabled but no RTT samples parsed")

        # Also write a single-pair latency.csv if requested
        if args.ping:
            pair = f"{int(args.ping_src)}-{int(args.ping_dst)}"
            if frames:
                sub = latm[latm["probe"] == pair][["timestamp", "rtt_ms"]].copy()
            else:
                sub = pd.DataFrame(columns=["timestamp", "rtt_ms"])
            sub.to_csv(latency_csv, index=False)
            print(f"[INFO] latency.csv rows={len(sub)} -> {latency_csv}")

    elif args.ping:
        # Single-probe mode
        src = hosts[int(args.ping_src)]
        dst_ip = hosts[int(args.ping_dst)].IP()
        stop_ping_trace(src, dst_ip)
        txt = src.cmd(f"cat {ping_log_in_container} || true")
        # Persist raw ping output for debugging
        (outdir / 'logs' / 'ping_raw.txt').write_text(txt)
        lat = parse_ping_log(txt, t0_epoch=ping_t0, interval_s=args.ping_interval)
        lat.to_csv(latency_csv, index=False)
        print(f"[INFO] latency.csv rows={len(lat)} -> {latency_csv}")

    net.stop()

    # ---- Build processed + dataset
    if args.ping and latency_csv.exists():
        sh([sys.executable, str(proc), str(data_csv), "--out", str(processed_csv), "--latency_csv", str(latency_csv)], check=True)
    else:
        sh([sys.executable, str(proc), str(data_csv), "--out", str(processed_csv)], check=True)

    prep_cmd = [
        sys.executable,
        str(prep),
        "--processed",
        str(processed_csv),
        "--latency_multi_csv",
        str(latency_multi_csv),
        "--links_file",
        str(links_txt),
        "--fraction",
        str(args.sensor_frac),
        "--include",
        args.include_sensors,
        "--bw_access",
        str(args.bw_access),
        "--bw_bottleneck",
        str(args.bw_core),
        "--out",
        str(dataset_npz),
    ]
    if args.mask_rates_non_sensors:
        prep_cmd.append("--mask_rates_non_sensors")
    sh(prep_cmd, check=True)

    print("\n[OK] Multi-target NSFNET dataset ready:")
    print(f"  {dataset_npz}")


if __name__ == "__main__":
    main()
