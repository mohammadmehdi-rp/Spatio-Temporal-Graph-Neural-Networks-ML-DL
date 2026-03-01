#!/usr/bin/env python3
"""run_dumbbell_capture_plus.py

End-to-end capture for *multi-target* NDT experiments:

  - Queue backlog (packets)
  - Link throughput (Mbps)
  - Link utilization (ratio)
  - End-to-end latency (RTT ms) via ping

This is a drop-in sibling of src/run_dumbbell_capture.py. It produces:

  data.csv
  latency.csv            (optional)
  processed_plus.csv
  links.txt
  dataset_multi.npz      (Y has 4 channels)

Why this exists
--------------
Your existing pipeline estimates backlog using (primarily) counter-derived
features. For the *additional experiment* in the paper, we want to also report
estimation performance for utilization/throughput/latency, using the same graph
and the same train/val/test split logic.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from comnetsemu.net import Containernet
from mininet.node import Controller
from mininet.clean import cleanup

from simple_dumbbell import build_dumbbell


def require_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("[ERROR] Mininet/ComNetsEmu must run as root.")
        print("        Re-run with: sudo -E python3 src/run_dumbbell_capture_plus.py ...")
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
    host.cmd(f"pkill -f 'iperf3 -s' || true")
    host.cmd(f"iperf3 -s -D -p {port} >/tmp/iperf3_server.log 2>&1")
    time.sleep(0.5)


def run_udp_clients(senders, dst_ip, per_flow_mbps, duration_s, port=5201, tag="phase"):
    for h in senders:
        log = f"/tmp/iperf3_{tag}_{h.name}.log"
        cmd = (
            f"pkill -f 'iperf3 -c {dst_ip}' || true; "
            f"iperf3 -c {dst_ip} -u -b {per_flow_mbps}M -l 1200 -t {duration_s} -p {port} "
            f"--forceflush > {log} 2>&1 &"
        )
        h.cmd(cmd)
    time.sleep(duration_s + 0.5)


def write_links_file(path, s1_port="s1-eth1", s2_port="s2-eth1"):
    Path(path).write_text(f"{s1_port} {s2_port}\n")


def start_ping_trace(src_host, dst_ip: str, interval_s: float, out_path_in_container: str) -> None:
    """Start ping with unix timestamps (-D) and ms RTT, logging inside the container."""
    # -n numeric; -D timestamp; -i interval; -O report no-answer; -q no summary disabled
    src_host.cmd(f"pkill -f 'ping .* {dst_ip}' || true")
    src_host.cmd(f"ping -D -n -O -i {interval_s:.3f} {dst_ip} > {out_path_in_container} 2>&1 &")
    time.sleep(0.2)


def stop_ping_trace(src_host, dst_ip: str) -> None:
    src_host.cmd(f"pkill -f 'ping .* {dst_ip}' || true")


def parse_ping_log(ping_text: str) -> pd.DataFrame:
    """Parse ping -D output into a DataFrame(timestamp,rtt_ms)."""
    import re

    rows = []
    # Example line:
    # [1700000000.123456] 64 bytes from 10.0.0.5: icmp_seq=1 ttl=64 time=20.3 ms
    pat = re.compile(r"^\[(?P<ts>[0-9]+\.[0-9]+)\].*time=(?P<rtt>[0-9.]+)\s*ms", re.M)
    for m in pat.finditer(ping_text):
        ts = float(m.group("ts"))
        rtt = float(m.group("rtt"))
        rows.append((ts, rtt))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "rtt_ms"])
    df = pd.DataFrame(rows, columns=["timestamp", "rtt_ms"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def tc_qdisc_replace_tbf(iface: str, rate_mbit: float, burst_kbit: int = 32, latency_ms: int = 400) -> None:
    subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        [
            "tc",
            "qdisc",
            "add",
            "dev",
            iface,
            "root",
            "tbf",
            "rate",
            f"{rate_mbit}mbit",
            "burst",
            f"{burst_kbit}kbit",
            "latency",
            f"{latency_ms}ms",
        ],
        check=True,
    )


def main() -> None:
    require_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs/dumbbell_seed1_plus", help="output directory")
    ap.add_argument("--n", type=int, default=4, help="hosts per side")
    ap.add_argument(
        "--img",
        default=os.environ.get("COMNETSEMU_DIMAGE", "ndt/host:focal"),
        help="Docker image for hosts",
    )
    ap.add_argument("--freq", type=float, default=5.0, help="telemetry sampling frequency (Hz)")
    ap.add_argument("--bw_access", type=float, default=100.0)
    ap.add_argument("--delay_access", default="1ms")
    ap.add_argument("--bw_bottleneck", type=float, default=10.0)
    ap.add_argument("--delay_bottleneck", default="20ms")
    ap.add_argument("--force_tbf", action="store_true")
    ap.add_argument("--force_tbf_both", action="store_true")
    ap.add_argument("--tbf_burst_kbit", type=int, default=32)
    ap.add_argument("--tbf_latency_ms", type=int, default=400)

    ap.add_argument("--under_mbps", type=float, default=6.0)
    ap.add_argument("--over_mbps", type=float, default=14.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--t_under", type=float, default=30.0)
    ap.add_argument("--t_over", type=float, default=60.0)
    ap.add_argument("--t_idle", type=float, default=20.0)
    ap.add_argument("--bursts", type=int, default=4)
    ap.add_argument("--t_on", type=float, default=10.0)
    ap.add_argument("--t_off", type=float, default=10.0)
    ap.add_argument("--final_over", type=float, default=20.0)

    ap.add_argument("--ping", action="store_true", help="Enable ping RTT collection")
    ap.add_argument("--ping_interval", type=float, default=0.2, help="Ping interval (s)")

    ap.add_argument("--include_sensors", default="s1-eth1,s2-eth1,s1-eth2,s2-eth2")
    ap.add_argument("--sensor_frac", type=float, default=0.4)
    ap.add_argument(
        "--mask_rates_non_sensors",
        action="store_true",
        help="Zero-out rate features for non-sensors in dataset build",
    )
    args = ap.parse_args()

    if not docker_image_exists(args.img):
        print(f"[ERROR] Docker image not found locally: {args.img}")
        sys.exit(2)

    outdir = Path(args.outdir)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    data_csv = outdir / "data.csv"
    latency_csv = outdir / "latency.csv"
    processed_csv = outdir / "processed_plus.csv"
    links_txt = outdir / "links.txt"
    dataset_npz = outdir / "dataset_multi.npz"

    write_links_file(links_txt, "s1-eth1", "s2-eth1")

    # ---- Start network
    cleanup()
    net = Containernet(controller=Controller)
    net.addController("c0")
    left, right = build_dumbbell(
        net,
        n=args.n,
        img=args.img,
        bw_access=args.bw_access,
        delay_access=args.delay_access,
        bw_bottleneck=args.bw_bottleneck,
        delay_bottleneck=args.delay_bottleneck,
    )
    net.start()

    if args.force_tbf:
        print("[INFO] Forcing bottleneck TBF on s1-eth1")
        tc_qdisc_replace_tbf("s1-eth1", rate_mbit=args.bw_bottleneck, burst_kbit=args.tbf_burst_kbit, latency_ms=args.tbf_latency_ms)
        if args.force_tbf_both:
            print("[INFO] Also forcing TBF on s2-eth1")
            tc_qdisc_replace_tbf("s2-eth1", rate_mbit=args.bw_bottleneck, burst_kbit=args.tbf_burst_kbit, latency_ms=args.tbf_latency_ms)

    recv = right[0]
    recv_ip = recv.IP()
    start_iperf_server(recv)

    # ---- Start telemetry collector
    repo_root = Path(__file__).resolve().parents[1]
    collector = repo_root / "src" / "collector_5hz.py"
    proc = repo_root / "src" / "process_data_plus.py"
    prep = repo_root / "src" / "gnn_prep_multitarget.py"

    total_dur = args.warmup + args.t_under + args.t_over + args.t_idle + args.bursts * (args.t_on + args.t_off) + args.final_over
    coll_cmd = [sys.executable, str(collector), str(data_csv), "--freq", str(args.freq), "--pattern", r"^s[0-9]+-eth[0-9]+$", "--duration", str(total_dur)]
    coll_p = subprocess.Popen(coll_cmd)

    # ---- Optional ping trace
    ping_src = left[0]
    ping_log_in_container = "/tmp/ping_rtt.log"
    if args.ping:
        start_ping_trace(ping_src, recv_ip, args.ping_interval, ping_log_in_container)

    # ---- Traffic schedule
    time.sleep(args.warmup)
    per_flow = max(args.under_mbps / max(len(left), 1), 0.1)
    run_udp_clients(left, recv_ip, per_flow, args.t_under, tag="under")

    per_flow = max(args.over_mbps / max(len(left), 1), 0.1)
    run_udp_clients(left, recv_ip, per_flow, args.t_over, tag="over")

    time.sleep(args.t_idle)

    for b in range(args.bursts):
        per_flow = max(args.over_mbps / max(len(left), 1), 0.1)
        run_udp_clients(left, recv_ip, per_flow, args.t_on, tag=f"burst{b}_on")
        time.sleep(args.t_off)

    if args.final_over > 0:
        per_flow = max(args.over_mbps / max(len(left), 1), 0.1)
        run_udp_clients(left, recv_ip, per_flow, args.final_over, tag="final_over")

    # ---- Stop collector and ping
    coll_p.wait()
    recv.cmd("pkill -f 'iperf3 -s' || true")
    if args.ping:
        stop_ping_trace(ping_src, recv_ip)

    # Pull ping log and write latency CSV
    if args.ping:
        txt = ping_src.cmd(f"cat {ping_log_in_container} || true")
        lat = parse_ping_log(txt)
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
        "--links_file",
        str(links_txt),
        "--fraction",
        str(args.sensor_frac),
        "--include",
        args.include_sensors,
        "--bw_access",
        str(args.bw_access),
        "--bw_bottleneck",
        str(args.bw_bottleneck),
        "--out",
        str(dataset_npz),
    ]
    if args.mask_rates_non_sensors:
        prep_cmd.append("--mask_rates_non_sensors")
    sh(prep_cmd, check=True)

    print("\n[OK] Multi-target dumbbell dataset ready:")
    print(f"  {dataset_npz}")


if __name__ == "__main__":
    main()
