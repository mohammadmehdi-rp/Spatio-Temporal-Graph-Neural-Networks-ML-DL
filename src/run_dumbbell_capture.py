#!/usr/bin/env python3
"""
run_dumbbell_capture.py
End-to-end: build dumbbell in ComNetsEmu, run controlled traffic, collect telemetry, and build an NPZ dataset.

Outputs (under --outdir):
  data.csv            raw 5 Hz counters + tc backlog per switch iface
  processed.csv       per-second derived features + chosen label
  links.txt           inter-switch port connectivity (for GNN edges)
  dataset.npz         ready for src/*_npz.py evaluation/training

Why this is defendable:
- Dumbbell is a standard topology for congestion/backlog experiments.
- Bottleneck is explicit (s1<->s2), with repeatable load regimes.
- Same splits / same sensor mask across methods (your NPZ pipeline already supports this).
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

from comnetsemu.net import Containernet
from mininet.node import Controller
from mininet.link import TCLink
from mininet.clean import cleanup

from simple_dumbbell import build_dumbbell


def require_root():
    """Mininet/OVS require root privileges."""
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("[ERROR] Mininet/ComNetsEmu must run as root.")
        print("        Re-run with: sudo -E python3 src/run_dumbbell_capture.py ...")
        sys.exit(1)


def docker_image_exists(img: str) -> bool:
    """Return True if Docker image exists locally."""
    try:
        subprocess.run(["docker", "image", "inspect", img],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def print_docker_images_hint():
    print("[HINT] List available images with: sudo docker images | head")
    print("[HINT] If you have ndt/host:focal, run with: --img ndt/host:focal")
    print("[HINT] Or build a minimal host image (example):")
    print("       cat > Dockerfile.ndt_host <<'EOF'")
    print("       FROM ubuntu:20.04")
    print("       RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \\")
    print("           iproute2 iputils-ping iperf3 tcpdump ethtool && rm -rf /var/lib/apt/lists/*")
    print("       CMD [\"/bin/bash\"]")
    print("       EOF")
    print("       sudo docker build -t ndt/host:focal -f Dockerfile.ndt_host .")

def sh(cmd, check=True):
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def start_iperf_server(host, port=5201):
    host.cmd(f"pkill -f 'iperf3 -s' || true")
    host.cmd(f"iperf3 -s -D -p {port} >/tmp/iperf3_server.log 2>&1")
    time.sleep(0.5)

def run_udp_clients(senders, dst_ip, per_flow_mbps, duration_s, port=5201, tag="phase"):
    """Run UDP iperf3 clients from `senders` to `dst_ip`.

    IMPORTANT: This command is executed *inside Docker hosts*.
    Therefore, we must not redirect logs to a host filesystem path (it doesn't exist inside containers).
    We always log to /tmp inside the container to avoid silent failures.
    """
    for h in senders:
        log = f"/tmp/iperf3_{tag}_{h.name}.log"
        # -u UDP; -b rate; -l 1200 typical payload; --forceflush for better timestamped logs
        cmd = (
            f"pkill -f 'iperf3 -c {dst_ip}' || true; "
            f"iperf3 -c {dst_ip} -u -b {per_flow_mbps}M -l 1200 -t {duration_s} -p {port} "
            f"--forceflush > {log} 2>&1 &"
        )
        h.cmd(cmd)
    time.sleep(duration_s + 0.5)

def write_links_file(path, s1_port="s1-eth1", s2_port="s2-eth1"):
    Path(path).write_text(f"{s1_port} {s2_port}\n")


def tc_qdisc_show(iface: str) -> str:
    """Return tc qdisc output (best-effort, for debugging)."""
    try:
        return subprocess.check_output(["tc", "-s", "qdisc", "show", "dev", iface], text=True)
    except Exception:
        return ""


def tc_qdisc_replace_tbf(iface: str, rate_mbit: float, burst_kbit: int = 32, latency_ms: int = 400) -> None:
    """Force a measurable bottleneck queue using TBF on iface.

    This is helpful when TCLink shaping does not create a visible backlog in `tc -s qdisc`.
    Note: replacing root qdisc may override any netem delay on the same interface.
    """
    # Delete root if present (ignore errors), then add TBF.
    subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([
        "tc", "qdisc", "add", "dev", iface, "root", "tbf",
        "rate", f"{rate_mbit}mbit",
        "burst", f"{burst_kbit}kbit",
        "latency", f"{latency_ms}ms",
    ], check=True)

def main():
    require_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs/dumbbell_seed1", help="output directory")
    ap.add_argument("--n", type=int, default=4, help="hosts per side")
    # Default matches the existing tree experiments in this repo.
    ap.add_argument("--img", default=os.environ.get("COMNETSEMU_DIMAGE", "ndt/host:focal"),
                    help="Docker image for hosts (default: env COMNETSEMU_DIMAGE or ndt/host:focal)")
    ap.add_argument("--freq", type=float, default=5.0, help="telemetry sampling frequency (Hz)")
    ap.add_argument("--bw_access", type=float, default=100.0)
    ap.add_argument("--delay_access", default="1ms")
    ap.add_argument("--bw_bottleneck", type=float, default=10.0)
    ap.add_argument("--delay_bottleneck", default="20ms")
    ap.add_argument("--force_tbf", action="store_true",
                    help="Replace root qdisc on s1-eth1 with TBF to ensure backlog is measurable")
    ap.add_argument("--force_tbf_both", action="store_true",
                    help="If set with --force_tbf, also apply TBF on s2-eth1 (recommended for dumbbell)")
    ap.add_argument("--tbf_burst_kbit", type=int, default=32)
    ap.add_argument("--tbf_latency_ms", type=int, default=400)
    ap.add_argument("--under_mbps", type=float, default=6.0, help="aggregate Mbps underload phase")
    ap.add_argument("--over_mbps", type=float, default=14.0, help="aggregate Mbps overload phase")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--t_under", type=float, default=30.0)
    ap.add_argument("--t_over", type=float, default=60.0)
    ap.add_argument("--t_idle", type=float, default=20.0)
    ap.add_argument("--bursts", type=int, default=4, help="number of ON/OFF burst cycles")
    ap.add_argument("--t_on", type=float, default=10.0)
    ap.add_argument("--t_off", type=float, default=10.0)
    ap.add_argument("--final_over", type=float, default=20.0,
                    help="Final overload phase duration (s) appended at the end to guarantee busy samples in the test tail")
    ap.add_argument("--include_sensors", default="s1-eth1,s2-eth1,s1-eth2,s2-eth2",
                    help="comma-list forced sensors passed to gnn_prep_v4.py")
    ap.add_argument("--sensor_frac", type=float, default=0.4, help="fraction sensors for automatic selection")
    args = ap.parse_args()

    # Fail fast with a readable message if the Docker host image is missing.
    if not docker_image_exists(args.img):
        print(f"[ERROR] Docker image not found locally: {args.img}")
        print_docker_images_hint()
        sys.exit(2)

    outdir = Path(args.outdir)
    (outdir/"logs").mkdir(parents=True, exist_ok=True)

    data_csv = outdir/"data.csv"
    processed_csv = outdir/"processed.csv"
    links_txt = outdir/"links.txt"
    dataset_npz = outdir/"dataset.npz"

    # Write inter-switch connectivity for GNN edges
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

    # Optional: force a measurable bottleneck queue using TBF.
    # This helps when TCLink bw/delay shaping doesn't yield visible backlog in tc stats.
    if args.force_tbf:
        print("[INFO] Forcing bottleneck TBF on s1-eth1 (may override netem delay on that iface)")
        tc_qdisc_replace_tbf("s1-eth1", rate_mbit=args.bw_bottleneck,
                             burst_kbit=args.tbf_burst_kbit,
                             latency_ms=args.tbf_latency_ms)
        qd = tc_qdisc_show("s1-eth1").strip()
        if qd:
            print("[INFO] tc -s qdisc show dev s1-eth1:\n" + qd)

        if args.force_tbf_both:
            print("[INFO] Also forcing TBF on s2-eth1 (recommended for dumbbell)")
            tc_qdisc_replace_tbf("s2-eth1", rate_mbit=args.bw_bottleneck,
                                 burst_kbit=args.tbf_burst_kbit,
                                 latency_ms=args.tbf_latency_ms)
            qd2 = tc_qdisc_show("s2-eth1").strip()
            if qd2:
                print("[INFO] tc -s qdisc show dev s2-eth1:\n" + qd2)

    # choose receiver as first right host (h{n+1})
    recv = right[0]
    recv_ip = recv.IP()

    # start iperf server
    start_iperf_server(recv)

    # start telemetry collector (collect switch ports only)
    repo_root = Path(__file__).resolve().parents[1]
    collector = repo_root/"src"/"collector_5hz.py"
    proc = repo_root/"src"/"process_data.py"
    prep = repo_root/"src"/"gnn_prep_v4.py"

    # Collector duration: cover the full schedule.
    total_dur = args.warmup + args.t_under + args.t_over + args.t_idle + args.bursts*(args.t_on+args.t_off) + args.final_over
    coll_cmd = [sys.executable, str(collector), str(data_csv),
                "--freq", str(args.freq),
                "--pattern", r"^s[0-9]+-eth[0-9]+$",
                "--duration", str(total_dur)]
    coll_p = subprocess.Popen(coll_cmd)

    # ---- Traffic schedule
    time.sleep(args.warmup)

    # Underload
    per_flow = max(args.under_mbps / max(len(left), 1), 0.1)
    run_udp_clients(left, recv_ip, per_flow, args.t_under, tag="under")

    # Overload
    per_flow = max(args.over_mbps / max(len(left), 1), 0.1)
    run_udp_clients(left, recv_ip, per_flow, args.t_over, tag="over")

    # Idle
    time.sleep(args.t_idle)

    # Bursty ON/OFF
    for b in range(args.bursts):
        per_flow = max(args.over_mbps / max(len(left), 1), 0.1)
        run_udp_clients(left, recv_ip, per_flow, args.t_on, tag=f"burst{b}_on")
        time.sleep(args.t_off)

    # Final overload tail: guarantees some busy frames near the end so that
    # chronological/test splits are not accidentally all-idle.
    if args.final_over > 0:
        per_flow = max(args.over_mbps / max(len(left), 1), 0.1)
        run_udp_clients(left, recv_ip, per_flow, args.final_over, tag="final_over")

    # ---- Stop collector
    coll_p.wait()

    # stop iperf server
    recv.cmd("pkill -f 'iperf3 -s' || true")

    net.stop()

    # ---- Build dataset (CSV -> processed -> NPZ)
    sh([sys.executable, str(proc), str(data_csv), "--out", str(processed_csv)], check=True)
    sh([sys.executable, str(prep),
        "--processed", str(processed_csv),
        "--links_file", str(links_txt),
        "--fraction", str(args.sensor_frac),
        "--include", args.include_sensors,
        "--out", str(dataset_npz)], check=True)

    # Sanity check: warn if backlog label never becomes > 0.
    try:
        import pandas as _pd
        _df = _pd.read_csv(processed_csv)
        if "backlog_pkts" in _df.columns and (_df["backlog_pkts"].max() <= 0):
            print("[WARN] backlog_pkts max is 0 in processed.csv — no congestion observed or tc backlog not captured.")
            print("       Try: --bw_bottleneck 2 --over_mbps 30, or enable --force_tbf.")
    except Exception:
        pass

    print("\n[OK] Dumbbell dataset ready:")
    print(f"  {dataset_npz}")

if __name__ == "__main__":
    main()
