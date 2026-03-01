from comnetsemu.net import Containernet
from comnetsemu.node import DockerHost
from mininet.node import Controller
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.clean import cleanup

"""
simple_dumbbell.py
ComNetsEmu/Containernet dumbbell topology (canonical bottleneck experiment).

Left senders  h1..hN  -- s1 --(bottleneck)-- s2 -- h(N+1)..h(2N)  right receivers

Notes:
- We add the s1<->s2 link FIRST so it becomes s1-eth1 and s2-eth1 (useful for tc/telemetry).
- Hosts are Docker containers (default image ndt/host:jammy).
"""

def build_dumbbell(net, n=4, img="ndt/host:jammy",
                   bw_access=100, delay_access="1ms",
                   bw_bottleneck=10, delay_bottleneck="20ms"):
    s1 = net.addSwitch("s1")
    s2 = net.addSwitch("s2")

    # Bottleneck link (added first → s1-eth1 / s2-eth1)
    net.addLink(s1, s2, cls=TCLink, bw=bw_bottleneck, delay=delay_bottleneck)

    left = []
    right = []
    # Left hosts: 10.0.0.1..n
    for i in range(1, n + 1):
        h = net.addDockerHost(
            f"h{i}",
            dimage=img,
            ip=f"10.0.0.{i}/24",
            docker_args={"hostname": f"h{i}"}
        )
        net.addLink(h, s1, cls=TCLink, bw=bw_access, delay=delay_access)
        left.append(h)

    # Right hosts: 10.0.0.(n+1)..2n
    for j in range(1, n + 1):
        idx = n + j
        h = net.addDockerHost(
            f"h{idx}",
            dimage=img,
            ip=f"10.0.0.{idx}/24",
            docker_args={"hostname": f"h{idx}"}
        )
        net.addLink(h, s2, cls=TCLink, bw=bw_access, delay=delay_access)
        right.append(h)

    return left, right

if __name__ == "__main__":
    cleanup()
    net = Containernet(controller=Controller)
    net.addController("c0")
    left, right = build_dumbbell(net)
    net.start()

    print("\n[DUMBBELL] Up. Inter-switch (bottleneck) interfaces are typically s1-eth1 and s2-eth1.")
    print("Example traffic (left->right):")
    print("  h5 iperf3 -s -D")
    print("  h1 iperf3 -c 10.0.0.5 -u -b 3M -t 30\n")
    CLI(net)
    net.stop()
