"""simple_nsfnet.py

NSFNET (14 nodes, 21 links) topology builder for ComNetsEmu/Containernet.

This module adds a *routed* forwarding fabric on top of an OVS switch mesh by
installing static OpenFlow rules for destination-based shortest paths.

Why OpenFlow rules?
-------------------
Your existing pipeline (collector_5hz.py + process_data*.py) polls Linux
interfaces named like `sX-ethY`. Those interfaces exist in the *root* namespace
when you use OVS switches in Mininet/ComNetsEmu. If we instead model routers as
Mininet hosts (network namespaces), their interfaces are *not* visible to the
collector without rewriting the telemetry path.

So we keep the core as OVS switches, but we avoid L2 loops and ensure
deterministic multi-hop paths by installing per-destination forwarding rules.

Distance-based delays
---------------------
Each core link has a length in km (from the standard NSFNET 14-node dataset
often used in RouteNet/Net2Vec papers). We convert km -> one-way propagation
delay assuming v_fiber ≈ 2e5 km/s (= 200 km/ms):

    delay_ms = km / 200

You can scale this with `delay_scale` to match a target RTT baseline.

Outputs
-------
build_nsfnet() returns:
  - hosts: list[DockerHost]
  - switches: dict[int, Switch]
  - port_map: dict[tuple[int,int], int] mapping (u,v)->port on u towards v
  - links_lines: list[str] lines for links.txt (core inter-switch links only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import subprocess

from comnetsemu.node import DockerHost
from mininet.link import TCLink


# ---------------------------------------------------------------------------
# NSFNET 14 nodes / 21 links with distances (km)
# Node IDs match the widely used "RouteNet" NSFNET numbering.
# ---------------------------------------------------------------------------

NSFNET_LINKS_KM: List[Tuple[int, int, int]] = [
    (0, 1, 260),
    (0, 2, 252),
    (0, 3, 324),
    (1, 2, 380),
    (1, 7, 868),
    (2, 5, 416),
    (3, 4, 248),
    (3, 10, 1140),
    (4, 5, 272),
    (4, 6, 292),
    (6, 7, 212),
    (7, 8, 224),
    (8, 9, 752),
    (5, 9, 704),
    (5, 12, 1036),
    (8, 11, 536),
    (8, 13, 668),
    (10, 11, 408),
    (11, 12, 664),
    (11, 13, 648),
    (12, 13, 352),
]


def km_to_delay_str(km: float, km_per_ms: float = 200.0, delay_scale: float = 1.0) -> str:
    """Convert km to a Mininet TCLink delay string (one-way), e.g. '3.210ms'."""
    ms = (float(km) / float(km_per_ms)) * float(delay_scale)
    return f"{ms:.3f}ms"


def _neighbors_from_links(links_km: Iterable[Tuple[int, int, int]]) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for u, v, _ in links_km:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return adj


def _build_port_map(adj: Dict[int, List[int]], host_port: int = 1, first_core_port: int = 2) -> Dict[Tuple[int, int], int]:
    """Reserve host_port for the attached host. Core ports start at first_core_port."""
    port_map: Dict[Tuple[int, int], int] = {}
    for u, nbrs in adj.items():
        p = first_core_port
        for v in sorted(nbrs):
            port_map[(u, v)] = p
            p += 1
        # host_port is implicitly reserved on each switch
    return port_map


def _shortest_paths_next_hop(n: int, links_km: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
    """Return next-hop map: next[(src,dst)] = neighbor of src on a shortest path to dst.

    Uses Dijkstra on the distance weights.
    """
    import heapq

    # adjacency with weights
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for u, v, km in links_km:
        adj[u].append((v, float(km)))
        adj[v].append((u, float(km)))

    next_hop: Dict[Tuple[int, int], int] = {}

    for dst in range(n):
        # Dijkstra from dst to all (so we can compute "towards dst" next hop)
        dist = {i: float("inf") for i in range(n)}
        prev = {i: None for i in range(n)}  # predecessor along shortest path tree rooted at dst
        dist[dst] = 0.0
        pq = [(0.0, dst)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        # For each src != dst, the next hop is prev[src] (one step closer to dst)
        for src in range(n):
            if src == dst:
                continue
            nh = prev[src]
            if nh is None:
                raise RuntimeError(f"NSFNET graph disconnected? no path {src}->{dst}")
            next_hop[(src, dst)] = int(nh)

    return next_hop


def install_static_shortest_path_flows(
    switches: Dict[int, object],
    host_macs: Dict[int, str],
    port_map: Dict[Tuple[int, int], int],
    n_nodes: int,
    links_km: List[Tuple[int, int, int]] = NSFNET_LINKS_KM,
    host_port: int = 1,
    flow_prio: int = 100,
) -> None:
    """Install per-destination L2 forwarding rules on each OVS switch.

    We match on destination MAC (dl_dst) and output to the next-hop port along
    the shortest path (by km). At the destination switch, we output to host_port.

    Assumes:
      - Each switch is named "s<id>" (bridge name matches switch.name)
      - Each host is attached to its switch on host_port
    """
    next_hop = _shortest_paths_next_hop(n_nodes, links_km)

    for sw_id, sw in switches.items():
        # Flush existing flows
        sw.cmd(f"ovs-ofctl -O OpenFlow13 del-flows {sw.name} || true")

    for dst in range(n_nodes):
        mac = host_macs[dst]
        for src in range(n_nodes):
            # For each switch 'src', install flow towards destination host 'dst'
            sw = switches[src]
            if src == dst:
                out_port = host_port
            else:
                nh = next_hop[(src, dst)]
                out_port = port_map[(src, nh)]

            sw.cmd(
                f"ovs-ofctl -O OpenFlow13 add-flow {sw.name} "
                f"\"priority={flow_prio},dl_dst={mac},actions=output:{out_port}\""
            )


def populate_static_arp(hosts: List[DockerHost]) -> None:
    """Populate permanent ARP entries so we avoid ARP broadcast in a loopy mesh.

    IMPORTANT (ComNetsEmu + DockerHost):
      Each DockerHost has *two* interfaces:
        - eth0 : Docker bridge (e.g., 172.17.0.0/16)
        - hX-eth0 : Mininet data-plane (e.g., 10.0.0.0/24)

      If ARP/neighbor entries are installed on the wrong device (eth0), packets to
      10.0.0.0/24 will still ARP on hX-eth0, broadcast will be dropped by our
      no-broadcast OpenFlow fabric, and you'll see:
        "Destination Host Unreachable" from the source.

    We therefore derive the correct egress device per-destination via:
      ip route get <dst_ip>
    which is robust even on minimal images (no awk/grep dependency).
    """
    import re as _re

    # Disable IPv6 chatter (optional but keeps traces cleaner)
    for h in hosts:
        h.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null 2>&1 || true")
        h.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1 >/dev/null 2>&1 || true")

    ips = {h.name: h.IP() for h in hosts}
    macs = {h.name: h.MAC() for h in hosts}

    def _route_dev(h: DockerHost, dst_ip: str) -> str:
        out = h.cmd(f"ip route get {dst_ip} 2>/dev/null").strip()
        m = _re.search(r"\bdev\s+(\S+)", out)
        return m.group(1) if m else "h0-eth0"

    for h in hosts:
        for other in hosts:
            if other is h:
                continue
            ip = ips[other.name]
            mac = macs[other.name]
            dev = _route_dev(h, ip)
            h.cmd(f"ip neigh replace {ip} lladdr {mac} dev {dev} nud permanent")


def build_nsfnet(
    net,
    img: str,
    bw_access: float = 1000.0,
    delay_access: str = "1ms",
    bw_core: float = 10.0,
    km_per_ms: float = 200.0,
    delay_scale: float = 1.0,
    links_km: List[Tuple[int, int, int]] = NSFNET_LINKS_KM,
) -> Tuple[List[DockerHost], Dict[int, object], Dict[Tuple[int, int], int], List[str]]:
    """Create NSFNET topology in a given Containernet.

    - Adds 14 switches: s0..s13
    - Adds 14 docker hosts: h0..h13, each attached to its corresponding switch
    - Adds the 21 core links with TCLink bw and distance-based delay
    - Returns the info needed to write links.txt and install flows
    """
    n_nodes = 14
    adj = _neighbors_from_links(links_km)
    port_map = _build_port_map(adj)

    switches: Dict[int, object] = {}
    for i in range(n_nodes):
        switches[i] = net.addSwitch(f"s{i}")

    # Core links with deterministic port numbers
    links_lines: List[str] = []
    for u, v, km in links_km:
        pu = port_map[(u, v)]
        pv = port_map[(v, u)]
        delay = km_to_delay_str(km, km_per_ms=km_per_ms, delay_scale=delay_scale)
        net.addLink(
            switches[u],
            switches[v],
            cls=TCLink,
            bw=bw_core,
            delay=delay,
            port1=pu,
            port2=pv,
        )
        links_lines.append(f"s{u}-eth{pu} s{v}-eth{pv}")

    # One host per node, attached to port1
    hosts: List[DockerHost] = []
    for i in range(n_nodes):
        h = net.addDockerHost(
            f"h{i}",
            dimage=img,
            ip=f"10.0.0.{i+1}/24",
            docker_args={"hostname": f"h{i}"},
        )
        net.addLink(h, switches[i], cls=TCLink, bw=bw_access, delay=delay_access, port2=1)
        hosts.append(h)

    return hosts, switches, port_map, links_lines
