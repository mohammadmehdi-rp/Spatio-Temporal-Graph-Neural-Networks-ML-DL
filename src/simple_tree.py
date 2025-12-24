from comnetsemu.net import Containernet
from comnetsemu.node import DockerHost
from mininet.node import Controller
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.clean import cleanup

# Build a fixed depth-2 tree with fanout=4 (1 root switch, 4 child switches, 16 hosts)
# Each host is a Docker container (Ubuntu) attached to its child switch.

def build_tree(net, fanout=4, img="ndt/host:jammy"):
    s1 = net.addSwitch("s1")
    hosts = []
    sw_idx = 2
    host_idx = 1
    for i in range(fanout):
        si = net.addSwitch(f"s{sw_idx}")
        sw_idx += 1
        net.addLink(s1, si, cls=TCLink, bw=100, delay="5ms")
        # Attach 'fanout' hosts under each child switch
        for j in range(fanout):
            # Assign /24 per child, host ip .10+. 10.0.<child+1>.<host+10>
            ip_octet = i + 1
            ip = f"10.0.{ip_octet}.{10 + j}/24"
            h = net.addDockerHost(
    		name=f"h{host_idx}",
    		dimage="ndt/host:focal",
    		dcmd="/bin/bash",
    		docker_args={"cap_add": ["NET_ADMIN"]},
    		ip=ip,
	    )
            host_idx += 1
            hosts.append(h)
            net.addLink(si, h, cls=TCLink, bw=100, delay="1ms")
    return hosts

if __name__ == "__main__":
    cleanup()
    net = Containernet(controller=Controller)
    net.addController("c0")
    hosts = build_tree(net, fanout=4)
    net.start()
    print("Network is up. Example: use CLI to run iperf3 between hosts.")
    CLI(net)
    net.stop()
