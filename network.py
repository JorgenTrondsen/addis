import subprocess
import re

def get_network_latency(overlay_network: str) -> dict:
    """
    Retrieves network latency for peers on the specified overlay network.
    Args:
        overlay_network (str): The name of the overlay network ('tailscale' or 'netbird').
    Returns:
        dict: A mapping of IP addresses to their measured latency in milliseconds.
    """
    if overlay_network == "tailscale":
        return _get_tailscale_latency()
    elif overlay_network == "netbird":
        return _get_netbird_latency()
    else:
        raise NotImplementedError(f"Overlay network '{overlay_network}' is not supported yet.")

def get_node_ip(overlay_network: str) -> str:
    """
    Returns the IPv4 address of the local node on the specified overlay network.
    Args:
        overlay_network (str): The name of the overlay network.
    Returns:
        str: The local node's IP address, or '127.0.0.1' if not found.
    """
    if overlay_network == "tailscale":
        try:
            return subprocess.check_output(['tailscale', 'ip', '-4']).decode('utf-8').strip()
        except Exception:
            pass
    elif overlay_network == "netbird":
        try:
            result = subprocess.check_output(['netbird', 'status']).decode('utf-8')
            for line in result.splitlines():
                if "NetBird IP:" in line:
                    return line.split("NetBird IP:")[1].strip().split("/")[0]
        except Exception:
            pass
    return "127.0.0.1"

def _get_netbird_latency() -> dict:
    """
    Internal helper to profile latencies for NetBird peers.
    """
    latency_map = {}

    try:
        result = subprocess.run(['netbird', 'status'], capture_output=True, text=True, check=True)
    except Exception as e:
        print(f"Error running netbird status: {e}")
        return latency_map

    ips = []
    current_ip = None
    is_online = False

    for line in result.stdout.splitlines():
        ip_match = re.search(r'NetBird IP:\s*(100\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
        if ip_match:
            if current_ip and is_online:
                ips.append(current_ip)
            current_ip = ip_match.group(1)
            is_online = False
        elif "Status:" in line and "Connected" in line:
            is_online = True

    if current_ip and is_online:
        ips.append(current_ip)

    ips = list(set(ips))

    import concurrent.futures

    def ping_ip(ip):
        try:
            ping_res = subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True, text=True)
            if ping_res.returncode == 0:
                match = re.search(r'time=([\d\.]+)\s*ms', ping_res.stdout)
                if not match:
                    match = re.search(r'min/avg/max/mdev = [\d\.]+/([\d\.]+)/[\d\.]+/[\d\.]+', ping_res.stdout)
                if match:
                    return ip, float(match.group(1))
        except Exception:
            pass
        return ip, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(ping_ip, ip): ip for ip in ips}
        for future in concurrent.futures.as_completed(futures):
            ip, lat = future.result()
            if lat is not None and lat > 0.0:
                latency_map[ip] = lat

    return latency_map

def _get_tailscale_latency() -> dict:
    """
    Internal helper to profile latencies for Tailscale peers using ping.
    """
    latency_map = {}

    try:
        result = subprocess.run(['tailscale', 'status'], capture_output=True, text=True, check=True)
    except Exception as e:
        print(f"Error running tailscale status: {e}")
        return latency_map

    ip_pattern = re.compile(r'\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    ips = []

    for line in result.stdout.splitlines():
        if "offline" in line.lower():
            continue
        match = ip_pattern.search(line)
        if match:
            ips.append(match.group(0))

    ips = list(set(ips))

    import concurrent.futures

    def ping_ip(ip):
        try:
            ping_res = subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True, text=True)
            if ping_res.returncode == 0:
                match = re.search(r'time=([\d\.]+)\s*ms', ping_res.stdout)
                if not match:
                    match = re.search(r'min/avg/max/mdev = [\d\.]+/([\d\.]+)/[\d\.]+/[\d\.]+', ping_res.stdout)
                if match:
                    return ip, float(match.group(1))
        except Exception:
            pass
        return ip, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(ping_ip, ip): ip for ip in ips}
        for future in concurrent.futures.as_completed(futures):
            ip, lat = future.result()
            if lat is not None and lat > 0.0:
                latency_map[ip] = lat

    return latency_map


if __name__ == "__main__":
    print(_get_tailscale_latency())
