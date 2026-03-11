import argparse
import socket
import json
import time
import subprocess
import sys
import os

from network import get_network_latency, get_node_ip
from pipeline import calculate_pipeline

def send_msg(sock, msg_dict):
    """
    Serializes a dictionary to JSON and sends it over a socket with a 4-byte length prefix.
    Args:
        sock (socket.socket): The destination socket.
        msg_dict (dict): The message content to send.
    """
    data = json.dumps(msg_dict).encode('utf-8')
    sock.sendall(len(data).to_bytes(4, byteorder='big'))
    sock.sendall(data)

def recv_msg(sock):
    """
    Receives a JSON-encoded message from a socket, preceded by a 4-byte length prefix.
    Args:
        sock (socket.socket): The source socket.
    Returns:
        dict: The decoded message dictionary, or None if connection closed.
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = int.from_bytes(raw_msglen, byteorder='big')
    data = recvall(sock, msglen)
    return json.loads(data.decode('utf-8'))

def recvall(sock, n):
    """
    Helper to receive exactly n bytes from a socket.
    Args:
        sock (socket.socket): The source socket.
        n (int): Number of bytes to receive.
    Returns:
        bytearray: The received bytes.
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def run_sglang_subprocess(role, args, assigned_rank, nnodes, pp_size, dist_init_addr):
    """
    Launches an sglang server instance in a subprocess with appropriate environment variables.
    Args:
        role (str): 'master' or 'worker'. Master binds to a public host/port.
        args (Namespace/dict): Configuration arguments for the model and distribution.
        assigned_rank (int): Distributed rank assigned to this node.
        nnodes (int): Total number of nodes in the pipeline.
        pp_size (int): Pipeline parallel size.
        dist_init_addr (str): Distributed initialization address (IP:PORT).
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sglang', 'python', 'sglang', 'launch_server.py')
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sglang', 'venv', 'bin', 'python')

    if os.path.exists(venv_python):
        python_exec = venv_python
    else:
        python_exec = sys.executable

    cmd = [
        python_exec,
        script_path,
        "--model-path", args.model_path if hasattr(args, "model_path") else args.get("model_path"),
        "--nnodes", str(nnodes),
        "--node-rank", str(assigned_rank),
        "--dist-init-addr", dist_init_addr,
        "--pp-size", str(pp_size),
        "--mem-fraction-static", "0.8"
    ]

    if role == 'master':
        cmd.extend(["--host", "0.0.0.0", "--port", "30000"])

    print(f"[{role}] Starting sglang server: {' '.join(cmd)}")

    env = os.environ.copy()
    env["NCCL_SOCKET_IFNAME"] = "tailscale0"
    env["GLOO_SOCKET_IFNAME"] = "tailscale0"

    venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sglang', 'venv', 'bin')
    if os.path.exists(venv_bin):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    sglang_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sglang', 'python')
    env["PYTHONPATH"] = sglang_python_dir + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(cmd, env=env)

def run_vllm_subprocess(role, args, pipeline_order, node_ip, master_ip):
    """
    Launches a vLLM instance using Ray for distributed execution.
    Args:
        role (str): 'master' (starts Ray head) or 'worker' (starts Ray worker node).
        args (dict): Configuration arguments including model path and overlay network info.
        pipeline_order (list): List of IP addresses defining the pipeline rank order.
        node_ip (str): IP address of the current node.
        master_ip (str): IP address of the Ray head node.
    """
    env = os.environ.copy()
    env["NCCL_SOCKET_IFNAME"] = args.get("overlay_network", "tailscale") + "0"
    env["GLOO_SOCKET_IFNAME"] = args.get("overlay_network", "tailscale") + "0"
    env["VLLM_HOST_IP"] = node_ip

    venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vllm', 'venv', 'bin')
    venv_py = os.path.join(venv_bin, 'python')

    if os.path.exists(venv_bin) and os.path.exists(venv_py):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        vllm_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vllm')
        env["PYTHONPATH"] = vllm_python_dir + os.pathsep + env.get("PYTHONPATH", "")
        ray_cmd_base = [venv_py, "-m", "ray.scripts.scripts"]
        vllm_cmd_base = [venv_py, "-m", "vllm.entrypoints.cli.main"]
    else:
        ray_cmd_base = ["ray"]
        vllm_cmd_base = ["vllm"]

    if role == 'worker':
        stop_cmd = ray_cmd_base + ["stop"]
        print(f"[{role}] Stopping existing ray instances: {' '.join(stop_cmd)}")
        subprocess.run(stop_cmd, env=env)
        time.sleep(2)

        cmd = ray_cmd_base + [
            "start", "--block",
            f"--address={master_ip}:6379",
            f"--node-ip-address={node_ip}"
        ]
        print(f"[{role}] Starting vllm worker (ray node): {' '.join(cmd)}")
        subprocess.run(cmd, env=env)

    elif role == 'master':
        stop_cmd = ray_cmd_base + ["stop"]
        print(f"[{role}] Stopping existing ray instances: {' '.join(stop_cmd)}")
        subprocess.run(stop_cmd, env=env)
        time.sleep(2)

        ray_cmd = ray_cmd_base + [
            "start", "--head", "--port=6379",
            f"--node-ip-address={node_ip}"
        ]
        print(f"[{role}] Starting vllm master ray head: {' '.join(ray_cmd)}")
        subprocess.run(ray_cmd, env=env, check=True)

        print(f"[{role}] Waiting for ray workers to connect...")
        time.sleep(5)

        env["VLLM_PP_RANK_ORDER"] = ",".join(pipeline_order)
        vllm_cmd = vllm_cmd_base + [
            "serve", args.get("model_path"),
            "--distributed-executor-backend", "ray",
            "--pipeline-parallel-size", str(len(pipeline_order)),
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.8",
            "--max-model-len", "2048"
        ]

        if args.get("kv_cache_size"):
            vllm_cmd.extend(["--kv-cache-memory-bytes", "1073741824"])

        print(f"[{role}] Starting vllm server: {' '.join(vllm_cmd)}")
        subprocess.run(vllm_cmd, env=env)

def master_mode(args):
    """
    Main execution loop for the master node. Orchestrates latency gathering from workers,
    calculates the optimal pipeline, and distributes rank assignments.
    Args:
        args (Namespace): CLI arguments containing config like number of workers and model path.
    """
    HOST = '0.0.0.0'
    PORT = 29999

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(args.num_workers)

    print(f"[Master] Listening for {args.num_workers} workers on port {PORT}...")

    worker_conns = []
    worker_addrs = []

    for i in range(args.num_workers):
        conn, addr = server_sock.accept()
        print(f"[Master] Accepted connection from {addr}")
        worker_conns.append(conn)
        worker_addrs.append(addr[0])

    config_msg = {"overlay_network": args.overlay_network, "inference_engine": args.inference_engine}
    for conn in worker_conns:
        send_msg(conn, config_msg)

    print("[Master] Sent overlay network config to workers. Waiting for latency maps...")

    print("[Master] Profiling my own latencies...")
    master_latency_data = get_network_latency(args.overlay_network)

    workers_latency_data = []

    for conn in worker_conns:
        print("waiting for worker latency data...")
        resp = recv_msg(conn)
        if resp and "latency" in resp and "ip" in resp:
            workers_latency_data.append(resp)
            print(f"[Master] Received latency map from worker {resp['ip']}")

    print("[Master] All worker data received. Calculating pipeline...")

    ts_ip = get_node_ip(args.overlay_network)

    pipeline_order = calculate_pipeline(workers_latency_data, master_latency_data, ts_ip)

    print(f"[Master] Determined pipeline order: {pipeline_order}")

    nnodes = args.num_workers + 1
    pp_size = nnodes

    dist_init_addr = f"{ts_ip}:20000"

    for i, conn in enumerate(worker_conns):
        worker_ip = workers_latency_data[i]["ip"]
        rank = pipeline_order.index(worker_ip)

        assign_msg = {
            "node_rank": rank,
            "nnodes": nnodes,
            "pp_size": pp_size,
            "dist_init_addr": dist_init_addr,
            "model_path": args.model_path,
            "inference_engine": args.inference_engine,
            "pipeline_order": pipeline_order,
            "master_ip": ts_ip
        }
        send_msg(conn, assign_msg)
        conn.close()

    master_rank = pipeline_order.index(ts_ip)
    print(f"[Master] Starting master node with rank {master_rank}")
    if args.inference_engine == "vllm":
        run_vllm_subprocess('master', vars(args), pipeline_order, ts_ip, ts_ip)
    else:
        run_sglang_subprocess('master', args, master_rank, nnodes, pp_size, dist_init_addr)
    server_sock.close()


def worker_mode(args):
    """
    Main execution loop for a worker node. Connects to the master, reports network latency,
    receives a rank assignment, and starts the inference backend.
    Args:
        args (Namespace): CLI arguments containing the master node's IP address.
    """
    master_ip = args.master_ip
    PORT = 29999

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print(f"[Worker] Connecting to master at {master_ip}:{PORT}...")
    connected = False
    for _ in range(10):
        try:
            sock.connect((master_ip, PORT))
            connected = True
            break
        except ConnectionRefusedError:
            time.sleep(2)

    if not connected:
        print("[Worker] Failed to connect to master.")
        return

    print("[Worker] Connected! Waiting for overlay network configuration...")
    config_msg = recv_msg(sock)
    overlay_network = config_msg.get("overlay_network", "tailscale")

    print(f"[Worker] Profiling network using {overlay_network}...")
    latency_map = get_network_latency(overlay_network)

    ts_ip = get_node_ip(overlay_network)

    resp_msg = {
        "ip": ts_ip,
        "latency": latency_map
    }

    send_msg(sock, resp_msg)
    print("[Worker] Sent latency map. Waiting for assignment...")

    assign_msg = recv_msg(sock)
    print(f"[Worker] Received assignment: {assign_msg}")
    sock.close()

    inference_engine = assign_msg.get("inference_engine", "sglang")
    if inference_engine == "vllm":
        run_vllm_subprocess(
            'worker',
            assign_msg,
            assign_msg["pipeline_order"],
            ts_ip,
            assign_msg["master_ip"]
        )
    else:
        run_sglang_subprocess(
            'worker',
            assign_msg,
            assign_msg["node_rank"],
            assign_msg["nnodes"],
            assign_msg["pp_size"],
            assign_msg["dist_init_addr"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CADIS Sglang Launcher")
    subparsers = parser.add_subparsers(dest="role", required=True)

    # Master
    master_parser = subparsers.add_parser("master")
    master_parser.add_argument("--num-workers", type=int, required=True)
    master_parser.add_argument("--model-path", type=str, required=True)
    master_parser.add_argument("--kv-cache-size", type=str, default="8G")
    master_parser.add_argument("--overlay-network", type=str, default="tailscale")
    master_parser.add_argument("--inference-engine", type=str, choices=["sglang", "vllm"], default="sglang")

    # Worker
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("master_ip", type=str)

    args = parser.parse_args()

    if args.role == "master":
        master_mode(args)
    elif args.role == "worker":
        worker_mode(args)

