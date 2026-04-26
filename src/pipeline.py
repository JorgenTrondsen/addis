import subprocess
from huggingface_hub import model_info
from transformers import AutoConfig

def calculate_pipeline(workers_latency_data: list, master_latency_data: dict, master_ip: str) -> list:
    """
    Determines the optimal pipeline sequence by using a greedy nearest neighbor approach.
    Starting from the master node (Rank 0), it iteratively adds the node with the lowest
    latency from the previous node's perspective, ensuring each node is added only once.
    Args:
        workers_latency_data (list): Latency maps and IPs from each worker.
        master_latency_data (dict): Latency map from the master node.
        master_ip (str): The IP address of the master node.
    Returns:
        list: Ordered list of IP addresses representing the pipeline (Rank 0, 1, ...).
    """
    graph = {}
    graph[master_ip] = master_latency_data

    for data in workers_latency_data:
        ip = data["ip"]
        graph[ip] = data["latency"]

    all_nodes = list(graph.keys())

    if len(all_nodes) <= 1:
        return all_nodes

    best_path = [master_ip]
    remaining_nodes = [node for node in all_nodes if node != master_ip]

    while remaining_nodes:
        last_node = best_path[-1]
        best_next_node = None
        min_lat = float('inf')

        for node in remaining_nodes:
            if last_node in graph and node in graph[last_node]:
                lat = graph[last_node][node]
                if lat < min_lat:
                    min_lat = lat
                    best_next_node = node

        if best_next_node is None:
            best_next_node = remaining_nodes[0]
            print(f"No direct latency found from {last_node}, picking next available: {best_next_node}")

        best_path.append(best_next_node)
        remaining_nodes.remove(best_next_node)

    total_latency = 0.0
    for i in range(len(best_path) - 1):
        current_node = best_path[i]
        next_node = best_path[i+1]
        if current_node in graph and next_node in graph[current_node]:
            total_latency += graph[current_node][next_node]

    print(f"Total estimated pipeline latency: {total_latency:.4f} ms")
    print(f"Determined pipeline order: {best_path}")

    return best_path


def get_hf_model_info(model_path: str) -> tuple[float, int]:
    """
    Fetches model size and layer count for a remote Hugging Face model.
    Returns:
        tuple: (model_size_gb, num_layers)
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers", 0)
    if num_layers == 0:
        num_layers = getattr(config, "n_layer", 0)

    info = model_info(model_path)
    total_size_bytes = info.used_storage

    model_size_gb = (total_size_bytes / (1000**3))

    return model_size_gb, num_layers


def calculate_partitions(nodes_data: list, pipeline_order: list, model_path: str) -> list:
    """
    Determines how many layers each rank in the pipeline should process.
    """
    model_size, model_layers = get_hf_model_info(model_path)
    node_map = {n["ip"]: n for n in nodes_data}

    total_vram = sum(node_map[ip]["vram"] for ip in pipeline_order)
    remaining_vram_buffer = (total_vram - model_size) / len(pipeline_order)

    partitions_list = []
    for ip in pipeline_order:
        node_memory = node_map[ip]["vram"]

        node_model_fraction = (node_memory - remaining_vram_buffer) / model_size
        node_layers = int(node_model_fraction * model_layers)
        partitions_list.append(max(0, node_layers))

    total_allocated = sum(partitions_list)
    remainder = model_layers - total_allocated

    idx = len(partitions_list) - 1
    while remainder > 0:
        partitions_list[idx] += 1
        remainder -= 1
        idx -= 1
        if idx < 0:
            idx = len(partitions_list) - 1

    print(f"Calculated layer partitions: {partitions_list} for model {model_path}")
    return [str(p) for p in partitions_list]


def calculate_usage(gpu_memory_utilization: float) -> float:
    """
    Calculates GPU memory usage based on nvidia-smi output and desired utilization percentage.
    Returns:
        float: Available VRAM in GB.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        total_vram_mib = float(result.stdout.strip().split('\n')[0].strip())
        total_vram_gb = total_vram_mib / 1024.0

        return total_vram_gb * gpu_memory_utilization
    except Exception as e:
        print(f"Error fetching GPU VRAM via nvidia-smi: {e}")
        return 16.0 * gpu_memory_utilization
