import itertools

def calculate_pipeline(workers_latency_data: list, master_latency_data: dict, master_ip: str) -> list:
    """
    Determines the optimal pipeline sequence by evaluating all node permutations.
    The goal is to minimize the total latency between consecutive nodes in the pipeline.
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

    workers = [node for node in all_nodes if node != master_ip]
    min_latency = float('inf')
    best_path = None

    for order in itertools.permutations(workers):
        current_path = [master_ip] + list(order)
        path_latency = 0

        for i in range(len(current_path) - 1):
            source_node = current_path[i]
            target_node = current_path[i + 1]
            if source_node in graph and target_node in graph[source_node]:
                path_latency += graph[source_node][target_node]
            else:
                path_latency += float('inf')

        if path_latency < min_latency:
            min_latency = path_latency
            best_path = current_path

    if best_path is None:
        best_path = [master_ip] + workers

    return best_path
