import matplotlib.pyplot as plt

# Data
latencies = [40, 80, 160]
addis_throughput = [14.54, 9.27, 5.37]
parallax_throughput = [9.46, 6.26, 4.13]
sglang_throughput = [2.70, 1.45, 0.76]
vllm_throughput = [10.73, 6.1, 3.1]

plt.figure(figsize=(9, 6))

plt.plot(latencies, addis_throughput, marker='s', label='ADDIS(Sglang)', color='orange', linestyle='-', markeredgecolor='black', linewidth=3.5, markersize=10)
plt.plot(latencies, vllm_throughput, marker='D', label='ADDIS(vLLM)', color='green', linestyle='-', markeredgecolor='black', linewidth=3.5, markersize=10)
plt.plot(latencies, parallax_throughput, marker='o', label='Parallax', color='skyblue', linestyle='-', markeredgecolor='black', linewidth=3.5, markersize=12)
plt.plot(latencies, sglang_throughput, marker='^', label='Sglang', color='yellow', linestyle='-', markeredgecolor='black', linewidth=3.5, markersize=12)

# Data labels for ADDIS
for latency, c_val in zip(latencies, addis_throughput):
    plt.text(latency, c_val + 0.5, f'{c_val:.1f}', ha='center', va='bottom', fontsize=20, color='darkorange')

# Axis limits and labels
plt.ylim(0, 17)
plt.xlabel("Total Pipeline Latency (ms)", fontsize=22)
plt.ylabel("Single Request (tok/s)", fontsize=22)

plt.legend(fontsize=18, loc='upper right')
plt.xticks(latencies, fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig("single_request.png", bbox_inches='tight')