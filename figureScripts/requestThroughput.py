import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

models = ["Qwen3-8B", "Qwen3-14B"]
datasets = ["ShareGPT", "WildGPT"]
req_rates = [4, 8, 32]

parallax_throughput = {
    "Qwen3-8B": {
        "ShareGPT": {4: 0.16, 8: 0.28, 32: 0.60},
        "WildGPT":  {4: 0.16, 8: 0.29, 32: 0.67}
    },
    "Qwen3-14B": {
        "ShareGPT": {4: 0.12, 8: 0.21, 32: 0.45},
        "WildGPT":  {4: 0.12, 8: 0.22, 32: 0.51}
    }
}

addis_throughput = {
    "Qwen3-8B": {
        "ShareGPT": {4: 0.20, 8: 0.38, 32: 1.42},
        "WildGPT":  {4: 0.20, 8: 0.37, 32: 1.39}
    },
    "Qwen3-14B": {
        "ShareGPT": {4: 0.14, 8: 0.26, 32: 0.96},
        "WildGPT":  {4: 0.14, 8: 0.25, 32: 0.94}
    }
}

fig, axes = plt.subplots(len(models), len(datasets), figsize=(18, 14), sharey='row')

x = np.arange(len(req_rates))
width = 0.35
x_offset = 0.23

for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]

        p_vals = [parallax_throughput[model][dataset][r] for r in req_rates]
        c_vals = [addis_throughput[model][dataset][r] for r in req_rates]

        rects1 = ax.bar(x - width/2, p_vals, width, label='Parallax', color='skyblue', edgecolor='black', linewidth=1.2, alpha=0.8)
        rects2 = ax.bar(x + width/2, c_vals, width, label='ADDIS', color='orange', edgecolor='black', linewidth=1.2, alpha=0.8)

        # Add percentage labels
        for k in range(len(req_rates)):
            p_val = p_vals[k]
            c_val = c_vals[k]
            percent_decrease = ((c_val - p_val) / c_val) * 100

            ax.text(x[k] - width/2 - x_offset, p_val + (max(c_vals)*0.02),
                    f"↓{percent_decrease:.0f}%",
                    ha='center', va='bottom', color='black', fontsize=38)

        ax.set_xlim(-0.75, len(req_rates) - 0.6)

        ax.yaxis.set_major_locator(MultipleLocator(0.25))

        if j == 0:
            ax.set_ylabel("Throughput (req/s)", fontsize=38)
            ax.tick_params(axis='y', labelsize=38)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

        if i == len(models) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r}" for r in req_rates], fontsize=38)
            ax.set_xlabel("Request Rate", fontsize=38)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

        if i == 0:
            ax.set_title(f"{dataset}", fontsize=38, pad=20)

        if j == len(datasets) - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(f"{model}", rotation=270, fontsize=38, labelpad=40)

fig.legend([rects1, rects2], ["Parallax", "ADDIS"], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=38)

plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.92], h_pad=5.0, w_pad=2.0)

plt.savefig("request_throughput.png", bbox_inches='tight')
print("Saved request_throughput.png")