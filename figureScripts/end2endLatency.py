import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Units converted to seconds (/1000)
# Format: [Mean, P95, P96, P97, P98, P99, P100]

models = ["Qwen3-8B", "Qwen3-14B"]
datasets = ["ShareGPT", "WildGPT"]
req_rates = [4, 8, 32]
metrics = ["Mean", "P95", "P96", "P97", "P98", "P99", "P100"]

parallax_data = {
    "Qwen3-8B": {
        "ShareGPT": {
            4: [24.591, 24.612, 24.612, 24.612, 24.612, 24.612, 24.612],
            8: [28.373, 28.386, 28.386, 28.386, 28.386, 28.386, 28.386],
            32: [52.997, 53.004, 53.004, 53.004, 53.004, 53.004, 53.004]
        },
        "WildGPT": {
            4: [23.437, 24.673, 24.673, 24.673, 24.673, 24.673, 24.673],
            8: [25.160, 26.992, 26.992, 26.992, 26.993, 26.993, 26.993],
            32: [47.556, 47.828, 47.828, 47.828, 47.828, 47.828, 47.828]
        }
    },
    "Qwen3-14B": {
        "ShareGPT": {
            4: [33.134, 33.165, 33.165, 33.165, 33.165, 33.165, 33.165],
            8: [38.423, 38.441, 38.441, 38.441, 38.441, 38.441, 38.441],
            32: [70.544, 70.597, 70.597, 70.597, 70.598, 70.598, 70.598]
        },
        "WildGPT": {
            4: [29.068, 32.706, 32.706, 32.706, 32.707, 32.707, 32.707],
            8: [34.254, 36.196, 36.197, 36.197, 36.197, 36.197, 36.197],
            32: [62.372, 63.082, 63.082, 63.082, 63.082, 63.083, 63.083]
        }
    }
}

addis_data = {
    "Qwen3-8B": {
        "ShareGPT": {
            4: [19.889, 19.892, 19.892, 19.893, 19.893, 19.893, 19.893],
            8: [21.377, 21.389, 21.389, 21.390, 21.390, 21.390, 21.390],
            32: [22.516, 22.529, 22.530, 22.530, 22.530, 22.530, 22.531]
        },
        "WildGPT": {
            4: [18.452, 20.365, 20.365, 20.365, 20.365, 20.365, 20.365],
            8: [20.749, 21.780, 21.780, 21.780, 21.780, 21.780, 21.780],
            32: [22.699, 22.970, 22.970, 22.971, 22.971, 22.972, 22.972]
        }
    },
    "Qwen3-14B": {
        "ShareGPT": {
            4: [28.613, 28.617, 28.617, 28.617, 28.617, 28.617, 28.617],
            8: [30.986, 31.004, 31.004, 31.004, 31.004, 31.004, 31.004],
            32: [33.286, 33.313, 33.313, 33.313, 33.314, 33.314, 33.314]
        },
        "WildGPT": {
            4: [25.018, 28.879, 28.879, 28.879, 28.879, 28.879, 28.879],
            8: [29.839, 31.777, 31.777, 31.777, 31.777, 31.777, 31.777],
            32: [33.394, 33.961, 33.961, 33.961, 33.962, 33.962, 33.962]
        }
    }
}

x = np.arange(len(metrics))

for model in models:
    fig, axes = plt.subplots(len(req_rates), len(datasets), figsize=(14, 18), sharey=True)
    fig.suptitle(f"{model}", fontsize=32)

    for i, rate in enumerate(req_rates):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

            p_vals = parallax_data[model][dataset][rate]
            c_vals = addis_data[model][dataset][rate]

            line1, = ax.plot(x, p_vals, marker='o', label='Parallax', color='skyblue', linestyle='-', markeredgecolor='black', linewidth=3.5, markersize=10)
            line2, = ax.plot(x, c_vals, marker='s', label='Addis', color='orange', linestyle='-', markeredgecolor='black', linewidth=3.5, markersize=10)

            ax.set_xticks(x)
            ax.set_xticklabels([])

            if i == 0:
                ax.set_title(f"{dataset}", fontsize=28, pad=18)
            if j == len(datasets) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"request rate = {rate}", rotation=270, fontsize=30, labelpad=34)
            if j == 0:
                ax.set_ylabel("Latency (s)", fontsize=28)
            if i == len(req_rates) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, fontsize=28, rotation=45, ha='right')
                ax.set_xlabel("Tail Latency Metrics", fontsize=28)

            ax.tick_params(axis='y', labelsize=28)

    fig.legend([line1, line2], ["Parallax", "ADDIS"], loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, fontsize=28)

    plt.tight_layout(rect=[0.05, 0.02, 0.92, 0.92], h_pad=4.0, w_pad=2.0)

    filename = f"end2end_latency_{model.lower()}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved {filename}")