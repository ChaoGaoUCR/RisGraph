
import matplotlib.pyplot as plt

# Corrected SSR Data
data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.75, 98.27, 97.41, 96.78, 95.12, 88.91],
        "dump": [99.03, 97.94, 96.7, 95.27, 93.61, 91.38]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [89.41, 83.59, 79.58, 76.86, 72.59, 70.42],
        "dump": [94.71, 88.13, 80.94, 77.03, 73.37, 70.49]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.58, 97.51, 92.56, 92.19, 85.99, 80.66],
        "dump": [98.05, 95.91, 93.55, 89.83, 86.09, 82.33]
    },
    "WEN": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [95.34, 96.1, 95.01, 93.38, 89.02, 76.84],
        "dump": [92.36, 86.97, 81.99, 77.45, 68.02, 64.26]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [95.36, 92.94, 92.97, 93.53, 93.27, 88.62],
        "dump": [94.11, 90.08, 87.49, 85.51, 83.93, 83.44]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [93.31, 96.78, 88.38, 81.2, 71.79, 59.41],
        "dump": [98.87, 94.62, 87.42, 78.9, 74.36, 71.71]
    }
}

# 设置颜色
colors = {
    "SX": "#1f77b4",
    "Wiki": "#ff7f0e",
    "OR": "#2ca02c",
    "WEN": "#d62728",
    "DL": "#9467bd",
    "TTW": "#8c564b"
}

plt.figure(figsize=(12, 8))

# 绘图
for name, d in data.items():
    color = colors.get(name, None)
    plt.plot(d["x"], d["EvoGIndex"], marker='o', linestyle='-', color=color, label=f'{name} - EvoGIndex')
    plt.plot(d["x"], d["dump"], marker='s', linestyle='--', color=color, label=f'{name} - dump')

plt.title("SSR Accuracy vs Time Window (Corrected)")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("SSR_index_accuracy_corrected.pdf")
plt.show()
