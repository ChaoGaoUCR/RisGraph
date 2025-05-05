
import matplotlib.pyplot as plt

# SSSP Data
data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [92.26, 90.84, 91.44, 93.14, 95.04, 96.52],
        "dump": [93.7, 88.74, 84.45, 81.29, 78.71, 76.71]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [82.39, 89.43, 93.9, 95.51, 93.88, 88.48],
        "dump": [66.82, 53.75, 48.58, 47.18, 47.84, 50.86]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [80.7, 90.49, 94.36, 96.01, 96.89, 93.0],
        "dump": [73.02, 62.66, 58.68, 56.93, 56.18, 56.03]
    },
    "WEN": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [80.68, 81.41, 85.44, 89.35, 93.17, 91.38],
        "dump": [79.96, 66.53, 58.46, 52.98, 49.87, 49.68]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4],
        "EvoGIndex": [92.08, 93.81, 95.88, 97.2, 97.68],
        "dump": [88.58, 82.31, 79.32, 78.43, 78.66]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [77.33, 76.82, 81.43, 85.6, 87.06, 87.39],
        "dump": [75.38, 56.81, 41.3, 31.29, 24.76, 22.34]
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

plt.title("SSSP Accuracy vs Time Window")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("SSSP_index_accuracy.pdf")
plt.show()
