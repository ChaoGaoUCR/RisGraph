
import matplotlib.pyplot as plt

# SSNP Data
data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [97.84, 96.73, 96.26, 95.15, 94.31, 93.23],
        "dump": [94.71, 91.08, 87.96, 84.9, 81.18, 78.16]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [93.22, 91.99, 91.19, 90.36, 87.28, 78.39],
        "dump": [83.51, 71.7, 61.35, 55.92, 52.48, 52.81]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.76, 97.19, 96.6, 92.86, 92.4, 87.68],
        "dump": [92.16, 84.31, 76.68, 69.61, 63.04, 59.5]
    },
    "Wen": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [96.94, 97.14, 96.39, 96.27, 93.99, 88.86],
        "dump": [87.08, 77.08, 69.67, 62.9, 54.8, 51.64]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [96.11, 95.63, 96.38, 96.18, 96.03, 92.94],
        "dump": [92.76, 87.46, 84.5, 82.14, 80.47, 80.06]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [93.42, 90.36, 89.33, 87.95, 82.39, 76.35],
        "dump": [83.48, 67.85, 51.21, 36.46, 31.1, 26.71]
    }
}

# 颜色分配
colors = {
    "SX": "#1f77b4",     # 蓝色
    "Wiki": "#ff7f0e",   # 橙色
    "OR": "#2ca02c",     # 绿色
    "Wen": "#d62728",    # 红色
    "DL": "#9467bd",     # 紫色
    "TTW": "#8c564b"     # 棕色
}

plt.figure(figsize=(12, 8))

# 绘图
for name, d in data.items():
    color = colors.get(name, None)
    plt.plot(d["x"], d["EvoGIndex"], marker='o', linestyle='-', color=color, label=f'{name} - EvoGIndex')
    plt.plot(d["x"], d["dump"], marker='s', linestyle='--', color=color, label=f'{name} - dump')

plt.title("SSNP Accuracy vs Time Window")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("ssnp_index_accuracy.pdf")
plt.show()
