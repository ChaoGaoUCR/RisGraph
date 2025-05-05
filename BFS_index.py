
import matplotlib.pyplot as plt
import itertools

# 所有数据集
data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [87.81, 86.18, 88.77, 90.9, 92.54, 93.58],
        "dump": [93.7, 88.74, 84.45, 81.29, 78.71, 76.71]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [73.19, 82.5, 87.03, 86.34, 82.55, 78.17],
        "dump": [66.82, 53.75, 48.58, 47.18, 47.84, 50.86]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [80.04, 90.37, 94.14, 95.58, 95.77, 92.71],
        "dump": [73.02, 62.66, 58.68, 56.93, 56.18, 56.03]
    },
    "Wen": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [71.56, 74.95, 82.01, 87.12, 90.49, 89.43],
        "dump": [79.96, 66.53, 58.46, 52.98, 49.87, 49.68]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [88.76, 90.97, 93.23, 94.47, 94.88, 94.14],
        "dump": [88.58, 82.31, 79.32, 78.43, 78.66, 79.68]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [62.17, 60.58, 69.99, 77.4, 80.64, 80.11],
        "dump": [75.38, 56.81, 41.3, 31.29, 24.76, 22.34]
    }
}

# 设置不同颜色（确保每组唯一）
colors = {
    "SX": "#1f77b4",     # 蓝色
    "Wiki": "#ff7f0e",   # 橙色
    "OR": "#2ca02c",     # 绿色
    "Wen": "#d62728",    # 红色
    "DL": "#9467bd",     # 紫色
    "TTW": "#8c564b"     # 棕色
}

plt.figure(figsize=(12, 8))

# 为每个数据集分配颜色并画线
for name, d in data.items():
    color = colors.get(name, None)
    plt.plot(d["x"], d["EvoGIndex"], marker='o', linestyle='-', color=color, label=f'{name} - EvoGIndex')
    plt.plot(d["x"], d["dump"], marker='s', linestyle='--', color=color, label=f'{name} - dump')

plt.title("Accuracy vs Time Window (Each Dataset Uses a Unique Color)")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("BFS_index.pdf", dpi=300, bbox_inches='tight')
plt.show()
