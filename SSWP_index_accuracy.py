
import matplotlib.pyplot as plt

data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [97.84, 96.72, 96.21, 95.11, 94.31, 93.09],
        "dump": [94.71, 91.08, 87.96, 84.9, 81.18, 78.16]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [93.31, 91.82, 91.22, 90.33, 87.17, 78.4],
        "dump": [83.51, 71.7, 61.35, 55.92, 52.48, 52.81]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.76, 97.19, 96.58, 92.85, 92.4, 87.69],
        "dump": [92.16, 84.31, 76.68, 69.61, 63.04, 59.5]
    },
    "WEN": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [97.01, 97.06, 96.72, 95.99, 93.76, 88.38],
        "dump": [87.08, 77.08, 69.67, 62.9, 54.8, 51.64]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [97.34, 95.73, 96.45, 96.17, 96.09, 92.93],
        "dump": [92.76, 87.46, 84.5, 82.14, 80.47, 80.06]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [93.48, 90.34, 89.26, 87.94, 82.8, 76.24],
        "dump": [83.48, 67.85, 51.21, 36.46, 31.1, 26.71]
    }
}

colors = {"SX":"#1f77b4","Wiki":"#ff7f0e","OR":"#2ca02c","WEN":"#d62728","DL":"#9467bd","TTW":"#8c564b"}

plt.figure(figsize=(12,8))
for name, d in data.items():
    plt.plot(d["x"], d["EvoGIndex"], marker='o', linestyle='-', color=colors[name], label=f'{name} - EvoGIndex')
    plt.plot(d["x"], d["dump"], marker='s', linestyle='--', color=colors[name], label=f'{name} - dump')
plt.title("SSWP Accuracy vs Time Window")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("SSWP_index_accuracy.pdf")
plt.show()
