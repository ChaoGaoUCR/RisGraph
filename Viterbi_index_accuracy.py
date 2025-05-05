
import matplotlib.pyplot as plt

data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4],
        "EvoGIndex": [97.7, 96.54, 95.67, 95.59, 96.14],
        "dump": [95.26, 91.07, 86.42, 82.54, 79.47]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [88.59, 89.42, 93.55, 96.76, 98.35, 99.44],
        "dump": [70.21, 55.49, 48.43, 46.49, 47.12, 50.34]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [82.54, 84.99, 90.36, 94.8, 96.81, 98.9],
        "dump": [80.49, 68.61, 61.93, 57.52, 55.92, 55.21]
    },
    "WEN": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [95.35, 93.79, 93.89, 92.35, 95.96, 97.54],
        "dump": [84.25, 72.16, 64.05, 56.88, 52.04, 50.69]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [97.26, 95.97, 96.76, 97.63, 98.8, 99.4],
        "dump": [89.85, 83.63, 80.1, 78.91, 78.85, 79.61]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [89.55, 85.21, 85.97, 90.12, 92.86, 95.13],
        "dump": [77.68, 59.14, 42.88, 31.19, 24.11, 21.53]
    }
}

colors = {"SX":"#1f77b4","Wiki":"#ff7f0e","OR":"#2ca02c","WEN":"#d62728","DL":"#9467bd","TTW":"#8c564b"}

plt.figure(figsize=(12,8))
for name, d in data.items():
    plt.plot(d["x"], d["EvoGIndex"], marker='o', linestyle='-', color=colors[name], label=f'{name} - EvoGIndex')
    plt.plot(d["x"], d["dump"], marker='s', linestyle='--', color=colors[name], label=f'{name} - dump')
plt.title("Viterbi Accuracy vs Time Window")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("Viterbi_index_accuracy.pdf")
plt.show()
