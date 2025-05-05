
import matplotlib.pyplot as plt

data = {
    "SX": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.59, 97.65, 96.59, 95.02, 92.03, 11.95],
        "dump": [98.61, 97.04, 95.26, 93.21, 90.82, 88.05]
    },
    "Wiki": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [87.22, 10.91, 16.7, 22.71, 29.14, 35.91],
        "dump": [94.61, 89.09, 83.3, 77.29, 70.86, 64.09]
    },
    "OR": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [93.03, 89.36, 13.99, 19.79, 26.41, 34.09],
        "dump": [95.83, 91.19, 86.01, 80.21, 73.59, 65.91]
    },
    "WEN": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.88, 97.79, 96.48, 94.51, 15.12, 18.98],
        "dump": [97.33, 94.54, 91.57, 88.36, 84.88, 81.02]
    },
    "DL": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [75.05, 82.88, 88.89, 93.04, 94.66, 65.58],
        "dump": [36.57, 36.09, 35.62, 35.19, 34.8, 34.42]
    },
    "TTW": {
        "x": [0.08, 0.16, 0.24, 0.32, 0.4, 0.48],
        "EvoGIndex": [98.74, 97.46, 94.89, 88.02, 65.7, 15.78],
        "dump": [98.71, 97.14, 95.15, 92.6, 89.29, 84.22]
    }
}

colors = {"SX":"#1f77b4","Wiki":"#ff7f0e","OR":"#2ca02c","WEN":"#d62728","DL":"#9467bd","TTW":"#8c564b"}

plt.figure(figsize=(12,8))
for name, d in data.items():
    plt.plot(d["x"], d["EvoGIndex"], marker='o', linestyle='-', color=colors[name], label=f'{name} - EvoGIndex')
    plt.plot(d["x"], d["dump"], marker='s', linestyle='--', color=colors[name], label=f'{name} - dump')
plt.title("WCC Accuracy vs Time Window")
plt.xlabel("Time Window (Batch Ratio)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("WCC_index_accuracy.pdf")
plt.show()
