import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# SSNP Data
data = {
    "graph": ["dl"] * 4 + ["or"] * 4 + ["sx"] * 4 + ["ttw"] * 4 + ["wen"] * 4 + ["wiki"] * 4,
    "batch_num": [8, 16, 32, 64] * 6,
    "CommonGraph Speedup": [8.455, 4.672, 2.968, 1.344, 11.202, 7.296, 3.056, 1.413,
                            5.230, 3.318, 1.649, 0.991, 10.002, 6.231, 3.303, 1.639,
                            10.469, 5.839, 2.927, 1.312, 11.657, 7.352, 4.827, 2.919],
    "Glign Speedup": [2.923, 1.359, 0.709, 0.294, 3.212, 1.564, 0.540, 0.213,
                      10.143, 5.309, 2.279, 1.294, 12.931, 6.314, 2.817, 1.307,
                      21.260, 10.468, 4.411, 1.914, 208.307, 106.054, 48.561, 27.739],
    "EvoGIndex Concurrent Speedup": [5.653, 3.373, 2.273, 1.098, 3.647, 2.000, 0.854, 0.418,
                                     16.606, 11.419, 6.227, 4.192, 15.338, 9.385, 5.414, 2.880,
                                     38.999, 24.134, 13.513, 6.351, 170.706, 91.530, 94.344, 57.451],
    "EvoGIndex Prediction Speedup": [27.032, 13.139, 7.616, 3.084, 120.092, 63.140, 23.502, 10.014,
                                     14.953, 8.057, 3.648, 2.114, 128.593, 64.385, 31.161, 14.512,
                                     109.920, 54.616, 24.954, 10.749, 167.458, 88.517, 50.594, 24.871]
}

df = pd.DataFrame(data)
graphs = df["graph"].unique()
methods = [
    "CommonGraph Speedup",
    "Glign Speedup",
    "EvoGIndex Concurrent Speedup",
    "EvoGIndex Prediction Speedup"
]
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "*"]

palette = sns.color_palette("tab10", n_colors=len(graphs))
fig, ax = plt.subplots(figsize=(14, 8))

# Plot lines
for i, graph in enumerate(graphs):
    sub_df = df[df["graph"] == graph]
    for j, method in enumerate(methods):
        ax.plot(
            sub_df["batch_num"],
            sub_df[method],
            label=f"{graph.upper()}" if j == 0 else None,
            linestyle=line_styles[j],
            marker=markers[j],
            markersize=7,
            linewidth=2.2,
            color=palette[i],
            alpha=0.9
        )

# Axes and scaling
ax.set_xlabel("Snapshot Number (batch_num)")
ax.set_ylabel("Log-scaled Speedup")
ax.set_yscale("log")
ax.grid(True)

# Legend for graph color
graph_legend = [
    Line2D([0], [0], color=palette[i], lw=4, label=graphs[i].upper())
    for i in range(len(graphs))
]
legend1 = ax.legend(handles=graph_legend, title="Graph", loc='upper left')

# Legend for method style
method_legend = [
    Line2D([0], [0], color='black', linestyle=line_styles[i], marker=markers[i],
           markersize=8, label=methods[i].replace(" Speedup", ""))
    for i in range(len(methods))
]
legend2 = ax.legend(handles=method_legend, title="Method Style", loc='lower right')
ax.add_artist(legend1)

fig.tight_layout()
fig.savefig("ssnp_speedup.pdf")
print("Saved to ssnp_speedup.pdf")
