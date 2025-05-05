import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Data
data = {
    "graph": ["dl"] * 4 + ["or"] * 4 + ["sx"] * 4 + ["ttw"] * 4 + ["wen"] * 4 + ["wiki"] * 4,
    "batch_num": [8, 16, 32, 64] * 6,
    "CommonGraph Speedup": [7.5, 4.52, 2.46, 1.17, 12.23, 7.41, 4.5, 1.8, 5.51, 3.23, 1.66, 0.97,
                            10.0, 6.29, 3.44, 1.69, 13.72, 7.81, 3.92, 2.0, 11.82, 7.83, 4.85, 3.1],
    "Glign Speedup": [3.51, 1.77, 0.84, 0.36, 2.89, 1.37, 0.67, 0.22, 11.8, 5.46, 2.46, 1.33,
                      13.29, 6.92, 3.12, 1.43, 24.33, 11.28, 4.92, 2.28, 214.89, 112.2, 57.47, 28.51],
    "EvoGIndex Concurrent Speedup": [5.68, 3.58, 2.34, 1.24, 3.78, 1.89, 1.04, 0.45, 16.55, 11.41, 6.61, 4.16,
                                     16.1, 10.04, 5.7, 3.03, 45.11, 25.91, 14.2, 7.77, 169.82, 136.16, 83.62, 59.8],
    "EvoGIndex Prediction Speedup": [25.2, 12.94, 6.42, 2.86, 90.9, 44.24, 22.24, 7.98, 17.08, 8.46, 3.98, 2.16,
                                     129.18, 64.78, 31.09, 14.6, 107.35, 52.37, 23.22, 10.9, 170.72, 96.17, 49.78, 26.66]
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

# Axis labels and log scale
ax.set_xlabel("Snapshot Number (batch_num)")
ax.set_ylabel("Log-scaled Speedup")
ax.set_yscale("log")
ax.grid(True)

# Legend 1: color → graph
graph_legend = [
    Line2D([0], [0], color=palette[i], lw=4, label=graphs[i].upper())
    for i in range(len(graphs))
]
legend1 = ax.legend(handles=graph_legend, title="Graph", loc='upper left')

# Legend 2: linestyle/marker → method
method_legend = [
    Line2D([0], [0], color='black', linestyle=line_styles[i], marker=markers[i],
           markersize=8, label=methods[i].replace(" Speedup", ""))
    for i in range(len(methods))
]
legend2 = ax.legend(handles=method_legend, title="Method Style", loc='lower right')

# Add first legend back manually (otherwise it gets overwritten)
ax.add_artist(legend1)

fig.tight_layout()
fig.savefig("sswp_speedup.pdf")
print("Saved to sswp_speedup.pdf")
