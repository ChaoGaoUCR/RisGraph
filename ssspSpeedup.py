import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# SSSP Data
data = {
    "graph": ["dl"] * 4 + ["or"] * 4 + ["sx"] * 4 + ["ttw"] * 4 + ["wen"] * 4 + ["wiki"] * 4,
    "batch_num": [8, 16, 32, 64] * 6,
    "CommonGraph Speedup": [6.99, 3.71, 2.07, 1.03, 7.76, 4.39, 2.31, 0.96,
                            4.22, 2.60, 1.31, 0.95, 7.16, 4.38, 2.30, 1.11,
                            7.12, 4.27, 2.19, 1.17, 9.45, 6.35, 3.51, 2.29],
    "Glign Speedup": [2.30, 1.09, 0.52, 0.24, 2.50, 1.10, 0.52, 0.21,
                      8.86, 5.10, 2.22, 1.55, 9.49, 4.87, 2.17, 1.01,
                      13.62, 6.72, 3.16, 1.58, 155.25, 82.54, 42.86, 22.23],
    "EvoGIndex Concurrent Speedup": [4.33, 2.21, 1.45, 0.79, 2.78, 1.28, 0.74, 0.33,
                                     16.69, 11.87, 7.07, 5.45, 15.69, 9.05, 5.28, 2.86,
                                     28.90, 17.63, 10.78, 6.37, 144.51, 115.31, 80.04, 49.42],
    "EvoGIndex Prediction Speedup": [35.20, 15.85, 8.41, 3.84, 121.39, 58.69, 28.02, 11.31,
                                     15.72, 9.42, 4.26, 2.90, 138.32, 72.55, 34.50, 16.33,
                                     105.41, 55.50, 27.07, 13.76, 191.84, 114.87, 61.62, 31.92]
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

ax.set_xlabel("Snapshot Number (batch_num)")
ax.set_ylabel("Log-scaled Speedup")
ax.set_yscale("log")
ax.grid(True)

# Graph color legend
graph_legend = [
    Line2D([0], [0], color=palette[i], lw=4, label=graphs[i].upper())
    for i in range(len(graphs))
]
legend1 = ax.legend(handles=graph_legend, title="Graph", loc='upper left')

# Method style legend
method_legend = [
    Line2D([0], [0], color='black', linestyle=line_styles[i], marker=markers[i],
           markersize=8, label=methods[i].replace(" Speedup", ""))
    for i in range(len(methods))
]
legend2 = ax.legend(handles=method_legend, title="Method Style", loc='lower right')
ax.add_artist(legend1)

fig.tight_layout()
fig.savefig("sssp_speedup.pdf")
print("Saved to sssp_speedup.pdf")
