import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# BFS Data
data = {
    "graph": ["dl"] * 4 + ["or"] * 4 + ["sx"] * 4 + ["ttw"] * 4 + ["wen"] * 4 + ["wiki"] * 4,
    "batch_num": [8, 16, 32, 64] * 6,
    "CommonGraph Speedup": [20.01, 11.44, 6.06, 2.59, 30.96, 18.59, 9.07, 3.98,
                            9.99, 6.04, 2.85, 1.64, 19.18, 12.67, 6.46, 3.33,
                            23.83, 13.91, 7.61, 3.58, 18.01, 10.37, 6.39, 4.03],
    "Glign Speedup": [8.43, 3.84, 1.86, 0.71, 9.85, 4.93, 1.96, 0.89,
                      31.32, 15.87, 6.74, 3.4, 30.8, 14.68, 6.72, 3.21,
                      48.97, 24.36, 11.33, 5.09, 498.85, 233.33, 111.84, 58.63],
    "EvoGIndex Concurrent Speedup": [24.79, 12.63, 8.13, 3.74, 17.98, 8.86, 4.75, 2.13,
                                     66.04, 45.43, 26.71, 15.75, 47.93, 27.49, 16.24, 9.04,
                                     129.85, 73.64, 45.76, 23.82, 500.92, 377.49, 267.8, 168.66],
    "EvoGIndex Prediction Speedup": [71.68, 35.57, 16.97, 6.96, 262.92, 136.1, 58.18, 24.86,
                                     32.56, 17.74, 7.9, 4.06, 240.52, 126.33, 60.36, 29.02,
                                     254.72, 132.29, 64.75, 29.21, 247.72, 129.84, 69.38, 36.94]
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
fig.savefig("bfs_speedup.pdf")
print("Saved to bfs_speedup.pdf")
