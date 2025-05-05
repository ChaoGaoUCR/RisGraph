import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Viterbi Data
data = {
    "graph": ["dl"] * 4 + ["or"] * 4 + ["sx"] * 4 + ["ttw"] * 4 + ["wen"] * 4 + ["wiki"] * 4,
    "batch_num": [8, 16, 32, 64] * 6,
    "CommonGraph Speedup": [3.65, 2.11, 1.65, 0.96, 6.78, 3.89, 2.13, 0.94,
                            5.85, 3.47, 1.92, 0.99, 8.22, 4.97, 2.72, 1.42,
                            8.84, 5.26, 2.86, 1.40, 8.06, 5.35, 3.37, 2.10],
    "Glign Speedup": [0.28, 0.13, 0.07, 0.03, 1.11, 0.52, 0.24, 0.09,
                      12.80, 6.48, 2.92, 1.43, 9.38, 4.86, 2.33, 1.14,
                      20.21, 9.75, 4.70, 2.24, 105.88, 57.43, 29.11, 14.80],
    "EvoGIndex Concurrent Speedup": [0.48, 0.29, 0.17, 0.10, 1.09, 0.55, 0.27, 0.13,
                                     17.97, 12.28, 7.16, 4.26, 13.04, 8.84, 5.25, 2.89,
                                     35.53, 21.59, 12.27, 6.92, 91.37, 68.22, 49.40, 29.12],
    "EvoGIndex Prediction Speedup": [4.08, 2.12, 1.07, 0.54, 93.78, 47.94, 21.77, 8.71,
                                     18.15, 9.16, 4.27, 2.20, 133.85, 71.90, 36.63, 17.65,
                                     115.94, 59.54, 29.18, 14.01, 179.77, 101.91, 56.92, 28.84]
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
fig.savefig("viterbi_speedup.pdf")
print("Saved to viterbi_speedup.pdf")
