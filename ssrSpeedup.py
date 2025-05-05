import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# SSR Data
data = {
    "graph": ["dl"] * 4 + ["or"] * 4 + ["sx"] * 4 + ["ttw"] * 4 + ["wen"] * 4 + ["wiki"] * 4,
    "batch_num": [8, 16, 32, 64] * 6,
    "CommonGraph Speedup": [9.42, 5.67, 3.88, 1.48, 20.99, 13.58, 6.55, 2.97,
                            6.55, 4.48, 2.18, 1.37, 17.93, 12.15, 6.90, 3.63,
                            42.75, 10.78, 5.93, 2.54, 15.56, 11.09, 7.12, 4.59],
    "Glign Speedup": [3.21, 1.57, 0.92, 0.30, 6.86, 3.40, 1.34, 0.56,
                      35.05, 14.99, 7.06, 3.19, 15.62, 7.31, 3.26, 1.53,
                      62.01, 9.52, 4.05, 1.65, 50.68, 32.60, 14.78, 8.33],
    "EvoGIndex Concurrent Speedup": [8.94, 4.82, 3.16, 1.12, 8.65, 4.28, 1.62, 0.71,
                                     25.99, 16.07, 8.91, 4.76, 23.18, 13.24, 7.01, 3.72,
                                     133.63, 28.29, 14.48, 6.20, 51.34, 28.32, 15.99, 7.48],
    "EvoGIndex Prediction Speedup": [28.14, 15.11, 8.78, 2.90, 91.70, 43.96, 17.62, 7.40,
                                     15.60, 7.63, 3.43, 1.53, 153.45, 76.57, 35.72, 17.02,
                                     316.30, 62.48, 28.88, 11.69, 179.05, 97.20, 51.39, 25.25]
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

# Graph legend
graph_legend = [
    Line2D([0], [0], color=palette[i], lw=4, label=graphs[i].upper())
    for i in range(len(graphs))
]
legend1 = ax.legend(handles=graph_legend, title="Graph", loc='upper left')

# Method legend
method_legend = [
    Line2D([0], [0], color='black', linestyle=line_styles[i], marker=markers[i],
           markersize=8, label=methods[i].replace(" Speedup", ""))
    for i in range(len(methods))
]
legend2 = ax.legend(handles=method_legend, title="Method Style", loc='lower right')
ax.add_artist(legend1)

fig.tight_layout()
fig.savefig("ssr_speedup.pdf")
print("Saved to ssr_speedup.pdf")
