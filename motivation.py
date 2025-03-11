import matplotlib.pyplot as plt
import numpy as np

# Updated data
num_snapshots_16 = np.array([1, 3, 5])
mutation_time_16 = np.array([429.334106, 430.29248, 426.116577])
computation_time_16 = np.array([133.69548, 182.001465, 328.733765])
total_time_16 = mutation_time_16 + computation_time_16

num_snapshots_32 = np.array([2, 4, 6])
mutation_time_32 = np.array([448.888, 451.85421, 455.0147])
computation_time_32 = np.array([114.16548, 150.3416, 279.703735])
total_time_32 = mutation_time_32 + computation_time_32

# Plot setup
fig, ax1 = plt.subplots(figsize=(8,6))

# Bar chart with breakdown for 16 snapshots (aligned)
bars1 = ax1.bar(num_snapshots_16, mutation_time_16, width=0.5, color='tab:blue', alpha=0.7)
bars2 = ax1.bar(num_snapshots_16, computation_time_16, width=0.5, bottom=mutation_time_16, color='tab:orange', alpha=0.7)

# Bar chart with breakdown for 32 snapshots (aligned)
bars3 = ax1.bar(num_snapshots_32, mutation_time_32, width=0.5, color='tab:purple', alpha=0.7)
bars4 = ax1.bar(num_snapshots_32, computation_time_32, width=0.5, bottom=mutation_time_32, color='tab:pink', alpha=0.7)

# Line plots
ax2 = ax1.twinx()
ax2.plot(num_snapshots_16, total_time_16, marker='o', linestyle='-', color='tab:red')
ax2.plot(num_snapshots_32, total_time_32, marker='s', linestyle='--', color='tab:green')

# Annotate line points
for i, txt in enumerate(total_time_16):
    ax2.annotate(f'{txt:.1f}', (num_snapshots_16[i], total_time_16[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=24, color='black')

for i, txt in enumerate(total_time_32):
    ax2.annotate(f'{txt:.1f}', (num_snapshots_32[i], total_time_32[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=24, color='black')

# Labels with font size 18
# ax1.set_xlabel("Number of Sources", fontsize=24)
ax1.set_ylabel("Time", fontsize=24)
# ax2.set_ylabel("Total Time", fontsize=24)

# Remove all legends
ax1.legend().remove()
ax2.legend().remove()

# Set custom x-axis labels at midpoints of (1-2), (3-4), (5-6)
ax1.set_xticks([1.5, 3.5, 5.5])
ax1.set_xticklabels(["4", "8", "16"], fontsize=24)

# Remove grid lines
ax1.grid(False)
ax2.grid(False)

# Remove the top border (spine)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Set tick label font size
ax1.tick_params(axis='both', labelsize=18)
ax2.tick_params(axis='both', labelsize=18)

# Show plot
# plt.show()
plt.savefig('motivation.pdf', bbox_inches='tight')
