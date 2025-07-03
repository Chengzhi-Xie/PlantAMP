import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# --- 1. Read and count sequence lengths for positive and negative samples ---
with open("PBPAMP.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

positive_lengths = []
negative_lengths = []
for i in range(0, len(lines), 2):
    if i + 1 < len(lines):
        label = lines[i].strip()
        seq   = lines[i+1].strip()
        if label == "1":
            positive_lengths.append(len(seq))
        elif label == "0":
            negative_lengths.append(len(seq))

# --- 2. Define bins and labels ---
max_len = max(positive_lengths + negative_lengths)
# Use max_len + 1 for the last bin upper bound to include max length
bins = [0, 30, 60, 90, 120, 150, max_len + 1]
bin_labels = [
    "1-30AA",
    "31-60AA",
    "61-90AA",
    "91-120AA",
    "121-150AA",
    ">150AA"
]
# Compute bin centers for x-ticks
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# --- 3. Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # , sharex=True

# Subplot A: Positive
ax = axes[0]
# Normalized histogram (density=True)
ax.hist(positive_lengths,
        bins=bins,
        density=True,
        color='salmon',
        edgecolor='black',
        alpha=0.7,
        label='Histogram')
# KDE curve
kde_pos = gaussian_kde(positive_lengths)
x_vals = np.linspace(0, max_len, 200)
ax.plot(x_vals,
        kde_pos(x_vals),
        color='salmon',
        linewidth=2,
        label='KDE')

# Style settings
ax.set_xticks(bin_centers)
ax.set_xticklabels(bin_labels,
                   fontsize=10,
                   fontweight='bold',
                   color='black')
ax.set_yticklabels(ax.get_yticks(),
                   fontsize=10,
                   fontweight='bold',
                   color='black')
ax.set_xlabel("Sequence Length",
              fontsize=12,
              fontweight='bold',
              color='black')
ax.set_ylabel("Density",
              fontsize=12,
              fontweight='bold',
              color='black')
ax.set_title("Sequence Length Distribution (Positive)",
             fontsize=14,
             fontweight='bold',
             color='black')
ax.legend(prop={'weight': 'bold'},
          edgecolor='black')

# Subplot B: Negative
ax = axes[1]
ax.hist(negative_lengths,
        bins=bins,
        density=True,
        color='lightblue',
        edgecolor='black',
        alpha=0.7,
        label='Histogram')
kde_neg = gaussian_kde(negative_lengths)
ax.plot(x_vals,
        kde_neg(x_vals),
        color='lightblue',
        linewidth=2,
        label='KDE')

# Same style settings
ax.set_xticks(bin_centers)
ax.set_xticklabels(bin_labels,
                   fontsize=10,
                   fontweight='bold',
                   color='black')
ax.set_yticklabels(ax.get_yticks(),
                   fontsize=10,
                   fontweight='bold',
                   color='black')
ax.set_xlabel("Sequence Length",
              fontsize=12,
              fontweight='bold',
              color='black')
ax.set_ylabel("Density",
              fontsize=12,
              fontweight='bold',
              color='black')
ax.set_title("Sequence Length Distribution (Negative)",
             fontsize=14,
             fontweight='bold',
             color='black')
ax.legend(prop={'weight': 'bold'},
          edgecolor='black')

plt.tight_layout()
plt.savefig("sequence_length_distribution1.png", dpi=600, bbox_inches="tight")
plt.savefig("sequence_length_distribution1.tiff", dpi=600, bbox_inches="tight")
plt.show()