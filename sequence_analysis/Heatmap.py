#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------
# Global configuration: English font
# -----------------------------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 20 amino acids and their colors (adjustable as needed)
# -----------------------------
amino_acid_colors = {
    'A':'#e6194b','C':'#3cb44b','D':'#ffe119','E':'#0082c8',
    'F':'#f58231','G':'#911eb4','H':'#46f0f0','I':'#d2f53c',
    'K':'#f032e6','L':'#fabebe','M':'#008080','N':'#e6beff',
    'P':'#aa6e28','Q':'#fffac8','R':'#800000','S':'#aaffc3',
    'T':'#808000','V':'#ffd8b1','W':'#808080','Y':'#000080'
}
standard_alphabet = list(amino_acid_colors.keys())

# -----------------------------
# Read labels and sequences
# -----------------------------
def read_sequences(file_path):
    labels, seqs = [], []
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        seqs.append(lines[i+1])
    return labels, seqs

labels, seqs = read_sequences('PBPAMP.txt')
aip_seqs     = [s for l, s in zip(labels, seqs) if l == 1]
non_aip_seqs = [s for l, s in zip(labels, seqs) if l == 0]

# -----------------------------
# Compute positional residue frequencies (column-normalized and trim empty columns)
# -----------------------------
def positional_freq_matrix(seq_list, alphabet):
    # maximum sequence length
    max_len = max(len(s) for s in seq_list)
    # count matrix and number of valid sequences per column
    raw_counts   = np.zeros((len(alphabet), max_len), dtype=int)
    counts_per_pos = np.zeros(max_len, dtype=int)

    for seq in seq_list:
        for i, aa in enumerate(seq):
            if aa in alphabet:
                j = alphabet.index(aa)
                raw_counts[j, i] += 1
                counts_per_pos[i] += 1

    # normalize by column
    freq = np.zeros_like(raw_counts, dtype=float)
    nonzero = counts_per_pos > 0
n    freq[:, nonzero] = raw_counts[:, nonzero] / counts_per_pos[nonzero]

    # find last non-zero column and trim
    if nonzero.any():
        eff_len = nonzero.nonzero()[0].max() + 1
    else:
        eff_len = 0

    return freq[:, :eff_len], eff_len

pamp_mat,   pamp_len   = positional_freq_matrix(aip_seqs,     standard_alphabet)
nonpamp_mat, nonpamp_len = positional_freq_matrix(non_aip_seqs, standard_alphabet)

# -----------------------------
# Use a unified color scale range for comparison
# -----------------------------
global_vmax = max(pamp_mat.max(), nonpamp_mat.max())

# -----------------------------
# Custom gradient color maps (or use plt.cm.Reds / plt.cm.Blues)
# -----------------------------
cmap_positive = mcolors.LinearSegmentedColormap.from_list('pos',['#FFFFFF','#D13D5D'])
cmap_negative = mcolors.LinearSegmentedColormap.from_list('neg',['#FFFFFF','#3A7FBF'])

# -----------------------------
# Plot positional residue frequency heatmap
# -----------------------------
def plot_position_heatmap(matrix, alphabet, title, cmap, vmax):
    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        matrix,
        aspect='auto',
        interpolation='none',
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        origin='upper'
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Amino Acid', fontsize=14, fontweight='bold')
    plt.xlabel('Sequence Position', fontsize=14, fontweight='bold')
    plt.yticks(range(len(alphabet)), alphabet, fontsize=12, fontweight='bold')
    # label every 10 positions
    plt.xticks(np.arange(0, matrix.shape[1], step=10), fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, label='Frequency')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold', color='black')
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_color('black')
    cbar.ax.yaxis.label.set_fontsize(14)
    plt.tight_layout()

# (1) PAMP heatmap
plot_position_heatmap(
    pamp_mat,
    standard_alphabet,
    'PAMP: Positional Amino Acid Frequency',
    cmap_positive,
    global_vmax
)
plt.savefig('pamp_freq.png', dpi=600, format='png')
plt.savefig('pamp_freq.tiff', dpi=600, format='tiff')
plt.show()

# (2) non-PAMP heatmap
plot_position_heatmap(
    nonpamp_mat,
    standard_alphabet,
    'non-PAMP: Positional Amino Acid Frequency',
    cmap_negative,
    global_vmax
)
plt.savefig('nonpamp_freq.png', dpi=600, format='png')
plt.savefig('nonpamp_freq.tiff', dpi=600, format='tiff')
plt.show()
