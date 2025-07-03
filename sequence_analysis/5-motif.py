#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib import transforms

# -----------------------------
# Global configuration: English font
# -----------------------------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Color mapping for 20 amino acids
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

# -----------------------------
# Compute total amino acid counts
# -----------------------------
def compute_amino_acid_count(seq_list, alphabet):
    return {aa: sum(seq.count(aa) for seq in seq_list) for aa in alphabet}

# -----------------------------
# Read data and filter PAMP sequences
# -----------------------------
labels, seqs = read_sequences('PBPAMP.txt')
aip_seqs = [s for l, s in zip(labels, seqs) if l == 1]

# -----------------------------
# Compute counts of each amino acid in PAMP and select top five
# -----------------------------
pamp_counts = compute_amino_acid_count(aip_seqs, standard_alphabet)
top5_aas = sorted(pamp_counts, key=lambda aa: pamp_counts[aa], reverse=True)[:5]
top5_counts = {aa: pamp_counts[aa] for aa in top5_aas}

# -----------------------------
# Plot Top-5 PAMP motif-like logo
# -----------------------------
fig, ax = plt.subplots(figsize=(max(8, len(top5_aas) * 0.4), 4))
x_pos = 0
max_cnt = max(top5_counts.values()) if top5_counts else 1e-9
gap = 0.1

for aa in top5_aas:
    cnt = top5_counts[aa]
    height = cnt / max_cnt * 5  # normalize height
    tp = TextPath((0, 0), aa, size=1,
                  prop=matplotlib.font_manager.FontProperties(family='Arial'))
    bb = tp.get_extents()
    sx = 1.0 / bb.width
    sy = height / bb.height
    path = transforms.Affine2D().scale(sx, sy).translate(x_pos, 0).transform_path(tp)
    ax.add_patch(PathPatch(path, color=amino_acid_colors.get(aa, 'black'), lw=0))
    x_pos += 1 + gap

# Beautify axes
ax.set_xlim(-0.5, x_pos - gap + 0.5)
ax.set_ylim(0, 6)
ax.set_xticks(np.arange(0, len(top5_aas) * (1 + gap), (1 + gap)))
ax.set_xticklabels(top5_aas, rotation=90)
plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')
ax.set_ylabel('Normalized Count', fontsize=14, fontweight='bold')
ax.set_title('Top 5 PAMP Motif-like Amino Acid Logo', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save and display
plt.savefig('pamp_top5_logo.png', dpi=600)
plt.savefig('pamp_top5_logo.tiff', dpi=600)
plt.show()
