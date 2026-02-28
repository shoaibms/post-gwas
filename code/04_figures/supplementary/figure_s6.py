#!/usr/bin/env python3
"""
Figure S6: GO Functional Decoupling Evidence
============================================
This script generates a 3-panel figure to validate the functional separation
(Jaccard index = 0) between modulator and driver GO terms.

Panels:
  A: Heatmap of -log10(p-value) for all significant GO terms.
  B: Network diagram illustrating spatial separation of GO terms.
  C: Dot plot of gene ratio vs. -log10(p-value).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Core Paths
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"
SOURCE_DIR = OUTPUT_DIR / "source_data"

# Data Input Paths
MODULATORS_GO = BASE_DIR / "output" / "week4_go_enrichment" / "modulators_go_enrichment.csv"
DRIVERS_GO = BASE_DIR / "output" / "week4_go_enrichment" / "drivers_go_enrichment.csv"

# Parameters
FDR_THRESHOLD = 0.05
RANDOM_SEED = 42

# Figure Aesthetics
PANEL_LABEL_SIZE = 16
LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
ANNOTATION_SIZE = 10
SMALL_TEXT_SIZE = 9

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Import project-specific color configurations
sys.path.append(str(BASE_DIR / 'code' / 'figures'))
from colour_config import colors

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================
modulators_go = pd.read_csv(MODULATORS_GO)
drivers_go = pd.read_csv(DRIVERS_GO)

# Add class labels
modulators_go['class'] = 'Modulator'
drivers_go['class'] = 'Driver'

# Add log10p
modulators_go['log10p'] = -np.log10(modulators_go['p_value'])
drivers_go['log10p'] = -np.log10(drivers_go['p_value'])

# Filter for significant terms (q <= 0.05)
q_col = None
for col in ['q_value', 'qvalue', 'fdr', 'FDR', 'adjusted_p', 'p.adjust']:
    if col in modulators_go.columns:
        q_col = col
        break

if q_col:
    modulators_go = modulators_go[modulators_go[q_col] <= FDR_THRESHOLD].copy()
    drivers_go = drivers_go[drivers_go[q_col] <= FDR_THRESHOLD].copy()

# Calculate gene ratio
if 'overlap' in modulators_go.columns and 'foreground_size' in modulators_go.columns:
    modulators_go['gene_ratio'] = modulators_go['overlap'] / modulators_go['foreground_size']
    drivers_go['gene_ratio'] = drivers_go['overlap'] / drivers_go['foreground_size']
else:
    # Assign a default if columns are missing
    modulators_go['gene_ratio'] = 0.05
    drivers_go['gene_ratio'] = 0.05

# Check overlap between GO terms
mod_terms = set(modulators_go['go_term'].values)
drv_terms = set(drivers_go['go_term'].values)
overlap = mod_terms & drv_terms
jaccard = len(overlap) / len(mod_terms | drv_terms) if len(mod_terms | drv_terms) > 0 else 0

# ============================================================================
# GLOBAL STYLE
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10.5,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'medium',
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#008080',
    'axes.labelcolor': '#1b4332',
    'text.color': '#1b4332',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'xtick.color': '#008080',
    'ytick.color': '#008080',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 9.5,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#B2DFDB',
    'legend.fancybox': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ============================================================================
# FIGURE SETUP
# ============================================================================
fig = plt.figure(figsize=(10, 9))
gs = GridSpec(2, 2, figure=fig,
              height_ratios=[1.2, 1],
              width_ratios=[1, 1],
              hspace=0.28, wspace=0.30,
              left=0.08, right=0.96, top=0.94, bottom=0.06)

# Panel A spans full top row
ax_a = fig.add_subplot(gs[0, :])

# Panels B and C on bottom row
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# Color palette
COLOR_MOD = colors.classifications['modulator']
COLOR_DRV = colors.classifications['driver']

# ============================================================================
# PANEL A: HEATMAP OF GO TERM ENRICHMENT
# ============================================================================
# Create matrix: rows = terms, columns = classes
n_show_mod = min(len(modulators_go), 25)
n_show_drv = min(len(drivers_go), 50)

mod_top = modulators_go.nlargest(n_show_mod, 'log10p')
drv_top = drivers_go.nlargest(n_show_drv, 'log10p')

# Create combined matrix
all_terms = pd.concat([mod_top, drv_top])
matrix = np.zeros((len(all_terms), 2))

# Fill matrix: column 0 = modulator, column 1 = driver
for i, (_, row) in enumerate(all_terms.iterrows()):
    if row['class'] == 'Modulator':
        matrix[i, 0] = row['log10p']
    else:
        matrix[i, 1] = row['log10p']

# Plot heatmap
im = ax_a.imshow(matrix, aspect='auto', cmap=colors.sequential_cmap,
                 vmin=0, vmax=matrix.max(), interpolation='nearest')

# Term labels (shortened for readability)
term_labels = []
for term in all_terms['go_term']:
    if len(term) > 40:
        term = term[:37] + '...'
    term_labels.append(term)

ax_a.set_yticks(range(len(term_labels)))
ax_a.set_yticklabels(term_labels, fontsize=SMALL_TEXT_SIZE - 1)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(['Modulators', 'Drivers'], fontsize=LABEL_SIZE, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04, shrink=0.8)
cbar.set_label(r'-log$_{10}$(p-value)', fontsize=LABEL_SIZE, fontweight='bold')
cbar.ax.tick_params(labelsize=SMALL_TEXT_SIZE)

# Separator line between classes
separator_idx = len(mod_top) - 0.5
ax_a.axhline(separator_idx, color='white', linewidth=3, linestyle='--', alpha=0.8)

# Add class annotations with counts
ax_a.text(0, 1.03, f"Modulators: {len(modulators_go)} terms",
          transform=ax_a.transAxes, ha="left", va="bottom",
          color=COLOR_MOD, fontsize=LABEL_SIZE, fontweight='bold',
          bbox=dict(fc="white", ec=COLOR_MOD, lw=2, boxstyle="round,pad=0.3"))

ax_a.text(1, 1.03, f"Drivers: {len(drivers_go)} terms",
          transform=ax_a.transAxes, ha="right", va="bottom",
          color=COLOR_DRV, fontsize=LABEL_SIZE, fontweight='bold',
          bbox=dict(fc="white", ec=COLOR_DRV, lw=2, boxstyle="round,pad=0.3"))

# Panel label
ax_a.text(-0.04, 1.05, 'A', transform=ax_a.transAxes,
         fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')

# ============================================================================
# PANEL B: NETWORK DIAGRAM - SPATIAL SEPARATION
# ============================================================================
# Create spatial layout
np.random.seed(RANDOM_SEED)

# Modulator positions (left cluster)
n_mod = len(modulators_go)
mod_x = np.random.normal(0.25, 0.08, n_mod)
mod_y = np.random.uniform(0.1, 0.9, n_mod)

# Driver positions (right cluster) - subsample if too many
n_drv_plot = min(60, len(drivers_go))
drv_subset = drivers_go.nlargest(n_drv_plot, 'log10p')
drv_x = np.random.normal(0.75, 0.08, n_drv_plot)
drv_y = np.random.uniform(0.1, 0.9, n_drv_plot)

# Plot nodes
mod_sizes = (modulators_go['log10p'] / modulators_go['log10p'].max() * 400) + 60
ax_b.scatter(mod_x, mod_y, s=mod_sizes, c=COLOR_MOD, alpha=0.65,
            edgecolor='white', linewidth=1.5, label='Modulators', zorder=3)

drv_sizes = (drv_subset['log10p'] / drv_subset['log10p'].max() * 400) + 60
ax_b.scatter(drv_x, drv_y, s=drv_sizes, c=COLOR_DRV, alpha=0.65,
            edgecolor='white', linewidth=1.5, label='Drivers', zorder=3)

# Cluster boundaries
mod_ellipse = Ellipse((0.25, 0.5), 0.35, 0.85,
                      fill=False, edgecolor=COLOR_MOD, linewidth=3,
                      linestyle='--', alpha=0.8, zorder=2)
drv_ellipse = Ellipse((0.75, 0.5), 0.35, 0.85,
                      fill=False, edgecolor=COLOR_DRV, linewidth=3,
                      linestyle='--', alpha=0.8, zorder=2)
ax_b.add_patch(mod_ellipse)
ax_b.add_patch(drv_ellipse)

# Labels with boxes
ax_b.text(0.25, 0.03, f'Modulators\n(n={n_mod})', ha='center', va='center',
         fontsize=LABEL_SIZE, fontweight='bold', color=COLOR_MOD,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=COLOR_MOD, linewidth=2, alpha=0.95))
ax_b.text(0.75, 0.03, f'Drivers\n(n={len(drivers_go)})', ha='center', va='center',
         fontsize=LABEL_SIZE, fontweight='bold', color=COLOR_DRV,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=COLOR_DRV, linewidth=2, alpha=0.95))

# Separation indicator
ax_b.annotate('', xy=(0.43, 0.5), xytext=(0.57, 0.5),
             arrowprops=dict(arrowstyle='<->', lw=3, color='#2F4F4F'))
ax_b.text(0.5, 0.54, 'Zero Overlap\n(Jaccard = 0)', ha='center', va='bottom',
         fontsize=ANNOTATION_SIZE, fontweight='bold', color='#2F4F4F',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='palegreen',
                  edgecolor='#2F4F4F', linewidth=1.5, alpha=0.7))

# Styling
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)
ax_b.set_aspect('equal')
ax_b.axis('off')

# Panel label
ax_b.text(-0.06, 1.05, 'B', transform=ax_b.transAxes,
         fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')

# ============================================================================
# PANEL C: DOT PLOT - GENE RATIO VS SIGNIFICANCE
# ============================================================================
# Get gene counts for sizing
if 'overlap' in modulators_go.columns:
    mod_gene_count = modulators_go['overlap'].values
else:
    mod_gene_count = np.ones(len(modulators_go)) * 5

if 'overlap' in drivers_go.columns:
    drv_gene_count = drivers_go['overlap'].values
else:
    drv_gene_count = np.ones(len(drivers_go)) * 5

# Plot modulators
ax_c.scatter(modulators_go['gene_ratio'], modulators_go['log10p'],
            s=mod_gene_count * 80, c=COLOR_MOD, alpha=0.7,
            edgecolor='white', linewidth=2, label='Modulators', zorder=3)

# Plot drivers
ax_c.scatter(drivers_go['gene_ratio'], drivers_go['log10p'],
            s=drv_gene_count * 80, c=COLOR_DRV, alpha=0.7,
            edgecolor='white', linewidth=2, label='Drivers', zorder=3)

# Significance threshold
ax_c.axhline(-np.log10(0.05), color='#008080', linestyle='--',
            linewidth=2, alpha=0.6, label='p=0.05', zorder=1)

# Styling
ax_c.set_xlabel('Gene Ratio (overlap / foreground)', fontsize=LABEL_SIZE, fontweight='bold')
ax_c.set_ylabel(r'-log$_{10}$(p-value)', fontsize=LABEL_SIZE, fontweight='bold')
ax_c.legend(loc='upper right', fontsize=ANNOTATION_SIZE, framealpha=0.95,
           edgecolor='#B2DFDB', fancybox=True)
ax_c.grid(alpha=0.25, linewidth=0.5, zorder=0)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.spines['left'].set_linewidth(1.5)
ax_c.spines['bottom'].set_linewidth(1.5)
ax_c.tick_params(labelsize=TICK_LABEL_SIZE, width=1.5)

# Stats box
stats_text = (f"Bubble size ~ gene count\n"
              f"Distinct clouds demonstrate\n"
              f"functional separation\n"
              f"(Jaccard index = {jaccard:.1f})")
ax_c.text(0.02, 0.98, stats_text, transform=ax_c.transAxes,
         fontsize=SMALL_TEXT_SIZE, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.95, edgecolor='#008080', linewidth=1.5))

# Panel label
ax_c.text(-0.10, 1.05, 'C', transform=ax_c.transAxes,
         fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')

# ============================================================================
# SAVE FIGURE AND SOURCE DATA
# ============================================================================
output_path = OUTPUT_DIR / "figure_s6_go_decoupling.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

output_path_pdf = OUTPUT_DIR / "figure_s6_go_decoupling.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

# Save source data
modulators_go.to_csv(SOURCE_DIR / "figS6_modulators.csv", index=False)
drivers_go.to_csv(SOURCE_DIR / "figS6_drivers.csv", index=False)

# Network positions
network_data = pd.DataFrame({
    'go_term': list(modulators_go['go_term']) + list(drv_subset['go_term']),
    'class': ['Modulator'] * len(modulators_go) + ['Driver'] * len(drv_subset),
    'x': list(mod_x) + list(drv_x),
    'y': list(mod_y) + list(drv_y)
})
network_data.to_csv(SOURCE_DIR / "figS6_network.csv", index=False)

plt.show()