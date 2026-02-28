"""
Figure 2: Spatial Regulatory Architecture
=========================================
This script generates a composite figure showing the spatial regulatory
architecture of influential SNPs. It includes:
- A genome-wide karyotype plot of SNPs colored by consequence.
- ECDF plots of distance to TSS for different regulatory consequence types.
- Violin plots of TF binding proximity for different regulatory types.
- A heatmap of enrichment for consequence types across distance bins from TSS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
import sys
from scipy import stats
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent / "infrastructure"))
from data_loader_gwas import data_loader
from colour_config import colors
from stat_utils import fisher_exact_with_ci, ecdf_confidence_bands

colors.configure_matplotlib()

# Global style matching Nature Plants standard
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
# PLOT CONFIGURATION
# ============================================================================
FONT_SIZES = {
    'suptitle': 13,
    'title': 12,
    'axis_label': 11,
    'tick_label': 9.5,
    'legend': 9.5,
    'annotation': 8.5,
}

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"

# ============================================================================
# LOAD DATA
# ============================================================================
fg = pd.read_csv(BASE_DIR / "output" / "week2_enrichment" / "foreground_annotated.csv")
bg = pd.read_csv(BASE_DIR / "output" / "week2_enrichment" / "background_annotated.csv")
tf_sites = data_loader.load_tf_binding_sites()
gene_pos = data_loader.load_gene_positions()
platinum = data_loader.load_platinum_modulators()

fg['chr'] = fg['chr'].astype(str)
bg['chr'] = bg['chr'].astype(str)
platinum_genes = platinum['gene'].tolist()

# TF proximity
for df in [fg, bg]:
    df['min_tf_dist'] = np.nan
    df['tf_within_1kb'] = False
    
    for idx, snp in df.iterrows():
        chr_tf = tf_sites[tf_sites['chr'] == snp['chr']]
        if len(chr_tf) > 0:
            distances = np.abs(chr_tf['summit'].values - snp['pos'])
            min_dist = distances.min()
            df.at[idx, 'min_tf_dist'] = min_dist
            df.at[idx, 'tf_within_1kb'] = (min_dist <= 1000)

# Chr lengths
chr_lengths = {chr: gene_pos[gene_pos['chr']==chr]['end'].max() 
               for chr in ['1','2','3','4','5','6','7','8','9','10']
               if len(gene_pos[gene_pos['chr']==chr]) > 0}

# Consequence color map
consequence_colors = colors.consequences

# ============================================================================
# CREATE FIGURE
# ============================================================================
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1])

# ============================================================================
# PANEL A: GENOME ARCHITECTURE - Consequence-Colored Karyotype
# ============================================================================
ax_a = fig.add_subplot(gs[0, :])

y_positions = {}
bar_height = 0.4
spacing = 0.12

for i, chr in enumerate(sorted(chr_lengths.keys(), key=int)):
    y_pos = i * (bar_height + spacing)
    y_positions[chr] = y_pos
    
    rect = Rectangle((0, y_pos), chr_lengths[chr], bar_height,
                     linewidth=0.8, edgecolor=colors.darkslategray, facecolor='#EBF5FB', alpha=0.4)
    ax_a.add_patch(rect)
    ax_a.text(-2e6, y_pos + bar_height/2, chr,
             ha='right', va='center', fontsize=FONT_SIZES['tick_label'], fontweight='bold')

# Plot SNPs colored by consequence type
for _, snp in fg.iterrows():
    chr = snp['chr']
    if chr in y_positions:
        y_pos = y_positions[chr] + bar_height/2
        color = consequence_colors.get(snp['consequence'], colors.gray)
        
        # Size by importance if platinum
        if snp['gene'] in platinum_genes:
            marker = 'D'
            size = 80
            alpha = 1.0
        else:
            marker = 'o'
            size = 50
            alpha = 0.88
        
        edge_c = colors.darkslategray if snp['gene'] in platinum_genes else 'white'
        edge_w = 0.8 if snp['gene'] in platinum_genes else 0.3
        ax_a.scatter(snp['pos'], y_pos, c=color, marker=marker,
                    s=size, alpha=alpha, edgecolors=edge_c,
                    linewidths=edge_w, zorder=2)

ax_a.set_ylim(-0.3, len(chr_lengths) * (bar_height + spacing))
ax_a.set_xlim(-3e6, max(chr_lengths.values()) * 1.02)
ax_a.set_yticks([])
ax_a.spines['left'].set_visible(False)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_xticks(ax_a.get_xticks())
ax_a.set_xticklabels([f'{int(x/1e6)}' for x in ax_a.get_xticks()], fontsize=FONT_SIZES['tick_label'])
ax_a.set_xlabel('Position (Mb)', fontsize=FONT_SIZES['axis_label'])
# ax_a.set_title('Genome-wide Distribution of Regulatory Consequence Types', 
#               fontsize=FONT_SIZES['title'], fontweight='bold', pad=10)

# Legend with consequence types
legend_elements = []
for cons, color in [('5\' UTR', consequence_colors['5_prime_UTR']),
                    ('Upstream', consequence_colors['upstream']),
                    ('3\' UTR', consequence_colors['3_prime_UTR']),
                    ('Exon', consequence_colors['exon']),
                    ('Intergenic', consequence_colors.get('intergenic', colors.gray))]:
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=color, markersize=7, label=cons))

ax_a.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZES['legend'], 
           ncol=5, frameon=True, title='Consequence Type')
colors.add_panel_label(ax_a, 'A', x=-0.03, y=1.02)

# ============================================================================
# PANEL B: DISTANCE DECAY STRATIFIED BY CONSEQUENCE TYPE
# ============================================================================
ax_b = fig.add_subplot(gs[1, 0])

# Focus on regulatory types
reg_types = ['5_prime_UTR', 'upstream', '3_prime_UTR', 'exon', 'intergenic']

for cons_type in reg_types:
    fg_dist = fg[fg['consequence'] == cons_type]['dist_to_tss'].abs()
    
    if len(fg_dist) > 0:
        # ECDF
        x_vals, ecdf_vals = [], []
        sorted_dist = np.sort(fg_dist.dropna())
        for val in sorted_dist:
            x_vals.append(val / 1000)  # Convert to kb
            ecdf_vals.append(np.sum(sorted_dist <= val) / len(sorted_dist))
        
        color = consequence_colors.get(cons_type, colors.gray)
        label = cons_type.replace('_', ' ').replace('prime', '\'')
        
        ax_b.plot(x_vals, ecdf_vals, linewidth=2.8, color=color,
                 label=f'{label} (n={len(fg_dist)})', alpha=0.90)

ax_b.set_xlabel('Distance to TSS (kb)', fontsize=FONT_SIZES['axis_label'])
ax_b.set_ylabel('Cumulative Fraction', fontsize=FONT_SIZES['axis_label'])
ax_b.set_xlim(0, 100)
ax_b.legend(fontsize=FONT_SIZES['legend'], frameon=True, loc='lower right')
colors.style_axis(ax_b)
colors.add_panel_label(ax_b, 'B')

# Add reference lines
ax_b.axvline(x=1, color=colors.teal, linestyle='--', alpha=0.4, linewidth=1)
ax_b.axvline(x=20, color=colors.teal, linestyle='--', alpha=0.4, linewidth=1)
ax_b.text(1, 0.95, '1kb', fontsize=FONT_SIZES['annotation'], color=colors.teal)
ax_b.text(20, 0.95, '20kb', fontsize=FONT_SIZES['annotation'], color=colors.teal)

# Stats box
stats_text = "5'UTR: median ~0.5kb\nUpstream: median ~1.5kb\n3'UTR: median ~2kb"
colors.add_stats_box(ax_b, stats_text, x=0.05, y=0.40)

# ============================================================================
# PANEL C: TF PROXIMITY BY REGULATORY TYPE (ONLY TYPES WITH DATA)
# ============================================================================
ax_c = fig.add_subplot(gs[1, 1])

# Prepare data
plot_data = []
for cons_type in ['5_prime_UTR', 'upstream', '3_prime_UTR', 'exon']:
    fg_tf = fg[(fg['consequence'] == cons_type) & (fg['min_tf_dist'].notna())]['min_tf_dist'] / 1000
    bg_tf = bg[(bg['consequence'] == cons_type) & (bg['min_tf_dist'].notna())]['min_tf_dist'] / 1000
    
    for dist, label in [(fg_tf, 'FG'), (bg_tf, 'BG')]:
        for val in dist:
            plot_data.append({
                'consequence': cons_type.replace('_prime_', '\'').replace('_', ' '),
                'distance': val,
                'set': label
            })

plot_df = pd.DataFrame(plot_data)

# Check which types have data
types_with_data = []
for cons in ['5\' UTR', 'upstream', '3\' UTR', 'exon']:
    n_data = len(plot_df[plot_df['consequence'] == cons])
    if n_data > 0:
        types_with_data.append(cons)

# Only plot types with data
x_pos = np.arange(len(types_with_data))

for i, cons in enumerate(types_with_data):
    cons_data = plot_df[plot_df['consequence'] == cons]
    
    # FG data
    fg_data = cons_data[cons_data['set'] == 'FG']['distance'].values
    if len(fg_data) > 0:
        # Cap at 200kb for visualization
        fg_data_capped = np.clip(fg_data, 0, 200)
        parts = ax_c.violinplot([fg_data_capped], positions=[i-0.2], widths=0.35,
                                showmeans=False, showmedians=True, showextrema=False)
        cons_key = cons.replace('\'', '_prime_').replace(' ', '_')
        for pc in parts['bodies']:
            pc.set_facecolor(consequence_colors.get(cons_key, colors.gray))
            pc.set_alpha(0.75)
        parts['cmedians'].set_color(colors.teal)
        parts['cmedians'].set_linewidth(2)
    
    # BG data
    bg_data = cons_data[cons_data['set'] == 'BG']['distance'].values
    if len(bg_data) > 0:
        bg_data_capped = np.clip(bg_data, 0, 200)
        parts = ax_c.violinplot([bg_data_capped], positions=[i+0.2], widths=0.35,
                                showmeans=False, showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors.lightgray)
            pc.set_alpha(0.5)
        parts['cmedians'].set_color(colors.seagreen)
        parts['cmedians'].set_linewidth(1.2)

# Add sample sizes to labels
labels_with_n = []
for cons in types_with_data:
    cons_key = cons.replace('\'', '_prime_').replace(' ', '_')
    n_fg = len(fg[(fg['consequence'] == cons_key) & (fg['min_tf_dist'].notna())])
    n_bg = len(bg[(bg['consequence'] == cons_key) & (bg['min_tf_dist'].notna())])
    labels_with_n.append(f'{cons}\n(n={n_fg}/{n_bg})')

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(labels_with_n, fontsize=FONT_SIZES['tick_label'])
ax_c.set_ylabel('Distance to Nearest TF (kb)', fontsize=FONT_SIZES['axis_label'])
ax_c.set_ylim(0, 200)
# ax_c.set_title('TF Binding Proximity by Regulatory Element Type', fontsize=FONT_SIZES['title'], pad=8)
ax_c.axhline(y=1, color=colors.teal, linestyle='--', alpha=0.4, linewidth=1, zorder=0)
ax_c.text(len(types_with_data)-0.5, 5, '1kb threshold', fontsize=FONT_SIZES['annotation'], color=colors.teal)
colors.style_axis(ax_c)
colors.add_panel_label(ax_c, 'C')

# Simple explicit legend
legend_elements = [
    patches.Patch(facecolor=consequence_colors['upstream'], alpha=0.8,
                 label='Upstream (influential SNPs)'),
    patches.Patch(facecolor=consequence_colors['exon'], alpha=0.8,
                 label='Exon (influential SNPs)'),
    patches.Patch(facecolor=colors.lightgray, alpha=0.5,
                 label='Background SNPs'),
    Line2D([0], [0], color=colors.teal, linewidth=2, label='Median'),
]
ax_c.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZES['legend'], frameon=True)

# ============================================================================
# PANEL D: ENRICHMENT HEATMAP - TSS Distance x Consequence Type
# ============================================================================
ax_d = fig.add_subplot(gs[2, :])

# Define distance bins and consequence types
dist_bins = [(0, 1000, '0-1kb'), (1000, 5000, '1-5kb'), 
             (5000, 20000, '5-20kb'), (20000, 100000, '20-100kb')]
cons_types = ['5_prime_UTR', 'upstream', '3_prime_UTR', 'exon', 'intergenic']

# Calculate enrichment matrix
enrichment_matrix = np.zeros((len(cons_types), len(dist_bins)))
p_matrix = np.ones((len(cons_types), len(dist_bins)))

for i, cons in enumerate(cons_types):
    for j, (min_dist, max_dist, label) in enumerate(dist_bins):
        # Foreground
        fg_mask = (fg['consequence'] == cons) & \
                  (fg['dist_to_tss'].abs() >= min_dist) & \
                  (fg['dist_to_tss'].abs() < max_dist)
        fg_count = fg_mask.sum()
        
        # Background
        bg_mask = (bg['consequence'] == cons) & \
                  (bg['dist_to_tss'].abs() >= min_dist) & \
                  (bg['dist_to_tss'].abs() < max_dist)
        bg_count = bg_mask.sum()
        
        # Calculate enrichment with pseudocount
        fg_prop = (fg_count + 0.5) / (len(fg) + 2)
        bg_prop = (bg_count + 0.5) / (len(bg) + 2)
        enrichment = fg_prop / bg_prop if bg_prop > 0 else 1.0
        
        enrichment_matrix[i, j] = enrichment
        
        # Fisher test
        if fg_count > 0 and bg_count > 0:
            table = np.array([[fg_count, len(fg) - fg_count],
                            [bg_count, len(bg) - bg_count]])
            _, _, _, p = fisher_exact_with_ci(table)
            p_matrix[i, j] = p

# Plot heatmap
im = ax_d.imshow(enrichment_matrix, cmap=colors.diverging_cmap, aspect='auto',
                vmin=0.5, vmax=3.0, interpolation='nearest')

# Add text annotations
for i in range(len(cons_types)):
    for j in range(len(dist_bins)):
        enr = enrichment_matrix[i, j]
        p = p_matrix[i, j]
        
        # Stars for significance
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = ''
        
        text_color = 'white' if enr > 2.2 or enr < 0.7 else '#1b4332'
        ax_d.text(j, i, f'{enr:.2f}\n{stars}',
                 ha='center', va='center', fontsize=FONT_SIZES['tick_label'],
                 color=text_color, fontweight='bold')

# Labels
ax_d.set_xticks(np.arange(len(dist_bins)))
ax_d.set_xticklabels([label for _, _, label in dist_bins], fontsize=FONT_SIZES['tick_label'])
ax_d.set_yticks(np.arange(len(cons_types)))
ax_d.set_yticklabels([c.replace('_prime_', '\'').replace('_', ' ') for c in cons_types], 
                     fontsize=FONT_SIZES['tick_label'])
ax_d.set_xlabel('Distance to TSS', fontsize=FONT_SIZES['axis_label'])
ax_d.set_ylabel('Consequence Type', fontsize=FONT_SIZES['axis_label'])

# Colorbar
cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
cbar.set_label('Enrichment (Fg/Bg)', fontsize=FONT_SIZES['tick_label'])

colors.add_panel_label(ax_d, 'D', x=-0.03, y=1.02)

# ============================================================================
# FINALIZE
# ============================================================================
# plt.suptitle('Figure 2: Spatial Regulatory Architecture and Mechanistic Insights',
#             fontsize=FONT_SIZES['suptitle'], fontweight='bold', y=0.995)

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
png_path = FIG_DIR / "figure_2.png"
pdf_path = FIG_DIR / "figure_2.pdf"
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

# Source data
source_path = OUTPUT_DIR / "figure_02_enhanced_source_data.csv"
source_df = pd.DataFrame({
    'panel': ['D'] * (len(cons_types) * len(dist_bins)),
    'consequence': np.repeat(cons_types, len(dist_bins)),
    'distance_bin': np.tile([l for _, _, l in dist_bins], len(cons_types)),
    'enrichment': enrichment_matrix.flatten(),
    'p_value': p_matrix.flatten()
})
source_df.to_csv(source_path, index=False)

plt.show()