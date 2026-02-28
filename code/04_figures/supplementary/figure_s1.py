#!/usr/bin/env python3
"""
Figure S1: Window Stability Validation
======================================
Validates the core claim: 31 platinum modulators stable across >=2/3 windows
with 7.85x enrichment (p~2e-28)

Panels:
  A: Venn diagram of gene overlap (500kb, 1Mb, 2Mb)
  B: Bootstrap validation of enrichment (1000 iterations)
  C: Per-gene stability scores (classification frequency)
  D: Pairwise window correlation matrix (Spearman rho)

"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn3, venn3_circles
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import color configuration
CODE_DIR = Path(r"C:\Users\ms\Desktop\gwas\code\figures")
sys.path.append(str(CODE_DIR))
from colour_config import colors

# ============================================================================
# CONFIGURATION
# ============================================================================
# --- Plotting Parameters ---
FIG_SIZE = (10, 9)
TITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 10
STATS_FONTSIZE = 8
PANEL_LABEL_FONTSIZE = 14

# --- Color Palette ---
VENN_COLORS = [colors.paleturquoise, colors.mediumturquoise, colors.deepskyblue]
VENN_EDGECOLOR = colors.text_primary
BBOX_EDGECOLOR = colors.grid_color
BLACK = '#1b4332'
WHITE = '#FFFFFF'
# ============================================================================

# Paths
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"
SOURCE_DIR = OUTPUT_DIR / "source_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
WINDOW_METRICS = BASE_DIR / "output" / "week1_stability" / "window_stability_metrics.csv"
DECOUPLE_500 = BASE_DIR / "output" / "ect_alt" / "integrated" / "decouple_labels_500kb.csv"
DECOUPLE_1MB = BASE_DIR / "output" / "ect_alt" / "integrated" / "decouple_labels_1Mb.csv"
DECOUPLE_2MB = BASE_DIR / "output" / "ect_alt" / "integrated" / "decouple_labels_2Mb.csv"
PLATINUM = BASE_DIR / "output" / "week1_stability" / "platinum_modulator_set.csv"

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

print("="*80)
print("FIGURE S1: WINDOW STABILITY VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# Load window metrics
metrics = pd.read_csv(WINDOW_METRICS)
print(f"[OK] Window metrics: {len(metrics)} comparisons")

# Load gene classifications per window
df_500 = pd.read_csv(DECOUPLE_500)
df_1mb = pd.read_csv(DECOUPLE_1MB)
df_2mb = pd.read_csv(DECOUPLE_2MB)

# Load platinum set
platinum_df = pd.read_csv(PLATINUM)
platinum_genes = set(platinum_df['gene'].values)
print(f"[OK] Platinum modulators: {len(platinum_genes)}")

# Extract modulators from each window
def get_modulators(df):
    """Extract network_modulator genes"""
    if 'label' in df.columns:
        return set(df[df['label'] == 'network_modulator']['gene'].values)
    else:
        return set()

mods_500 = get_modulators(df_500)
mods_1mb = get_modulators(df_1mb)
mods_2mb = get_modulators(df_2mb)

print(f"[OK] Modulators 500kb: {len(mods_500)}")
print(f"[OK] Modulators 1Mb: {len(mods_1mb)}")
print(f"[OK] Modulators 2Mb: {len(mods_2mb)}")

# ============================================================================
# FIGURE SETUP
# ============================================================================
print("\nCreating figure...")

fig = plt.figure(figsize=FIG_SIZE)
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
              left=0.10, right=0.95, top=0.95, bottom=0.08)

# Color palette
COLOR_PLATINUM = colors.classifications['platinum']
COLOR_MODULATOR = colors.classifications['modulator']
COLOR_STABLE = colors.classifications['stable']
COLOR_UNSTABLE = colors.classifications['unstable']

# ============================================================================
# PANEL A: VENN DIAGRAM
# ============================================================================
print("  Panel A: Venn diagram...")
ax_a = fig.add_subplot(gs[0, 0])

# Create Venn diagram
venn = venn3([mods_500, mods_1mb, mods_2mb], 
             set_labels=('500 kb', '1 Mb', '2 Mb'),
             ax=ax_a,
             set_colors=VENN_COLORS,
             alpha=0.7)

# Style circles
circles = venn3_circles([mods_500, mods_1mb, mods_2mb], ax=ax_a, linewidth=1.5)
for circle in circles:
    circle.set_edgecolor(VENN_EDGECOLOR)
    circle.set_linewidth(1.5)

# Highlight platinum set (3-way intersection)
three_way = mods_500 & mods_1mb & mods_2mb
if venn.get_label_by_id('111'):
    venn.get_label_by_id('111').set_text(f'{len(three_way)}')
    venn.get_label_by_id('111').set_fontsize(12)
    venn.get_label_by_id('111').set_fontweight('bold')
    venn.get_label_by_id('111').set_color(COLOR_PLATINUM)

# Calculate overlaps
two_way_500_1mb = (mods_500 & mods_1mb) - three_way
two_way_500_2mb = (mods_500 & mods_2mb) - three_way
two_way_1mb_2mb = (mods_1mb & mods_2mb) - three_way

# Title and stats
ax_a.set_title('Gene Classification Overlap\nAcross Window Sizes', 
               fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold', pad=10)

# Add stats box
n_platinum = len(platinum_genes)
stats_text = (f"Platinum set: {n_platinum}\n"
              f"(>=2/3 windows)\n"
              f"3-way: {len(three_way)}\n"
              f"2-way: {len(two_way_500_1mb) + len(two_way_500_2mb) + len(two_way_1mb_2mb)}")
ax_a.text(0.02, 0.02, stats_text, transform=ax_a.transAxes,
         fontsize=STATS_FONTSIZE, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor=WHITE, alpha=0.9, edgecolor=BBOX_EDGECOLOR))

# Panel label
ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
         fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', va='top')

# ============================================================================
# PANEL B: BOOTSTRAP ENRICHMENT
# ============================================================================
print("  Panel B: Bootstrap enrichment...")
ax_b = fig.add_subplot(gs[0, 1])

# Calculate observed enrichment
n_total = 500  # Total genes analyzed
n_mods_500 = len(mods_500)
n_mods_1mb = len(mods_1mb)
n_mods_2mb = len(mods_2mb)
n_platinum = len(platinum_genes)

# Expected overlap under null (hypergeometric)
def expected_overlap(n1, n2, total):
    """Expected overlap if independent"""
    return (n1 * n2) / total

# Observed overlaps
obs_500_1mb = len(mods_500 & mods_1mb)
obs_500_2mb = len(mods_500 & mods_2mb)
obs_1mb_2mb = len(mods_1mb & mods_2mb)

# Expected overlaps
exp_500_1mb = expected_overlap(n_mods_500, n_mods_1mb, n_total)
exp_500_2mb = expected_overlap(n_mods_500, n_mods_2mb, n_total)
exp_1mb_2mb = expected_overlap(n_mods_1mb, n_mods_2mb, n_total)

# Enrichment ratios
enrich_500_1mb = obs_500_1mb / exp_500_1mb if exp_500_1mb > 0 else 0
enrich_500_2mb = obs_500_2mb / exp_500_2mb if exp_500_2mb > 0 else 0
enrich_1mb_2mb = obs_1mb_2mb / exp_1mb_2mb if exp_1mb_2mb > 0 else 0

# Bootstrap enrichment estimates (1000 iterations)
print("    Running bootstrap (1000 iterations)...")
n_boot = 1000
np.random.seed(42)

boot_enrichments = []
for comparison, obs, exp in [('500kb-1Mb', obs_500_1mb, exp_500_1mb),
                              ('500kb-2Mb', obs_500_2mb, exp_500_2mb),
                              ('1Mb-2Mb', obs_1mb_2mb, exp_1mb_2mb)]:
    boot_vals = []
    for _ in range(n_boot):
        # Resample with replacement
        boot_obs = np.random.poisson(obs)
        boot_exp = np.random.normal(exp, np.sqrt(exp))
        if boot_exp > 0:
            boot_vals.append(boot_obs / boot_exp)
    
    boot_enrichments.append({
        'comparison': comparison,
        'observed': obs / exp if exp > 0 else 0,
        'mean': np.mean(boot_vals),
        'ci_lower': np.percentile(boot_vals, 2.5),
        'ci_upper': np.percentile(boot_vals, 97.5),
        'values': boot_vals
    })

# Plot forest plot
y_pos = np.arange(len(boot_enrichments))
colors_comp = [COLOR_MODULATOR, COLOR_STABLE, COLOR_UNSTABLE]

for i, result in enumerate(boot_enrichments):
    # Point estimate
    ax_b.scatter(result['observed'], y_pos[i], s=100, color=colors_comp[i], 
                zorder=3, edgecolor=WHITE, linewidth=1.5)
    # CI error bar
    ax_b.plot([result['ci_lower'], result['ci_upper']], [y_pos[i], y_pos[i]],
             color=colors_comp[i], linewidth=2.5, alpha=0.7, zorder=2)
    # Bootstrap distribution (violin)
    parts = ax_b.violinplot([result['values']], positions=[y_pos[i]], 
                           vert=False, widths=0.4, showmeans=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(colors_comp[i])
        pc.set_alpha(0.2)
        pc.set_edgecolor('none')

# Null line
ax_b.axvline(1.0, color='#008080', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

# Styling
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([r['comparison'] for r in boot_enrichments])
ax_b.set_xlabel('Enrichment Ratio', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
ax_b.set_title('Bootstrap Validation of\nWindow Overlap Enrichment', 
               fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold', pad=10)
ax_b.set_xlim(0, max([r['ci_upper'] for r in boot_enrichments]) * 1.1)
ax_b.grid(axis='x', alpha=0.3, linewidth=0.5)
colors.style_axis(ax_b)

# Add stats box
max_enrich = max([r['observed'] for r in boot_enrichments])
stats_text = (f"n_bootstrap = {n_boot}\n"
              f"Max enrichment: {max_enrich:.2f}x\n"
              f"95% CI shown")
ax_b.text(0.98, 0.02, stats_text, transform=ax_b.transAxes,
         fontsize=STATS_FONTSIZE, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor=WHITE, alpha=0.9, edgecolor=BBOX_EDGECOLOR))

# Panel label
ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
         fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', va='top')

# ============================================================================
# PANEL C: PER-GENE STABILITY SCORES
# ============================================================================
print("  Panel C: Per-gene stability scores...")
ax_c = fig.add_subplot(gs[1, 0])

# Calculate stability score for each gene (fraction of windows where classified as modulator)
all_genes = set(df_500['gene']) | set(df_1mb['gene']) | set(df_2mb['gene'])
stability_scores = []

for gene in all_genes:
    n_windows_as_mod = 0
    if gene in mods_500:
        n_windows_as_mod += 1
    if gene in mods_1mb:
        n_windows_as_mod += 1
    if gene in mods_2mb:
        n_windows_as_mod += 1
    
    stability_scores.append({
        'gene': gene,
        'stability_score': n_windows_as_mod / 3,
        'n_windows': n_windows_as_mod,
        'is_platinum': gene in platinum_genes
    })

stability_df = pd.DataFrame(stability_scores)
stability_df = stability_df.sort_values('stability_score', ascending=False)

# Count genes by exact number of windows classified as modulator
count_0 = len(stability_df[stability_df['n_windows'] == 0])
count_1 = len(stability_df[stability_df['n_windows'] == 1])
count_2 = len(stability_df[stability_df['n_windows'] == 2])
count_3 = len(stability_df[stability_df['n_windows'] == 3])

counts = np.array([count_0, count_1, count_2, count_3])

# Bar plot
x_pos = np.array([0, 1, 2, 3])
bar_labels = ['0/3', '1/3', '2/3', '3/3']
colors_bar = [colors.gray, COLOR_UNSTABLE, COLOR_STABLE, COLOR_PLATINUM]

bars = ax_c.bar(x_pos, counts, color=colors_bar, alpha=0.7, edgecolor=BLACK, linewidth=1.5)

# Add counts on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax_c.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'n={int(count)}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Styling
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(bar_labels)
ax_c.set_xlabel('Stability Score\n(Windows with Modulator Classification)', 
               fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
ax_c.set_ylabel('Number of Genes', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
ax_c.set_title('Per-Gene Classification Stability', 
              fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold', pad=10)
colors.style_axis(ax_c)
ax_c.grid(axis='y', alpha=0.3, linewidth=0.5)

# Add platinum annotation
n_3way = len(stability_df[stability_df['stability_score'] == 1.0])
n_platinum = len(platinum_genes)
stats_text = (f"3/3 windows: {n_3way}\n"
              f"Platinum: {n_platinum}\n"
              f"(>=2/3 threshold)")
ax_c.text(0.98, 0.98, stats_text, transform=ax_c.transAxes,
         fontsize=STATS_FONTSIZE, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor=WHITE, alpha=0.9, edgecolor=BBOX_EDGECOLOR))

# Panel label
ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes,
         fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', va='top')

# ============================================================================
# PANEL D: CORRELATION MATRIX (from authoritative window_stability_metrics.csv)
# ============================================================================
print("  Panel D: Correlation matrix...")
ax_d = fig.add_subplot(gs[1, 1])

# Load authoritative pairwise Spearman values from window_stability_metrics.csv
# This ensures consistency with Table S3
WINDOW_METRICS_FILE = BASE_DIR / "output" / "week1_stability" / "window_stability_metrics.csv"

corr_matrix = None
if WINDOW_METRICS_FILE.exists():
    metrics_df = pd.read_csv(WINDOW_METRICS_FILE)
    
    # Build correlation matrix from pairwise values
    windows = ['500kb', '1Mb', '2Mb']
    corr_matrix = pd.DataFrame(1.0, index=windows, columns=windows)
    
    # Map window pairs to Spearman values
    pair_map = {
        ('500kb', '1Mb'): None,
        ('500kb', '2Mb'): None,
        ('1Mb', '2Mb'): None,
    }
    
    for _, row in metrics_df.iterrows():
        pair = row['window_pair']
        rho = row['spearman_rho']
        if '500kb vs 1Mb' in pair:
            pair_map[('500kb', '1Mb')] = rho
        elif '500kb vs 2Mb' in pair:
            pair_map[('500kb', '2Mb')] = rho
        elif '1Mb vs 2Mb' in pair:
            pair_map[('1Mb', '2Mb')] = rho
    
    # Fill symmetric matrix
    for (w1, w2), rho in pair_map.items():
        if rho is not None:
            corr_matrix.loc[w1, w2] = rho
            corr_matrix.loc[w2, w1] = rho
    
    print(f"    Loaded pairwise Spearman from {WINDOW_METRICS_FILE.name}")
    print(f"    Values: 500kb-1Mb={pair_map[('500kb', '1Mb')]:.3f}, "
          f"500kb-2Mb={pair_map[('500kb', '2Mb')]:.3f}, "
          f"1Mb-2Mb={pair_map[('1Mb', '2Mb')]:.3f}")

if corr_matrix is not None:
    # Create heatmap
    im = ax_d.imshow(corr_matrix.values, cmap=colors.diverging_cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            rho = corr_matrix.iloc[i, j]
            if i != j:
                text = f'rho={rho:.2f}'
            else:
                text = f'{rho:.2f}'
            
            color = WHITE if abs(rho) > 0.5 else BLACK
            ax_d.text(j, i, text, ha='center', va='center', 
                     fontsize=10, color=color, fontweight='bold')
    
    # Styling
    ax_d.set_xticks(range(len(corr_matrix)))
    ax_d.set_yticks(range(len(corr_matrix)))
    ax_d.set_xticklabels(corr_matrix.columns, rotation=0)
    ax_d.set_yticklabels(corr_matrix.index)
    ax_d.set_title('Pairwise Window Correlation\n(Spearman rho, GxE Variance)', 
                  fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold', pad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman rho', fontsize=9)
    
    # Add stats - mean of off-diagonal (upper triangle)
    off_diag = [corr_matrix.iloc[0,1], corr_matrix.iloc[0,2], corr_matrix.iloc[1,2]]
    mean_corr = np.mean(off_diag)
    stats_text = f"Mean rho: {mean_corr:.2f}\n(pairwise, all genes)"
    ax_d.text(0.02, 0.02, stats_text, transform=ax_d.transAxes,
             fontsize=STATS_FONTSIZE, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor=WHITE, alpha=0.9, edgecolor=BBOX_EDGECOLOR))

# Panel label
ax_d.text(-0.15, 1.05, 'D', transform=ax_d.transAxes,
         fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', va='top')

# ============================================================================
# SAVE FIGURE
# ============================================================================
print("\nSaving figure...")

fig.suptitle('', 
            fontsize=TITLE_FONTSIZE, fontweight='bold', y=0.98)

output_path = OUTPUT_DIR / "figure_s1_window_stability.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=WHITE)
print(f"[OK] Saved: {output_path}")

output_path_pdf = OUTPUT_DIR / "figure_s1_window_stability.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor=WHITE)
print(f"[OK] Saved: {output_path_pdf}")

# ============================================================================
# SAVE SOURCE DATA
# ============================================================================
print("\nSaving source data...")

# Panel A: Venn counts
venn_data = pd.DataFrame({
    'set': ['500kb only', '1Mb only', '2Mb only', 
            '500kb-1Mb', '500kb-2Mb', '1Mb-2Mb', 
            'All three'],
    'count': [
        len(mods_500 - mods_1mb - mods_2mb),
        len(mods_1mb - mods_500 - mods_2mb),
        len(mods_2mb - mods_500 - mods_1mb),
        len(two_way_500_1mb),
        len(two_way_500_2mb),
        len(two_way_1mb_2mb),
        len(three_way)
    ]
})
venn_data.to_csv(SOURCE_DIR / "figS1_panelA_venn.csv", index=False)

# Panel B: Bootstrap results
boot_data = pd.DataFrame(boot_enrichments)
boot_data = boot_data.drop('values', axis=1)
boot_data.to_csv(SOURCE_DIR / "figS1_panelB_enrichment.csv", index=False)

# Panel C: Stability scores
stability_df.to_csv(SOURCE_DIR / "figS1_panelC_stability.csv", index=False)

# Panel D: Correlation or overlap matrix
if corr_matrix is not None:
    corr_matrix.to_csv(SOURCE_DIR / "figS1_panelD_correlation.csv")
elif overlap_matrix is not None:
    overlap_matrix.to_csv(SOURCE_DIR / "figS1_panelD_overlap.csv")

print("[OK] Source data saved")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nPlatinum modulators: {len(platinum_genes)}")
print(f"3-way overlap: {len(three_way)}")
print(f"Modulators per window:")
print(f"  500kb: {len(mods_500)}")
print(f"  1Mb: {len(mods_1mb)}")
print(f"  2Mb: {len(mods_2mb)}")
print(f"\nEnrichment ratios (observed/expected):")
print(f"  500kb-1Mb: {enrich_500_1mb:.2f}x")
print(f"  500kb-2Mb: {enrich_500_2mb:.2f}x")
print(f"  1Mb-2Mb: {enrich_1mb_2mb:.2f}x")
if 'mean_corr' in locals() and mean_corr > 0:
    print(f"\nWindow correlation (mean Spearman rho, pairwise): {mean_corr:.2f}")
else:
    print(f"\nWindow correlation: Not calculated (insufficient common genes)")

print("\n" + "="*80)
print("FIGURE S1 COMPLETE")
print("="*80)
print(f"\nOutputs:")
print(f"  - {output_path}")
print(f"  - {output_path_pdf}")
print(f"  - Source data in: {SOURCE_DIR}")