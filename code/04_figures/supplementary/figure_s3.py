#!/usr/bin/env python3
"""
Figure S3: TF Binding Proximity Validation
===========================================
Panels:
  A: SNP-TFBS distance distribution (split violin)
  B: Fold enrichment of proximal SNPs across distance thresholds
  C: Proximity rate (<1 kb) by chromosome
  D: Consequence type x TF proximity interaction matrix
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import ks_2samp, fisher_exact, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Import color configuration
try:
    from colour_config import colors
except ImportError:
    print("Could not import colour_config.py, please ensure it is in the Python path.")
    sys.exit(1)


# Paths
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"
SOURCE_DIR = OUTPUT_DIR / "source_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
INFLUENTIAL_TF = BASE_DIR / "output" / "week3_tf_binding" / "influential_tf_distances.csv"
BACKGROUND_TF = BASE_DIR / "output" / "week3_tf_binding" / "background_tf_distances.csv"
FOREGROUND_ANNOTATED = BASE_DIR / "output" / "week2_enrichment" / "foreground_annotated.csv"

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
print("FIGURE S3: TF BINDING PROXIMITY VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

inf_tf = pd.read_csv(INFLUENTIAL_TF)
bg_tf = pd.read_csv(BACKGROUND_TF)

# FIX #2: Normalize chromosome labels (handle chr/Chr/int formats)
def norm_chr(x):
    """Normalize chromosome to integer: chr1/Chr1/1 -> 1"""
    x = str(x).lower().replace("chr", "")
    return int(x)

inf_tf['chr'] = inf_tf['chr'].apply(norm_chr)
bg_tf['chr'] = bg_tf['chr'].apply(norm_chr)

# Better legend labels matching paper terminology
inf_tf['type'] = 'Foreground (modulator-linked)'
bg_tf['type'] = 'Background'

print(f"[OK] Influential: {len(inf_tf)} SNPs")
print(f"[OK] Background: {len(bg_tf)} SNPs")

# Quick assertions - catch data issues early
assert len(inf_tf) == 93 and len(bg_tf) == 930, \
    f"Counts differ from manuscript (93/930). Got: {len(inf_tf)}/{len(bg_tf)}"
print("[OK] Data counts match manuscript expectations")

all_snps_tf = pd.concat([inf_tf, bg_tf], ignore_index=True)

# FIX #1: Load consequence data from FOREGROUND annotations (not background!)
fg_annotated = pd.read_csv(FOREGROUND_ANNOTATED)
fg_consequence = fg_annotated[['snp', 'consequence']].drop_duplicates(subset=['snp'])
fg_consequence.columns = ['snp_id', 'consequence']  # Rename to match TF distance file

inf_tf_with_cons = inf_tf.merge(fg_consequence, on='snp_id', how='left')
inf_tf_with_cons['consequence'] = inf_tf_with_cons['consequence'].fillna('unknown')

n_not_unknown = (inf_tf_with_cons['consequence'] != 'unknown').sum()
print(f"[OK] Added consequence data: {n_not_unknown}/93 SNPs with annotations")

# ============================================================================
# FIGURE SETUP
# ============================================================================
print("\nCreating figure...")

fig = plt.figure(figsize=(10, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
              left=0.10, right=0.95, top=0.93, bottom=0.08)

# Use defined colors
COLOR_FG = colors.classifications['modulator']
COLOR_BG = colors.classifications['driver']

# ============================================================================
# PANEL A: SNP-TFBS DISTRIBUTION
# ============================================================================
print("  Panel A: Distance distribution...")
ax_a = fig.add_subplot(gs[0, 0])

fg_dist = inf_tf['nearest_tf_distance'].values / 1000
bg_dist = bg_tf['nearest_tf_distance'].values / 1000

med_fg_full = np.median(fg_dist)
med_bg_full = np.median(bg_dist)

fg_dist_plot = fg_dist[fg_dist <= 200]
bg_dist_plot = bg_dist[bg_dist <= 200]

parts_fg = ax_a.violinplot([fg_dist_plot], positions=[0], widths=0.7,
                           vert=True, showmeans=False, showextrema=False)
for pc in parts_fg['bodies']:
    pc.set_facecolor(COLOR_FG)
    pc.set_alpha(0.7)
    pc.set_edgecolor(colors.darkslategray)
    pc.set_linewidth(1.5)
    m = np.mean(pc.get_paths()[0].vertices[:, 0])
    vertices = pc.get_paths()[0].vertices
    vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, m)

parts_bg = ax_a.violinplot([bg_dist_plot], positions=[0], widths=0.7,
                           vert=True, showmeans=False, showextrema=False)
for pc in parts_bg['bodies']:
    pc.set_facecolor(COLOR_BG)
    pc.set_alpha(0.7)
    pc.set_edgecolor(colors.darkslategray)
    pc.set_linewidth(1.5)
    m = np.mean(pc.get_paths()[0].vertices[:, 0])
    vertices = pc.get_paths()[0].vertices
    vertices[:, 0] = np.clip(vertices[:, 0], m, np.inf)

ax_a.hlines(med_fg_full, -0.35, 0, colors=COLOR_FG, linewidth=3, 
            label=f'Foreground (med={med_fg_full:.0f}kb)')
ax_a.hlines(med_bg_full, 0, 0.35, colors=COLOR_BG, linewidth=3, 
            label=f'Background (med={med_bg_full:.0f}kb)')

ax_a.set_xlim(-0.5, 0.5)
ax_a.set_ylim(0, 200)
ax_a.set_xticks([])
ax_a.set_ylabel('Distance to Nearest TF Peak (kb)', fontsize=10, fontweight='bold')
ax_a.legend(loc='upper right', fontsize=8)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.spines['bottom'].set_visible(False)

# Calculate both KS and Mann-Whitney U tests
ks_stat, ks_p = ks_2samp(fg_dist, bg_dist)
mw_stat, mw_p = mannwhitneyu(fg_dist, bg_dist, alternative='less')

# Format p-values
ks_p_text = f"{ks_p:.2e}" if ks_p < 0.001 else f"{ks_p:.3f}"
mw_p_text = f"{mw_p:.2e}" if mw_p < 0.001 else f"{mw_p:.3f}"

# Format statistical test results for display
stats_text = (f"KS test:\n"
              f"  D = {ks_stat:.3f}\n"
              f"  p = {ks_p_text}\n\n"
              f"Mann-Whitney U:\n"
              f"  p = {mw_p_text}\n\n"
              f"n(FG) = {len(fg_dist)}\n"
              f"n(BG) = {len(bg_dist)}")
ax_a.text(0.02, 0.98, stats_text, transform=ax_a.transAxes,
         fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors.grid_color))

ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# PANEL B: FOLD ENRICHMENT OF PROXIMAL SNPS
# ============================================================================
print("  Panel B: Enrichment curve...")
ax_b = fig.add_subplot(gs[0, 1])

thresholds = [1, 2, 5, 10, 20, 50, 100, 200]
enrichments = []
ci_lowers = []
ci_uppers = []
p_values = []

for thresh in thresholds:
    fg_within = (fg_dist <= thresh).sum()
    bg_within = (bg_dist <= thresh).sum()
    fg_total = len(fg_dist)
    bg_total = len(bg_dist)
    
    fg_pct = fg_within / fg_total
    bg_pct = bg_within / bg_total
    fold = fg_pct / bg_pct if bg_pct > 0 else 0
    
    table = [[fg_within, fg_total - fg_within],
             [bg_within, bg_total - bg_within]]
    try:
        _, p_val = fisher_exact(table, alternative='greater')
        p_values.append(p_val)
    except:
        p_values.append(1.0)
    
    n_boot = 1000
    boot_folds = []
    np.random.seed(42)
    for _ in range(n_boot):
        fg_boot = np.random.choice(fg_dist, size=fg_total, replace=True)
        bg_boot = np.random.choice(bg_dist, size=bg_total, replace=True)
        
        fg_boot_within = (fg_boot <= thresh).sum()
        bg_boot_within = (bg_boot <= thresh).sum()
        
        fg_boot_pct = fg_boot_within / fg_total
        bg_boot_pct = bg_boot_within / bg_total
        
        if bg_boot_pct > 0:
            boot_folds.append(fg_boot_pct / bg_boot_pct)
    
    if boot_folds:
        ci_lower = np.percentile(boot_folds, 2.5)
        ci_upper = np.percentile(boot_folds, 97.5)
    else:
        ci_lower = fold
        ci_upper = fold
    
    enrichments.append(fold)
    ci_lowers.append(ci_lower)
    ci_uppers.append(ci_upper)

fg_1kb = (fg_dist <= 1).sum()
bg_1kb = (bg_dist <= 1).sum()
enrich_1kb = enrichments[0]

ax_b.plot(thresholds, enrichments, 'o-', color=COLOR_FG, linewidth=2.5, 
         markersize=8, label='Observed enrichment', zorder=3)
ax_b.fill_between(thresholds, ci_lowers, ci_uppers, 
                  color=COLOR_FG, alpha=0.2, label='95% CI (bootstrap)')
ax_b.axhline(y=1, color=colors.gray, linestyle='--', linewidth=1.5, 
            label='No enrichment', zorder=1)

ax_b.plot(1, enrich_1kb, 'o', color=colors.forestgreen, markersize=12, zorder=4,
         markeredgewidth=2, markeredgecolor='white')
ax_b.annotate(f'{enrich_1kb:.2f}x @ 1kb\n(p={p_values[0]:.2e})', 
             xy=(1, enrich_1kb), xytext=(2, enrich_1kb + 1),
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=COLOR_FG, linewidth=2),
             arrowprops=dict(arrowstyle='->', color=COLOR_FG, lw=2))

ax_b.set_xscale('log')
ax_b.set_xlim(0.8, 250)
ax_b.set_ylim(0, max(enrichments) * 1.3)
ax_b.set_xlabel('Distance Threshold (kb)', fontsize=10, fontweight='bold')
ax_b.set_ylabel('Fold Enrichment', fontsize=10, fontweight='bold')
ax_b.legend(loc='upper right', fontsize=8)
ax_b.grid(True, alpha=0.3, linestyle=':')

# Add footnote explaining cumulative nature
footnote_text = ("Note: Cumulative enrichment (<=threshold). Point estimate\n"
                 "at 1kb differs from main text due to different denominators.")
ax_b.text(0.02, 0.02, footnote_text, transform=ax_b.transAxes,
         fontsize=6, style='italic', verticalalignment='bottom',
         color=colors.slategray)

ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# PANEL C: PROXIMITY RATE BY CHROMOSOME
# ============================================================================
print("  Panel C: Proximity rate by chromosome...")
ax_c = fig.add_subplot(gs[1, 0])

# Get actual chromosomes present in data
chromosomes = sorted(set(inf_tf['chr'].unique()) | set(bg_tf['chr'].unique()))

# Calculate proximity rate (<1kb) per chromosome
chr_prox = []
for chrom in chromosomes:
    fg_chr = inf_tf[inf_tf['chr'] == chrom]
    bg_chr = bg_tf[bg_tf['chr'] == chrom]
    
    fg_total = len(fg_chr)
    bg_total = len(bg_chr)
    
    fg_proximal = (fg_chr['nearest_tf_distance'] < 1000).sum()
    bg_proximal = (bg_chr['nearest_tf_distance'] < 1000).sum()
    
    fg_rate = (fg_proximal / fg_total * 100) if fg_total > 0 else 0
    bg_rate = (bg_proximal / bg_total * 100) if bg_total > 0 else 0
    
    chr_prox.append({
        'chr': chrom,
        'inf_rate': fg_rate,
        'bg_rate': bg_rate,
        'inf_n': fg_total,
        'bg_n': bg_total,
        'inf_prox': fg_proximal,
        'bg_prox': bg_proximal
    })

chr_df = pd.DataFrame(chr_prox)

# Create grouped bar plot
x = np.arange(len(chromosomes))
width = 0.35

bars1 = ax_c.bar(x - width/2, chr_df['inf_rate'], width, 
                 label='Foreground (modulator-linked)', color=COLOR_FG, alpha=0.7, edgecolor=colors.spine_color, linewidth=1.5)
bars2 = ax_c.bar(x + width/2, chr_df['bg_rate'], width, 
                 label='Background', color=COLOR_BG, alpha=0.7, edgecolor=colors.spine_color, linewidth=1.5)

# Add genome-wide average line
genome_inf_rate = (inf_tf['nearest_tf_distance'] < 1000).sum() / len(inf_tf) * 100
genome_bg_rate = (bg_tf['nearest_tf_distance'] < 1000).sum() / len(bg_tf) * 100

ax_c.axhline(y=genome_inf_rate, color=COLOR_FG, linestyle='--', linewidth=2, 
             alpha=0.5, label=f'Genome avg (FG: {genome_inf_rate:.1f}%)')
ax_c.axhline(y=genome_bg_rate, color=COLOR_BG, linestyle='--', linewidth=2, 
             alpha=0.5, label=f'Genome avg (BG: {genome_bg_rate:.1f}%)')

# Add enrichment fold as text above bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    
    # Calculate enrichment for this chromosome
    if height2 > 0:
        enrich = height1 / height2
        max_height = max(height1, height2)
        ax_c.text(i, max_height + 1, f'{enrich:.1f}x',
                 ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color=colors.forestgreen if enrich > 2 else colors.darkslategray)

# Styling
ax_c.set_xlabel('Chromosome', fontsize=10, fontweight='bold')
ax_c.set_ylabel('Proximal Rate (<1kb) (%)', fontsize=10, fontweight='bold')
ax_c.set_xticks(x)
ax_c.set_xticklabels([f'Chr{c}' for c in chromosomes], rotation=0)
ax_c.legend(loc='upper left', fontsize=7, framealpha=0.9)
ax_c.grid(True, axis='y', alpha=0.3, linestyle=':')
ax_c.set_ylim(0, max(chr_df[['inf_rate', 'bg_rate']].max().max() * 1.4, 20))

# Panel label
ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# PANEL D: PROXIMITY BY SNP CONSEQUENCE
# ============================================================================
print("  Panel D: Consequence interaction...")
ax_d = fig.add_subplot(gs[1, 1])

consequence_types = ['5\'UTR', '3\'UTR', 'Upstream', 'Exon', 'Intergenic', 'Other']
prox_bins = ['<1kb', '1-10kb', '10-100kb', '>100kb']

interaction_matrix = np.zeros((len(consequence_types), len(prox_bins)))

def categorize_cons(cons):
    if pd.isna(cons): return 'Other'
    cons = str(cons).lower()
    if '5' in cons and 'utr' in cons: return '5\'UTR'
    if '3' in cons and 'utr' in cons: return '3\'UTR'
    if 'upstream' in cons or 'promoter' in cons: return 'Upstream'
    if 'exon' in cons or 'coding' in cons: return 'Exon'
    if 'intergenic' in cons: return 'Intergenic'
    return 'Other'

fg_with_cons = inf_tf_with_cons.copy()
fg_with_cons['cons_cat'] = fg_with_cons['consequence'].apply(categorize_cons)
fg_with_cons['dist_kb'] = fg_with_cons['nearest_tf_distance'] / 1000

for i, cons in enumerate(consequence_types):
    cons_snps = fg_with_cons[fg_with_cons['cons_cat'] == cons]
    
    if len(cons_snps) > 0:
        interaction_matrix[i, 0] = (cons_snps['dist_kb'] < 1).sum()
        interaction_matrix[i, 1] = ((cons_snps['dist_kb'] >= 1) & (cons_snps['dist_kb'] < 10)).sum()
        interaction_matrix[i, 2] = ((cons_snps['dist_kb'] >= 10) & (cons_snps['dist_kb'] < 100)).sum()
        interaction_matrix[i, 3] = (cons_snps['dist_kb'] >= 100).sum()

row_sums = interaction_matrix.sum(axis=1, keepdims=True)
interaction_pct = np.divide(interaction_matrix, row_sums, 
                           where=row_sums!=0, out=np.zeros_like(interaction_matrix))

im = ax_d.imshow(interaction_pct, cmap=colors.sequential_cmap_green, aspect='auto', vmin=0, vmax=1)

for i in range(len(consequence_types)):
    for j in range(len(prox_bins)):
        val = interaction_pct[i, j]
        count = int(interaction_matrix[i, j])
        if count > 0:
            text = f'{val*100:.0f}%\n(n={count})'
            color = colors.mintcream if val > 0.5 else colors.darkslategray
            ax_d.text(j, i, text, ha='center', va='center',
                     color=color, fontsize=7, fontweight='bold')

ax_d.set_xticks(range(len(prox_bins)))
ax_d.set_xticklabels(prox_bins)
ax_d.set_yticks(range(len(consequence_types)))
ax_d.set_yticklabels(consequence_types)
ax_d.set_xlabel('TF Proximity', fontsize=10, fontweight='bold')
ax_d.set_ylabel('Consequence Type', fontsize=10, fontweight='bold')

cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
cbar.set_label('Proportion', fontsize=9)

ax_d.text(-0.15, 1.05, 'D', transform=ax_d.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# SAVE FIGURE
# ============================================================================
print("\nSaving figure...")

output_path = OUTPUT_DIR / "figure_s3_tf_proximity_FINAL.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_path}")

output_path_pdf = OUTPUT_DIR / "figure_s3_tf_proximity_FINAL.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_path_pdf}")

# ============================================================================
# SAVE SOURCE DATA
# ============================================================================
print("\nSaving source data...")

all_snps_tf.to_csv(SOURCE_DIR / "figS3_tf_distances.csv", index=False)

enrichment_curve = pd.DataFrame({
    'threshold_kb': thresholds,
    'fold_enrichment': enrichments,
    'ci_lower': ci_lowers,
    'ci_upper': ci_uppers,
    'p_value': p_values
})
enrichment_curve.to_csv(SOURCE_DIR / "figS3_enrichment_curve.csv", index=False)

chr_df.to_csv(SOURCE_DIR / "figS3_chromosome_distribution.csv", index=False)

cons_matrix = pd.DataFrame(interaction_matrix, 
                          index=consequence_types,
                          columns=prox_bins)
cons_matrix.to_csv(SOURCE_DIR / "figS3_consequence_proximity.csv")

# ADDED: Save both statistical test results
stats_df = pd.DataFrame([{
    'test': 'Kolmogorov-Smirnov',
    'statistic': ks_stat,
    'p_value': ks_p,
    'fg_median_kb': med_fg_full,
    'bg_median_kb': med_bg_full
}, {
    'test': 'Mann-Whitney',
    'statistic': mw_stat,
    'p_value': mw_p,
    'fg_median_kb': med_fg_full,
    'bg_median_kb': med_bg_full
}])
stats_df.to_csv(SOURCE_DIR / "figS3_statistical_tests.csv", index=False)

print("[OK] Source data saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTF Proximity Analysis:")
print(f"  Foreground (modulator-linked) SNPs: {len(fg_dist)}")
print(f"  Background SNPs: {len(bg_dist)}")
print(f"\nMedian distances:")
print(f"  Foreground: {med_fg_full:.1f} kb")
print(f"  Background: {med_bg_full:.1f} kb")
print(f"  Ratio: {med_bg_full/med_fg_full:.2f}x")
print(f"\nProximity (<1kb):")
print(f"  Foreground: {fg_1kb} ({fg_1kb/len(fg_dist)*100:.1f}%)")
print(f"  Background: {bg_1kb} ({bg_1kb/len(bg_dist)*100:.1f}%)")
print(f"  Enrichment: {enrich_1kb:.2f}x")
print(f"  Fisher p-value: {p_values[0]:.2e}")
print(f"\nStatistical Tests:")
print(f"  Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_p:.2e}")
print(f"  Mann-Whitney U: U={mw_stat:,.0f}, p={mw_p:.2e}")
print(f"\nProximity Rate by Chromosome (<1kb):")
print(chr_df[['chr', 'inf_rate', 'bg_rate', 'inf_prox', 'bg_prox']].to_string(index=False))

print("\n" + "="*80)
print("FIGURE S3 SCRIPT COMPLETE")
print("="*80)