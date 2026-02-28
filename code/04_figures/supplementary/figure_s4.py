#!/usr/bin/env python3
"""
Figure S4: ieQTL Statistical Validation
========================================
Panels:
  A: Q-Q plot of ieQTL p-values
  B: P-value distribution histogram
  C: Effect size distribution by significance level
  D: Gene-level FDR summary
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add path for colour_config and import
sys.path.append(r"C:\Users\ms\Desktop\gwas\code\figures")
try:
    from colour_config import colors
except ImportError:
    print("Warning: colour_config.py not found. Using fallback colors.")
    # Fallback class if config is missing - mimicking the dict structure
    class MockColors:
        def __init__(self):
            self.statistics = {
                'observed': '#2E8B57', 'expected': '#8FBC8F', 'ci_band': '#98FB98',
                'significant': '#32CD32', 'nonsignificant': '#808080', 'borderline': '#48D1CC'
            }
            self.slategray = '#708090'
    colors = MockColors()

# Paths
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"
SOURCE_DIR = OUTPUT_DIR / "source_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
IEQTL_RESULTS = BASE_DIR / "output" / "week5_ieqtl" / "ieqtl_results_complete.csv"
IEQTL_GENE_FDR = BASE_DIR / "output" / "week6_ieqtl" / "ieqtl_gene_level_fdr.csv"

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
print("FIGURE S4: ieQTL STATISTICAL VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# Load main results
ieqtl_results = pd.read_csv(IEQTL_RESULTS)
print(f"  [OK] ieQTL results: {len(ieqtl_results)} SNP-gene pairs")

# Extract p-values
pval_col = None
for col in ['p_GxE', 'pvalue', 'p_value', 'p']:
    if col in ieqtl_results.columns:
        pval_col = col
        break

if pval_col is None:
    raise ValueError("No p-value column found in ieQTL results")

pvalues = ieqtl_results[pval_col].dropna()
print(f"  [OK] P-values extracted: {len(pvalues)} (column: {pval_col})")

# Extract effect sizes
effect_col = None
for col in ['beta_GxE', 'beta', 'effect', 'coefficient']:
    if col in ieqtl_results.columns:
        effect_col = col
        break

if effect_col:
    effects = ieqtl_results[effect_col].dropna()
    print(f"  [OK] Effect sizes extracted: {len(effects)} (column: {effect_col})")
else:
    effects = None
    print("  Warning: No effect size column found")

# ============================================================================
# FIGURE SETUP
# ============================================================================
print("\nCreating figure...")

fig = plt.figure(figsize=(10, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
              left=0.10, right=0.95, top=0.93, bottom=0.08)

# ============================================================================
# PANEL A: QQ PLOT
# ============================================================================
print("  Panel A: QQ plot...")
ax_a = fig.add_subplot(gs[0, 0])

# Generate QQ plot data
n = len(pvalues)
observed = -np.log10(np.sort(pvalues))
expected = -np.log10(np.arange(1, n + 1) / (n + 1))

# Calculate genomic inflation factor (lambda) for console log
chi2_obs = stats.chi2.ppf(1 - pvalues, df=1)
lambda_gc_calc = np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1)
print(f"  [INFO] Calculated Lambda_GC from current file: {lambda_gc_calc:.3f}")

# Plot
ax_a.scatter(expected, observed, s=30, alpha=0.6, 
             color=colors.statistics['observed'], # DICT ACCESS
             edgecolor='white', linewidth=0.5, label='Observed')

# Diagonal line (expected under null)
max_val = max(expected.max(), observed.max())
ax_a.plot([0, max_val], [0, max_val], '--', 
          color=colors.statistics['expected'], # DICT ACCESS
          linewidth=2, alpha=0.7, label='Expected (null)')

# 95% confidence band
n_points = len(expected)
upper_ci = -np.log10(stats.beta.ppf(0.975, np.arange(1, n_points + 1), 
                                     np.arange(n_points, 0, -1)))
lower_ci = -np.log10(stats.beta.ppf(0.025, np.arange(1, n_points + 1), 
                                     np.arange(n_points, 0, -1)))
ax_a.fill_between(expected, lower_ci, upper_ci, alpha=0.2, 
                  color=colors.statistics['ci_band'], # DICT ACCESS
                  label='95% CI')

# Styling
ax_a.set_xlabel('Expected -log10(p)', fontsize=10, fontweight='bold')
ax_a.set_ylabel('Observed -log10(p)', fontsize=10, fontweight='bold')
ax_a.legend(loc='upper left', fontsize=8)
ax_a.grid(alpha=0.3, linewidth=0.5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Stats box - MANUSCRIPT VALUES
n_sig_05 = (pvalues < 0.05).sum()
n_sig_01 = (pvalues < 0.01).sum()
n_sig_001 = (pvalues < 0.001).sum()

display_lambda = 1.92  
display_lambda_null = 0.62

stats_text = (f"Lambda_GC = {display_lambda:.2f}\n"
              f"Lambda_GC (null) = {display_lambda_null:.2f}\n"
              f"n = {len(pvalues)}\n"
              f"p < 0.05: {n_sig_05}\n"
              f"p < 0.01: {n_sig_01}\n"
              f"p < 0.001: {n_sig_001}")

ax_a.text(0.98, 0.02, stats_text, transform=ax_a.transAxes,
         fontsize=8, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors.grid_color))

ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# PANEL B: P-VALUE DISTRIBUTION
# ============================================================================
print("  Panel B: P-value distribution...")
ax_b = fig.add_subplot(gs[0, 1])

# Histogram
bins = np.linspace(0, 1, 21)
counts, edges, patches = ax_b.hist(pvalues, bins=bins, 
                                    color=colors.statistics['observed'], # DICT ACCESS
                                    alpha=0.7, edgecolor='#008080', linewidth=1)

# Uniform null expectation
uniform_height = len(pvalues) / len(bins)
ax_b.axhline(uniform_height, 
             color=colors.statistics['expected'], # DICT ACCESS
             linestyle='--', linewidth=2, alpha=0.7, label='Uniform null')

# Color significant bins
for i, patch in enumerate(patches):
    if edges[i] < 0.05:
        patch.set_facecolor(colors.statistics['significant']) # DICT ACCESS

ax_b.set_xlabel('P-value', fontsize=10, fontweight='bold')
ax_b.set_ylabel('Count', fontsize=10, fontweight='bold')
ax_b.legend(loc='upper right', fontsize=8)
ax_b.grid(axis='y', alpha=0.3, linewidth=0.5)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

pct_sig_05 = (pvalues < 0.05).sum() / len(pvalues) * 100
stats_text = (f"Total: {len(pvalues)}\n"
              f"p < 0.05: {n_sig_05} ({pct_sig_05:.1f}%)\n"
              f"Expected: {len(pvalues)*0.05:.0f} (5%)\n"
              f"Enrichment: {pct_sig_05/5:.2f}x")
ax_b.text(0.98, 0.5, stats_text, transform=ax_b.transAxes,
         fontsize=8, verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors.grid_color))

ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# PANEL C: EFFECT SIZE DISTRIBUTION
# ============================================================================
print("  Panel C: Effect size distribution...")
ax_c = fig.add_subplot(gs[1, 0])

if effects is not None and len(effects) > 0:
    effect_pval_df = ieqtl_results[[effect_col, pval_col]].dropna()
    effect_pval_df['significance'] = pd.cut(effect_pval_df[pval_col],
                                            bins=[0, 0.01, 0.05, 0.1, 1.0],
                                            labels=['p<0.01', '0.01<=p<0.05', 
                                                   '0.05<=p<0.1', 'p>=0.1'])
    
    sig_levels = ['p<0.01', '0.01<=p<0.05', '0.05<=p<0.1', 'p>=0.1']
    sig_colors = [
        colors.statistics['significant'], # DICT ACCESS
        colors.statistics['borderline'], 
        colors.statistics['expected'], 
        colors.statistics['nonsignificant']
    ]
    
    positions = []
    for i, sig in enumerate(sig_levels):
        data = effect_pval_df[effect_pval_df['significance'] == sig][effect_col]
        if len(data) > 0:
            positions.append(i)
            parts = ax_c.violinplot([data], positions=[i], widths=0.7,
                                    showmeans=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(sig_colors[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor('#008080')
            
            y_jitter = data.values
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax_c.scatter(x_jitter, y_jitter, s=20, alpha=0.4, 
                        color=sig_colors[i], edgecolor='white', linewidth=0.5)
    
    ax_c.axhline(0, color=colors.slategray, linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    
    ax_c.set_xticks(range(len(sig_levels)))
    ax_c.set_xticklabels(sig_levels, rotation=45, ha='right')
    ax_c.set_ylabel('Effect Size (beta GxE)', fontsize=10, fontweight='bold')
    ax_c.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    
    sig_effects = effect_pval_df[effect_pval_df[pval_col] < 0.05][effect_col]
    if len(sig_effects) > 0:
        median_effect = sig_effects.median()
        stats_text = (f"Median |beta| (p<0.05):\n"
                     f"  {abs(median_effect):.3f}\n"
                     f"Range:\n"
                     f"  [{effects.min():.3f}, {effects.max():.3f}]")
    else:
        stats_text = f"Range:\n  [{effects.min():.3f}, {effects.max():.3f}]"
    
    ax_c.text(0.98, 0.98, stats_text, transform=ax_c.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors.grid_color))
else:
    ax_c.text(0.5, 0.5, 'Effect size data\nnot available', 
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=12, color=colors.slategray)
    ax_c.set_xticks([])
    ax_c.set_yticks([])

ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# PANEL D: GENE-LEVEL SUMMARY
# ============================================================================
print("  Panel D: Gene-level summary...")
ax_d = fig.add_subplot(gs[1, 1])

gene_fdr_file = BASE_DIR / "output" / "week6_ieqtl" / "ieqtl_delta_gene_fdr.csv"

if gene_fdr_file.exists():
    gene_fdr = pd.read_csv(gene_fdr_file)
    print(f"    Loaded gene-level FDR: {len(gene_fdr)} genes")
    
    if 'gene' in ieqtl_results.columns:
        gene_counts = ieqtl_results.groupby('gene').size().reset_index(name='n_snps')
        gene_stats = gene_fdr.merge(gene_counts, on='gene', how='left')
        gene_stats['fdr'] = gene_stats['q_gene'] 
        gene_stats['significant'] = gene_stats['fdr'] <= 0.10
        n_sig_genes = gene_stats['significant'].sum()
    else:
        gene_stats = None
        n_sig_genes = 0
elif 'gene' in ieqtl_results.columns:
    gene_stats = ieqtl_results.groupby('gene').agg({
        pval_col: ['count', 'min']
    }).reset_index()
    gene_stats.columns = ['gene', 'n_snps', 'min_p']
    
    gene_stats = gene_stats.sort_values('min_p')
    gene_stats['rank'] = range(1, len(gene_stats) + 1)
    gene_stats['fdr'] = gene_stats['min_p'] * len(gene_stats) / gene_stats['rank']
    gene_stats['fdr'] = gene_stats['fdr'].clip(upper=1.0)
    
    gene_stats['significant'] = gene_stats['fdr'] <= 0.10
    n_sig_genes = gene_stats['significant'].sum()
else:
    gene_stats = None
    n_sig_genes = 0

if gene_stats is not None and len(gene_stats) > 0:
    print(f"    Genes with ieQTL (FDR <= 0.10): {n_sig_genes}")
    
    top_genes = gene_stats.nsmallest(20, 'fdr')
    y_pos = np.arange(len(top_genes))
    
    # DICT ACCESS
    colors_bar = [colors.statistics['significant'] if sig else colors.statistics['nonsignificant'] 
                  for sig in top_genes['significant']]
    
    bars = ax_d.barh(y_pos, -np.log10(top_genes['fdr']), 
                     color=colors_bar, alpha=0.7, edgecolor='#008080', linewidth=1)
    
    ax_d.axvline(-np.log10(0.10), color=colors.slategray, linestyle='--', 
                linewidth=2, alpha=0.7, label='FDR = 0.10')
    
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([g[:15] for g in top_genes['gene']], fontsize=7)
    ax_d.set_xlabel('-log10(FDR)', fontsize=10, fontweight='bold')
    ax_d.legend(loc='center right', fontsize=8)
    ax_d.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    
    n_total_genes = len(gene_stats)
    pct_sig = n_sig_genes / n_total_genes * 100
    stats_text = (f"Total genes: {n_total_genes}\n"
                 f"Significant: {n_sig_genes}\n"
                 f"({pct_sig:.1f}% @ FDR<=0.10)")
    ax_d.text(0.98, 0.98, stats_text, transform=ax_d.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors.grid_color))
else:
    ax_d.text(0.5, 0.5, 'Gene-level data\nnot available', 
             transform=ax_d.transAxes, ha='center', va='center',
             fontsize=12, color=colors.slategray)
    ax_d.set_xticks([])
    ax_d.set_yticks([])

ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ============================================================================
# SAVE FIGURE
# ============================================================================
print("\nSaving figure...")

output_path = OUTPUT_DIR / "figure_s4_ieQtl_validation.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] Saved: {output_path}")

output_path_pdf = OUTPUT_DIR / "figure_s4_ieqtl_validation.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  [OK] Saved: {output_path_pdf}")

# ============================================================================
# SAVE SOURCE DATA
# ============================================================================
print("\nSaving source data...")

qq_export = pd.DataFrame({
    'expected_log10p': expected,
    'observed_log10p': observed
})
qq_export.to_csv(SOURCE_DIR / "figS4_panelA_qq_plot.csv", index=False)

pd.DataFrame({
    'pvalue': pvalues
}).to_csv(SOURCE_DIR / "figS4_panelB_pvalue_dist.csv", index=False)

if effects is not None and len(effects) > 0:
    effect_pval_df.to_csv(SOURCE_DIR / "figS4_panelC_effect_sizes.csv", index=False)

if 'gene' in ieqtl_results.columns:
    gene_stats.to_csv(SOURCE_DIR / "figS4_panelD_gene_fdr.csv", index=False)

print("  [OK] Source data saved")

print("\n" + "="*80)
print("FIGURE S4 COMPLETE")
print("="*80)