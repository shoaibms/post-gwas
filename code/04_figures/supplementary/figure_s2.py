"""
Figure S2: Regulatory Consequence Enrichment
=============================================
Panels:
  A: Enrichment by consequence type (dot plot)
  B: Consequence-stratified distance ECDFs
  C: Chi-square standardized residuals heatmap
  D: Distance-stratified regulatory enrichment
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2_contingency, fisher_exact, ks_2samp, mannwhitneyu
from colour_config import colors

# ============================================================================
# SCRIPT-SPECIFIC COLOR ADAPTATION
# ============================================================================

class ScriptColors:
    """A new color configuration object for this script."""
    pass

script_colors = ScriptColors()

# Map colors from the central colour_config to this script's specific needs
script_colors.regulatory = colors.classifications["driver"]
script_colors.nonregulatory = colors.gray
script_colors.diverging_cmap = colors.diverging_cmap
script_colors.category_colors = {
    "5'UTR": colors.consequences["5_prime_UTR"],
    "3'UTR": colors.consequences["3_prime_UTR"],
    'Upstream': colors.consequences["upstream"],
    'Downstream': colors.consequences["downstream"],
    'Exon': colors.consequences["exon"],
    'Intron': colors.consequences["intron"],
    'Intergenic': colors.consequences["intergenic"],
    'Other': colors.gainsboro
}

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"
SOURCE_DIR = OUTPUT_DIR / "source_data"
DATA_DIR = BASE_DIR / "output" / "week2_enrichment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

FOREGROUND_FILE = DATA_DIR / "foreground_annotated.csv"
BACKGROUND_FILE = DATA_DIR / "background_annotated.csv"

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

print("=" * 80)
print("FIGURE S2: REGULATORY CONSEQUENCE ENRICHMENT")
print("=" * 80)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def categorize_consequence(cons_str):
    """Categorize consequence into standardized categories."""
    if pd.isna(cons_str):
        return 'Other'
    
    cons = str(cons_str).lower()
    
    if any(c in cons for c in ['5_prime_utr', 'utr_5', "5'utr"]):
        return "5'UTR"
    if any(c in cons for c in ['3_prime_utr', 'utr_3', "3'utr"]):
        return "3'UTR"
    if 'upstream' in cons or 'promoter' in cons:
        return 'Upstream'
    if 'downstream' in cons:
        return 'Downstream'
    if any(c in cons for c in ['exon', 'missense', 'synonymous', 'coding']):
        return 'Exon'
    if 'intron' in cons:
        return 'Intron'
    if 'intergenic' in cons:
        return 'Intergenic'
    
    return 'Other'


def load_and_process_data():
    """Load and preprocess annotated SNP data."""
    print("\nLoading data...")
    
    fg_df = pd.read_csv(FOREGROUND_FILE)
    bg_df = pd.read_csv(BACKGROUND_FILE)
    
    # Extract and rename columns
    fg_cols = {}
    bg_cols = {}
    
    # Flexible column detection
    for col in fg_df.columns:
        if 'snp' in col.lower() and 'id' not in col.lower():
            fg_cols[col] = 'snp'
        elif 'snp_id' in col.lower() or col == 'snp':
            fg_cols[col] = 'snp'
    
    for col in fg_df.columns:
        if 'dist' in col.lower() and 'tss' in col.lower() and 'fg' not in col.lower():
            fg_cols[col] = 'dist_to_tss'
            break
    
    if 'consequence' in fg_df.columns:
        fg_cols['consequence'] = 'consequence'
    
    # Similar for background
    for col in bg_df.columns:
        if col == 'snp' or ('snp' in col.lower() and 'bg' not in col.lower() and 'fg' not in col.lower()):
            bg_cols[col] = 'snp'
            break
    
    for col in bg_df.columns:
        if 'dist' in col.lower() and 'tss' in col.lower() and 'fg' not in col.lower():
            bg_cols[col] = 'dist_to_tss'
            break
    
    if 'consequence' in bg_df.columns:
        bg_cols['consequence'] = 'consequence'
    
    fg_snps = fg_df.rename(columns=fg_cols)[['snp', 'dist_to_tss', 'consequence']].copy()
    bg_snps = bg_df.rename(columns=bg_cols)[['snp', 'dist_to_tss', 'consequence']].copy()
    
    # Ensure absolute distances
    fg_snps['dist_to_tss'] = fg_snps['dist_to_tss'].abs()
    bg_snps['dist_to_tss'] = bg_snps['dist_to_tss'].abs()
    
    print(f"  Foreground SNPs: {len(fg_snps)}")
    print(f"  Background SNPs: {len(bg_snps)}")
    
    # Categorize consequences
    fg_snps['category'] = fg_snps['consequence'].apply(categorize_consequence)
    bg_snps['category'] = bg_snps['consequence'].apply(categorize_consequence)
    
    # Define regulatory categories
    regulatory_categories = ["5'UTR", "3'UTR", "Upstream", "Downstream"]
    fg_snps['is_regulatory'] = fg_snps['category'].isin(regulatory_categories)
    bg_snps['is_regulatory'] = bg_snps['category'].isin(regulatory_categories)
    
    print(f"\nForeground regulatory: {fg_snps['is_regulatory'].sum()} ({100*fg_snps['is_regulatory'].mean():.1f}%)")
    print(f"Background regulatory: {bg_snps['is_regulatory'].sum()} ({100*bg_snps['is_regulatory'].mean():.1f}%)")
    
    return fg_snps, bg_snps, regulatory_categories


# ============================================================================
# PANEL PLOTTING FUNCTIONS
# ============================================================================

def plot_panel_a(ax, fg_df, bg_df):
    """Panel A: Enrichment by consequence type (dot plot)."""
    fg_counts = fg_df['category'].value_counts()
    bg_counts = bg_df['category'].value_counts()
    
    categories = sorted(list(set(fg_counts.index) | set(bg_counts.index)))
    
    enrich_data = []
    for cat in categories:
        fg_n = fg_counts.get(cat, 0)
        bg_n = bg_counts.get(cat, 0)
        
        fg_prop = fg_n / len(fg_df)
        bg_prop = bg_n / len(bg_df)
        
        enrichment = fg_prop / bg_prop if bg_prop > 0 else 0
        
        table = [[fg_n, len(fg_df) - fg_n], [bg_n, len(bg_df) - bg_n]]
        _, p_val = fisher_exact(table, alternative='greater')
        
        is_reg = cat in ["5'UTR", "3'UTR", "Upstream", "Downstream"]
        
        enrich_data.append({
            'category': cat,
            'enrichment': enrichment,
            'p_value': p_val,
            'is_regulatory': is_reg
        })
    
    enrich_df = pd.DataFrame(enrich_data).sort_values('enrichment', ascending=True)
    
    # Plot
    y_positions = np.arange(len(enrich_df))
    
    for i, row in enumerate(enrich_df.itertuples()):
        ax.plot([0, row.enrichment], [i, i], color='#B8B8B8', alpha=0.4, linewidth=1, zorder=1)
    
    point_colors = [script_colors.regulatory if row['is_regulatory'] else script_colors.nonregulatory 
                   for _, row in enrich_df.iterrows()]
    
    ax.scatter(enrich_df['enrichment'], y_positions, 
              c=point_colors, s=100, zorder=3, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Null line
    ax.axvline(1.0, linestyle='--', color='#1b4332', alpha=0.3, linewidth=1.5, zorder=0)
    
    # Significance stars
    for i, row in enrich_df.iterrows():
        if row['p_value'] < 0.001:
            stars = '***'
        elif row['p_value'] < 0.01:
            stars = '**'
        elif row['p_value'] < 0.05:
            stars = '*'
        else:
            stars = ''
        
        if stars:
            ax.text(row['enrichment'] + 0.2, y_positions[enrich_df.index.get_loc(i)],
                   stars, va='center', ha='left', fontsize=11, 
                   fontweight='bold', color=script_colors.regulatory)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(enrich_df['category'], fontsize=10)
    ax.set_xlabel('Fold Enrichment\n(Influential / Background)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, enrich_df['enrichment'].max() * 1.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)
    
    # Panel label
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')
    
    return enrich_df


def plot_panel_b(ax, fg_df, bg_df, regulatory_cats):
    """
    Panel B: Consequence-stratified distance ECDFs
    
    Shows regulatory vs non-regulatory SNPs separately to demonstrate
    that regulatory SNPs ARE proximal, explaining the enrichment.
    """
    # Split foreground by regulatory status
    fg_reg = fg_df[fg_df['is_regulatory']].copy()
    fg_nonreg = fg_df[~fg_df['is_regulatory']].copy()
    
    fg_reg_dist = fg_reg['dist_to_tss'].dropna()
    fg_nonreg_dist = fg_nonreg['dist_to_tss'].dropna()
    
    # Plot ECDFs
    if len(fg_reg_dist) > 0:
        fg_reg_sorted = np.sort(fg_reg_dist)
        fg_reg_ecdf = np.arange(1, len(fg_reg_sorted) + 1) / len(fg_reg_sorted)
        ax.plot(fg_reg_sorted / 1000, fg_reg_ecdf, 
               color=script_colors.regulatory, linewidth=3.5, 
               label='Regulatory', alpha=0.95, zorder=3)
    
    if len(fg_nonreg_dist) > 0:
        fg_nonreg_sorted = np.sort(fg_nonreg_dist)
        fg_nonreg_ecdf = np.arange(1, len(fg_nonreg_sorted) + 1) / len(fg_nonreg_sorted)
        ax.plot(fg_nonreg_sorted / 1000, fg_nonreg_ecdf, 
               color=script_colors.nonregulatory, linewidth=3.5, 
               label='Non-regulatory', alpha=0.95, zorder=2)
    
    # Statistical tests
    if len(fg_reg_dist) > 0 and len(fg_nonreg_dist) > 0:
        ks_stat, ks_p = ks_2samp(fg_reg_dist, fg_nonreg_dist)
        mw_stat, mw_p = mannwhitneyu(fg_reg_dist, fg_nonreg_dist, alternative='less')
        
        # Calculate medians
        reg_median = fg_reg_dist.median()
        nonreg_median = fg_nonreg_dist.median()
        
        # Stats box with key insight
        stats_text = (
            f"Regulatory: n={len(fg_reg_dist)}, median={reg_median/1000:.1f} kb\n"
            f"Non-regulatory: n={len(fg_nonreg_dist)}, median={nonreg_median/1000:.1f} kb\n"
            f"\n"
            f"KS test: D={ks_stat:.3f}, p={ks_p:.2e}\n"
            f"Regulatory SNPs are {nonreg_median/reg_median:.1f}x closer to TSS"
        )
        
        ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, 
               ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.6', fc='white', 
                        alpha=0.95, edgecolor='#B2DFDB', linewidth=1.5),
               family='monospace')
    
    # Reference lines for bins
    ax.axvline(5, color='#B8B8B8', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(20, color='#B8B8B8', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(100, color='#B8B8B8', linestyle=':', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Distance to TSS (kb)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Proportion', fontsize=11, fontweight='bold')
    
    # Legend
    legend = ax.legend(frameon=True, loc='upper left', fontsize=10,
                      title='Influential SNPs', title_fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#B2DFDB')
    legend.get_frame().set_linewidth(1.5)
    
    # Set limits
    x_max = max(fg_reg_dist.quantile(0.95), fg_nonreg_dist.quantile(0.95)) / 1000
    ax.set_xlim(0, x_max * 1.1)
    ax.set_ylim(0, 1.05)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.5)
    
    # Panel label
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')
    
    return fg_reg_dist, fg_nonreg_dist


def plot_panel_c(ax, fg_df, bg_df):
    """Panel C: Chi-square standardized residuals heatmap."""
    fg_labeled = fg_df.copy()
    bg_labeled = bg_df.copy()
    fg_labeled['set'] = 'Influential'
    bg_labeled['set'] = 'Background'
    
    combined = pd.concat([fg_labeled, bg_labeled], ignore_index=True)
    contingency = pd.crosstab(combined['set'], combined['category'])
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    residuals = (contingency - expected) / np.sqrt(expected)
    
    # Select top categories
    top_categories = contingency.sum(axis=0).nlargest(8).index
    residuals_subset = residuals[top_categories]
    
    # Heatmap
    im = ax.imshow(residuals_subset.values, cmap=script_colors.diverging_cmap, 
                  aspect='auto', vmin=-5, vmax=5)
    
    # Annotations
    for i in range(residuals_subset.shape[0]):
        for j in range(residuals_subset.shape[1]):
            val = residuals_subset.values[i, j]
            color = 'white' if abs(val) > 2.5 else '#1b4332'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(top_categories)))
    ax.set_xticklabels(top_categories, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Influential', 'Background'], fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Standardized Residual', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Stats
    stats_text = r"$\chi^{2}$" + f" = {chi2:.1f}\np = {p:.2e}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, va='top', ha='left',
           bbox=dict(boxstyle='round', fc='white', alpha=0.95, 
                    edgecolor='#B2DFDB', linewidth=1.5))
    
    # Panel label
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')
    
    return chi2, p, residuals_subset


def plot_panel_d(ax, fg_df, bg_df, regulatory_cats):
    """Panel D: Distance-stratified regulatory enrichment."""
    bins = [0, 5000, 20000, 100000, np.inf]
    labels = ['0-5kb', '5-20kb', '20-100kb', '>100kb']
    
    fg_df['dist_bin'] = pd.cut(fg_df['dist_to_tss'], bins=bins, labels=labels)
    bg_df['dist_bin'] = pd.cut(bg_df['dist_to_tss'], bins=bins, labels=labels)
    
    dist_enrich = []
    for bin_label in labels:
        fg_bin = fg_df[fg_df['dist_bin'] == bin_label]
        bg_bin = bg_df[bg_df['dist_bin'] == bin_label]
        
        if len(fg_bin) > 0 and len(bg_bin) > 0:
            fg_reg_n = fg_bin['is_regulatory'].sum()
            bg_reg_n = bg_bin['is_regulatory'].sum()
            
            fg_reg_prop = fg_reg_n / len(fg_bin)
            bg_reg_prop = bg_reg_n / len(bg_bin)
            
            enrichment = fg_reg_prop / bg_reg_prop if bg_reg_prop > 0 else 0
            
            table = [[fg_reg_n, len(fg_bin) - fg_reg_n],
                    [bg_reg_n, len(bg_bin) - bg_reg_n]]
            _, p_val = fisher_exact(table, alternative='greater')
            
            dist_enrich.append({
                'bin': bin_label,
                'enrichment': enrichment,
                'p_value': p_val,
                'fg_n': len(fg_bin),
                'fg_reg_n': fg_reg_n
            })
    
    dist_df = pd.DataFrame(dist_enrich)
    
    # Bar plot
    x_pos = np.arange(len(dist_df))
    bars = ax.bar(x_pos, dist_df['enrichment'], 
                  color=script_colors.regulatory, alpha=0.8,
                  edgecolor='#1b4332', linewidth=1.5)
    
    # Null line
    ax.axhline(1.0, linestyle='--', color='#1b4332', alpha=0.4, linewidth=1.5, zorder=0)
    
    # Significance stars
    for i, row in dist_df.iterrows():
        if row['p_value'] < 0.001:
            stars = '***'
        elif row['p_value'] < 0.01:
            stars = '**'
        elif row['p_value'] < 0.05:
            stars = '*'
        else:
            stars = ''
        
        if stars:
            ax.text(i, row['enrichment'] + 0.15, stars,
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dist_df['bin'], fontsize=10)
    ax.set_xlabel('Distance to TSS', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fold Enrichment\n(Regulatory Consequences)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(2.5, dist_df['enrichment'].max() * 1.3))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    
    # Stats box
    max_enrich = dist_df['enrichment'].max()
    max_bin = dist_df.loc[dist_df['enrichment'].idxmax(), 'bin']
    stats_text = f"Peak enrichment:\n  {max_enrich:.2f}x @ {max_bin}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, va='top', ha='right',
           bbox=dict(boxstyle='round', fc='white', alpha=0.95, 
                    edgecolor='#B2DFDB', linewidth=1.5))
    
    # Panel label
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')
    
    return dist_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate publication-quality Figure S2."""
    
    # Load data
    fg_snps, bg_snps, regulatory_cats = load_and_process_data()
    
    # Create figure
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30,
                 left=0.08, right=0.96, top=0.94, bottom=0.07)
    
    # Create subplots
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Generate panels
    print("  Panel A: Enrichment by consequence type...")
    enrich_df = plot_panel_a(ax_a, fg_snps, bg_snps)
    
    print("  Panel B: Consequence-stratified distances...")
    fg_reg_dist, fg_nonreg_dist = plot_panel_b(ax_b, fg_snps, bg_snps, regulatory_cats)
    
    print("  Panel C: Chi-square residuals...")
    chi2, p_val, residuals_df = plot_panel_c(ax_c, fg_snps, bg_snps)
    
    print("  Panel D: Distance-stratified enrichment...")
    dist_df = plot_panel_d(ax_d, fg_snps, bg_snps, regulatory_cats)
    
    # Save figure
    print("\nSaving figure...")
    output_path = OUTPUT_DIR / "figure_s2_publication.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  PNG: {output_path}")
    
    output_path_pdf = OUTPUT_DIR / "figure_s2_publication.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"  PDF: {output_path_pdf}")
    
    plt.close()
    
    # Save source data
    print("\nSaving source data...")
    enrich_df.to_csv(SOURCE_DIR / "figS2_panelA_enrichment.csv", index=False)
    
    pd.DataFrame({
        'regulatory_dist_kb': fg_reg_dist / 1000,
    }).to_csv(SOURCE_DIR / "figS2_panelB_regulatory_distances.csv", index=False)
    
    pd.DataFrame({
        'nonregulatory_dist_kb': fg_nonreg_dist / 1000,
    }).to_csv(SOURCE_DIR / "figS2_panelB_nonregulatory_distances.csv", index=False)
    
    residuals_df.to_csv(SOURCE_DIR / "figS2_panelC_residuals.csv")
    dist_df.to_csv(SOURCE_DIR / "figS2_panelD_distance_enrichment.csv", index=False)
    
    print(f"  Source data: {SOURCE_DIR}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("KEY RESULTS")
    print("=" * 80)
    
    # Overall enrichment
    fg_reg = fg_snps['is_regulatory'].sum()
    bg_reg = bg_snps['is_regulatory'].sum()
    fg_reg_pct = 100 * fg_reg / len(fg_snps)
    bg_reg_pct = 100 * bg_reg / len(bg_snps)
    overall_enrich = fg_reg_pct / bg_reg_pct if bg_reg_pct > 0 else 0
    
    table = [[fg_reg, len(fg_snps) - fg_reg],
            [bg_reg, len(bg_snps) - bg_reg]]
    _, reg_p = fisher_exact(table, alternative='greater')
    
    print(f"\nOverall regulatory enrichment:")
    print(f"  Influential: {fg_reg}/{len(fg_snps)} ({fg_reg_pct:.1f}%)")
    print(f"  Background: {bg_reg}/{len(bg_snps)} ({bg_reg_pct:.1f}%)")
    print(f"  Enrichment: {overall_enrich:.2f}x")
    print(f"  p-value: {reg_p:.2e}")
    
    print(f"\nPanel B - Consequence-stratified distances:")
    print(f"  Regulatory SNPs median: {fg_reg_dist.median()/1000:.1f} kb")
    print(f"  Non-regulatory SNPs median: {fg_nonreg_dist.median()/1000:.1f} kb")
    print(f"  Ratio: {fg_nonreg_dist.median()/fg_reg_dist.median():.1f}x farther")
    print(f"  Regulatory SNPs are proximal")
    
    print(f"\nPanel D - Distance-stratified enrichment:")
    for _, row in dist_df.iterrows():
        print(f"  {row['bin']:10s}: {row['enrichment']:.2f}x (p={row['p_value']:.2e})")
    
    print("\n" + "=" * 80)
    print("FIGURE S2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()