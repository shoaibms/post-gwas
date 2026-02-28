#!/usr/bin/env python3
"""
Window stability analysis and platinum modulator set definition.

Computes pairwise enrichment ratios, Jaccard indices, and Spearman rank
correlations across cis-window sizes, then defines the platinum set as
genes appearing in at least 2 of 3 windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, hypergeom, norm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

# Configuration
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
LABELS_DIR = BASE_DIR / "output" / "ect_alt" / "integrated"
OUTPUT_DIR = BASE_DIR / "output" / "week1_stability"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input files
FILES = {
    "500kb": LABELS_DIR / "decouple_labels_500kb.csv",
    "1Mb": LABELS_DIR / "decouple_labels_1Mb.csv",
    "2Mb": LABELS_DIR / "decouple_labels_2Mb.csv"
}

# Total candidate pool size
N_TOTAL_GENES = 500


def load_modulator_sets(files: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load decoupling label files and extract network modulators.
    
    Parameters
    ----------
    files : dict
        Dictionary mapping window size to file paths
        
    Returns
    -------
    dict
        Dictionary mapping window size to full dataframes
    """
    datasets = {}
    for window, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Missing file: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {window}: {len(df)} genes total")
        
        # Count by label
        label_counts = df['label'].value_counts()
        print(f"  network_modulator: {label_counts.get('network_modulator', 0)}")
        print(f"  additive_driver: {label_counts.get('additive_driver', 0)}")
        print(f"  unclear: {label_counts.get('unclear', 0)}")
        
        datasets[window] = df
    
    return datasets


def calculate_enrichment_ratio(set_a_genes: set, set_b_genes: set, 
                               n_total: int) -> Tuple[float, float]:
    """
    Calculate enrichment ratio and hypergeometric p-value.
    
    ER = observed_overlap / expected_overlap
    Expected = (|A| * |B|) / N
    
    Parameters
    ----------
    set_a_genes : set
        First gene set
    set_b_genes : set
        Second gene set
    n_total : int
        Total candidate pool size
        
    Returns
    -------
    tuple
        (enrichment_ratio, p_value)
    """
    n_a = len(set_a_genes)
    n_b = len(set_b_genes)
    observed = len(set_a_genes & set_b_genes)
    
    # Expected overlap by chance
    expected = (n_a * n_b) / n_total
    
    # Enrichment ratio
    if expected == 0:
        enrichment_ratio = np.nan
    else:
        enrichment_ratio = observed / expected
    
    # Hypergeometric test for enrichment
    # P(X >= observed) where X ~ Hypergeometric(N, |A|, |B|)
    p_value = hypergeom.sf(observed - 1, n_total, n_a, n_b)
    
    return enrichment_ratio, p_value


def calculate_jaccard_index(set_a: set, set_b: set) -> float:
    """
    Calculate Jaccard similarity index.
    
    J = |A intersection B| / |A union B|
    """
    if len(set_a | set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def odds_ratio_and_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05):
    """
    Odds ratio and (1-alpha) CI from a 2x2 table:
              SetB   ~SetB
      SetA      a       b
      ~SetA     c       d
    Uses Haldane-Anscombe 0.5 correction if any cell is zero.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0.0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
    z = norm.ppf(1 - alpha/2.0)
    lo = np.exp(np.log(or_val) - z * se)
    hi = np.exp(np.log(or_val) + z * se)
    return float(or_val), float(lo), float(hi)


def risk_ratio_and_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05):
    """
    Katz log CI for risk/enrichment ratio using the same 2x2:
      RR = (a/(a+b)) / (c/(c+d))
    Applies 0.5 continuity correction if any cell is zero.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0.0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    n1, n2 = a + b, c + d
    rr = (a / n1) / (c / n2) if c > 0 and n2 > 0 else np.inf
    se = np.sqrt((1.0/a) - (1.0/n1) + (1.0/c) - (1.0/n2))
    z = norm.ppf(1 - alpha/2.0)
    lo = np.exp(np.log(rr) - z * se)
    hi = np.exp(np.log(rr) + z * se)
    return float(rr), float(lo), float(hi)


def calculate_rank_correlation(df_a: pd.DataFrame, df_b: pd.DataFrame,
                               common_genes: List[str]) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Spearman rank correlation with bootstrap CI.
    
    Parameters
    ----------
    df_a, df_b : DataFrame
        DataFrames with interaction scores
    common_genes : list
        List of genes present in both datasets
        
    Returns
    -------
    tuple
        (spearman_rho, (ci_lower, ci_upper))
    """
    # Subset to common genes
    df_a_common = df_a[df_a['gene'].isin(common_genes)].set_index('gene')
    df_b_common = df_b[df_b['gene'].isin(common_genes)].set_index('gene')
    
    # Align by gene
    df_a_common = df_a_common.loc[common_genes]
    df_b_common = df_b_common.loc[common_genes]
    
    # Calculate interaction score: max(H, gxe_fraction)
    score_a = np.maximum(df_a_common['H'].fillna(0), 
                         df_a_common['gxe_fraction'].fillna(0))
    score_b = np.maximum(df_b_common['H'].fillna(0), 
                         df_b_common['gxe_fraction'].fillna(0))
    
    # Spearman correlation
    rho, p_value = spearmanr(score_a, score_b)
    
    # Bootstrap 95% CI
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_rhos = []
    
    n_genes = len(common_genes)
    for _ in range(n_boot):
        idx = rng.choice(n_genes, size=n_genes, replace=True)
        boot_rho, _ = spearmanr(score_a.iloc[idx], score_b.iloc[idx])
        boot_rhos.append(boot_rho)
    
    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)
    
    return rho, (ci_lower, ci_upper)


def compute_pairwise_stability(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute all pairwise stability metrics.
    
    Returns
    -------
    DataFrame
        Stability metrics for all window pairs
    """
    windows = list(datasets.keys())
    results = []
    
    for i, window_a in enumerate(windows):
        for window_b in windows[i+1:]:
            print(f"\nComparing {window_a} vs {window_b}...")
            
            df_a = datasets[window_a]
            df_b = datasets[window_b]
            
            # Extract modulator sets
            mods_a = set(df_a[df_a['label'] == 'network_modulator']['gene'])
            mods_b = set(df_b[df_b['label'] == 'network_modulator']['gene'])
            
            # Common genes and overlap
            n_common = len(mods_a & mods_b)
            common_genes = list(mods_a & mods_b)
            
            # Jaccard index
            jaccard = calculate_jaccard_index(mods_a, mods_b)
            
            # Enrichment ratio
            er, p_hyper = calculate_enrichment_ratio(mods_a, mods_b, N_TOTAL_GENES)
            
            # 2x2 table for A vs B membership
            a = len(mods_a & mods_b)                  # overlap
            b = len(mods_a - mods_b)
            c = len(mods_b - mods_a)
            d = N_TOTAL_GENES - (a + b + c)
            or_val, or_lo, or_hi = odds_ratio_and_ci(a, b, c, d)
            rr_val, rr_lo, rr_hi = risk_ratio_and_ci(a, b, c, d)
            
            # Rank correlation (all genes)
            all_genes = set(df_a['gene']) & set(df_b['gene'])
            rho, (ci_low, ci_high) = calculate_rank_correlation(
                df_a, df_b, list(all_genes)
            )
            
            results.append({
                'window_pair': f"{window_a} vs {window_b}",
                'n_set1': len(mods_a),
                'n_set2': len(mods_b),
                'n_overlap': n_common,
                'jaccard_index': jaccard,
                'enrichment_ratio': er,
                'er_p_value': p_hyper,
                'overlap_or': or_val,
                'overlap_or_ci_lower': or_lo,
                'overlap_or_ci_upper': or_hi,
                'overlap_rr': rr_val,
                'overlap_rr_ci_lower': rr_lo,
                'overlap_rr_ci_upper': rr_hi,
                'spearman_rho': rho,
                'spearman_ci_lower': ci_low,
                'spearman_ci_upper': ci_high
            })
            
            print(f"  Overlap: {n_common} genes")
            print(f"  Jaccard: {jaccard:.4f}")
            print(f"  Enrichment Ratio: {er:.2f}x (p={p_hyper:.2e}); "
                  f"OR={or_val:.2f} [95% CI {or_lo:.2f}, {or_hi:.2f}] ; "
                  f"RR={rr_val:.2f} [95% CI {rr_lo:.2f}, {rr_hi:.2f}]")
            print(f"  Spearman rho: {rho:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
    
    return pd.DataFrame(results)


def define_platinum_set(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Define platinum modulator set: genes appearing in >=2 of 3 windows.
    
    Returns
    -------
    DataFrame
        Platinum modulator genes with appearance counts
    """
    print("\n" + "="*60)
    print("DEFINING PLATINUM MODULATOR SET")
    print("="*60)
    
    # Count appearances per gene
    gene_appearances = {}
    windows_per_gene = {}
    
    for window, df in datasets.items():
        modulators = df[df['label'] == 'network_modulator']['gene'].tolist()
        
        for gene in modulators:
            gene_appearances[gene] = gene_appearances.get(gene, 0) + 1
            if gene not in windows_per_gene:
                windows_per_gene[gene] = []
            windows_per_gene[gene].append(window)
    
    # Filter to genes appearing in >=2 windows
    platinum_genes = {gene: count for gene, count in gene_appearances.items() 
                     if count >= 2}
    
    # Create summary dataframe
    platinum_df = pd.DataFrame([
        {
            'gene': gene,
            'n_windows': count,
            'windows': ', '.join(windows_per_gene[gene])
        }
        for gene, count in platinum_genes.items()
    ]).sort_values('n_windows', ascending=False)
    
    print(f"\nPlatinum Set Size: {len(platinum_df)} genes")
    print(f"  Appear in 3 windows: {(platinum_df['n_windows'] == 3).sum()}")
    print(f"  Appear in 2 windows: {(platinum_df['n_windows'] == 2).sum()}")
    
    return platinum_df


def plot_stability_heatmap(stability_df: pd.DataFrame, output_path: Path):
    """
    Create heatmap visualization of stability metrics.
    """
    # Prepare data for heatmap
    windows = ['500kb', '1Mb', '2Mb']
    n_windows = len(windows)
    
    # Initialize matrices
    jaccard_matrix = np.zeros((n_windows, n_windows))
    er_matrix = np.zeros((n_windows, n_windows))
    rho_matrix = np.zeros((n_windows, n_windows))
    
    # Fill diagonal with 1.0
    np.fill_diagonal(jaccard_matrix, 1.0)
    np.fill_diagonal(er_matrix, 1.0)
    np.fill_diagonal(rho_matrix, 1.0)
    
    # Map window names to indices
    window_idx = {w: i for i, w in enumerate(windows)}
    
    # Fill matrices from stability_df
    for _, row in stability_df.iterrows():
        pair = row['window_pair']
        w1, w2 = pair.split(' vs ')
        i, j = window_idx[w1], window_idx[w2]
        
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = row['jaccard_index']
        er_matrix[i, j] = er_matrix[j, i] = row['enrichment_ratio']
        rho_matrix[i, j] = rho_matrix[j, i] = row['spearman_rho']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot Jaccard
    sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=windows, yticklabels=windows, 
                vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Jaccard Index'})
    axes[0].set_title('Jaccard Similarity', fontweight='bold')
    
    # Plot Enrichment Ratio
    sns.heatmap(er_matrix, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=windows, yticklabels=windows,
                vmin=0, ax=axes[1], cbar_kws={'label': 'Enrichment Ratio'})
    axes[1].set_title('Enrichment Ratio', fontweight='bold')
    
    # Plot Spearman
    sns.heatmap(rho_matrix, annot=True, fmt='.3f', cmap='Purples',
                xticklabels=windows, yticklabels=windows,
                vmin=0, vmax=1, ax=axes[2], cbar_kws={'label': 'Spearman rho'})
    axes[2].set_title('Rank Correlation', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved stability heatmap: {output_path}")


def plot_spearman_barplot(stability_df: pd.DataFrame, output_path: Path):
    """
    Create bar plot of Spearman correlations with 95% CI error bars.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x_pos = np.arange(len(stability_df))
    
    # Plot bars
    bars = ax.bar(x_pos, stability_df['spearman_rho'], 
                   color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add error bars (95% CI)
    errors_lower = stability_df['spearman_rho'] - stability_df['spearman_ci_lower']
    errors_upper = stability_df['spearman_ci_upper'] - stability_df['spearman_rho']
    
    ax.errorbar(x_pos, stability_df['spearman_rho'],
                yerr=[errors_lower, errors_upper],
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stability_df['window_pair'], rotation=0)
    ax.set_ylabel('Spearman Rank Correlation (rho)', fontweight='bold')
    ax.set_xlabel('Window Comparison', fontweight='bold')
    ax.set_title('Rank Stability Across Window Sizes', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Spearman barplot: {output_path}")


def evaluate_decision_gate_1(stability_df: pd.DataFrame, 
                             platinum_df: pd.DataFrame) -> bool:
    """
    Evaluate Decision Gate 1 criteria.
    
    Proceed if:
    - Platinum set >= 12 genes
    - 500kb-2Mb: ER > 2.5 AND Spearman rho >= 0.65 (CI excludes zero)
    
    Returns
    -------
    bool
        True if all criteria met
    """
    print("\n" + "="*60)
    print("DECISION GATE 1: WINDOW STABILITY")
    print("="*60)
    
    # Criterion 1: Platinum set size
    n_platinum = len(platinum_df)
    criterion_1 = n_platinum >= 12
    
    print(f"\nCriterion 1: Platinum Set Size")
    print(f"  Required: >= 12 genes")
    print(f"  Actual: {n_platinum} genes")
    print(f"  Status: {'PASS' if criterion_1 else 'FAIL'}")
    
    # Criterion 2: 500kb-2Mb stability
    row_500_2mb = stability_df[stability_df['window_pair'] == '500kb vs 2Mb'].iloc[0]
    er = row_500_2mb['enrichment_ratio']
    rho = row_500_2mb['spearman_rho']
    ci_lower = row_500_2mb['spearman_ci_lower']
    
    criterion_2a = er > 2.5
    criterion_2b = (rho >= 0.65) and (ci_lower > 0)
    criterion_2 = criterion_2a and criterion_2b
    
    print(f"\nCriterion 2: 500kb-2Mb Stability")
    print(f"  Enrichment Ratio: {er:.2f}x (required: > 2.5)")
    print(f"    Status: {'PASS' if criterion_2a else 'FAIL'}")
    print(f"  Spearman rho: {rho:.3f} (required: >= 0.65)")
    print(f"  95% CI lower bound: {ci_lower:.3f} (required: > 0)")
    print(f"    Status: {'PASS' if criterion_2b else 'FAIL'}")
    
    # Overall decision
    pass_gate = criterion_1 and criterion_2
    
    print(f"\n{'='*60}")
    print(f"GATE 1 DECISION: {'PROCEED TO WEEK 2' if pass_gate else 'REEVALUATE APPROACH'}")
    print(f"{'='*60}")
    
    if not pass_gate:
        print("\nRECOMMENDED ACTIONS:")
        if not criterion_1:
            print("  - Review gene classification thresholds")
            print("  - Analyze continuous interaction scores")
        if not criterion_2:
            print("  - Verify window-specific results")
            print("  - Consider focusing on strongest window")
            print("  - Pivot to methods-focused paper (Path C)")
    
    return pass_gate


def main():
    """
    Main execution function for Week 1 analysis.
    """
    print("="*60)
    print("WEEK 1: WINDOW STABILITY ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nStep 1: Loading decoupling labels...")
    datasets = load_modulator_sets(FILES)
    
    # Compute pairwise stability
    print("\nStep 2: Computing pairwise stability metrics...")
    stability_df = compute_pairwise_stability(datasets)
    
    # Save stability results
    stability_output = OUTPUT_DIR / "window_stability_metrics.csv"
    stability_df.to_csv(stability_output, index=False)
    print(f"\nSaved: {stability_output}")
    
    # Define platinum set
    print("\nStep 3: Defining platinum modulator set...")
    platinum_df = define_platinum_set(datasets)
    
    # Save platinum set
    platinum_output = OUTPUT_DIR / "platinum_modulator_set.csv"
    platinum_df.to_csv(platinum_output, index=False)
    print(f"Saved: {platinum_output}")
    
    # Create visualizations
    print("\nStep 4: Creating visualizations...")
    plot_stability_heatmap(stability_df, FIGURES_DIR / "stability_heatmap.png")
    plot_spearman_barplot(stability_df, FIGURES_DIR / "spearman_correlation.png")
    
    # Evaluate Decision Gate 1
    print("\nStep 5: Evaluating Decision Gate 1...")
    gate_pass = evaluate_decision_gate_1(stability_df, platinum_df)
    
    # Save gate decision
    gate_decision = {
        'gate': 'Gate 1: Window Stability',
        'decision': 'PASS' if gate_pass else 'FAIL',
        'n_platinum_genes': len(platinum_df),
        'er_500kb_2mb': stability_df[stability_df['window_pair'] == '500kb vs 2Mb']['enrichment_ratio'].iloc[0],
        'rho_500kb_2mb': stability_df[stability_df['window_pair'] == '500kb vs 2Mb']['spearman_rho'].iloc[0]
    }
    
    gate_df = pd.DataFrame([gate_decision])
    gate_output = OUTPUT_DIR / "gate1_decision.csv"
    gate_df.to_csv(gate_output, index=False)
    print(f"\nSaved gate decision: {gate_output}")
    
    print("\n" + "="*60)
    print("WEEK 1 ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    if gate_pass:
        print("\nPROCEED TO WEEK 2: SNP Selection & Consequence Enrichment")
    else:
        print("\nREEVALUATE APPROACH - Consider Path C (methods paper)")


if __name__ == "__main__":
    main()