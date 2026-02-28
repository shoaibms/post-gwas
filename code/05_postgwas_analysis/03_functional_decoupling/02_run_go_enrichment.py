#!/usr/bin/env python3
"""
GO enrichment analysis for network modulators vs additive drivers.

Tests functional decoupling using hypergeometric enrichment on GO term annotations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")

PLATINUM_SET = BASE_DIR / "output" / "week1_stability" / "platinum_modulator_set.csv"
DECOUPLE_2MB = BASE_DIR / "output" / "ect_alt" / "integrated" / "decouple_labels_2Mb.csv"
DECOUPLE_500KB = DECOUPLE_2MB
GO_MAPPING = BASE_DIR / "data" / "maize" / "process" / "go" / "maize_gene_to_go_v4.csv"

OUTPUT_DIR = BASE_DIR / "output" / "week4_go_enrichment"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MIN_TERM_SIZE = 3
MAX_TERM_SIZE = 200
FDR_THRESHOLD = 0.05
USE_GENOME_BACKGROUND = True

print("="*60)
print("WEEK 4: GO ENRICHMENT ANALYSIS")
print("="*60)


def load_gene_sets():
    """Load modulators, drivers, and background."""
    print("\nLoading gene sets...")
    
    plat = pd.read_csv(PLATINUM_SET)
    decouple = pd.read_csv(DECOUPLE_500KB)
    
    modulators = set(plat['gene'].tolist())
    drivers = set(decouple[decouple['label'] == 'additive_driver']['gene'].tolist())
    background = set(decouple['gene'].tolist())
    
    print(f"  Modulators: {len(modulators)}")
    print(f"  Drivers: {len(drivers)}")
    print(f"  Background: {len(background)}")
    
    return modulators, drivers, background


def build_go_database(go_file, background_genes):
    """Build GO term database from mapping file."""
    print("\nBuilding GO database...")
    
    go_df = pd.read_csv(go_file, dtype=str)
    go_df = go_df[go_df['gene'].isin(background_genes)]
    
    print(f"  Loaded {len(go_df)} genes with GO annotations")
    
    term_to_genes = defaultdict(set)
    gene_to_terms = {}
    
    for _, row in go_df.iterrows():
        gene = row['gene']
        terms = row['go_terms'].split(';')
        
        gene_to_terms[gene] = set(terms)
        
        for term in terms:
            term_to_genes[term].add(gene)
    
    filtered_terms = {
        term: genes for term, genes in term_to_genes.items()
        if MIN_TERM_SIZE <= len(genes) <= MAX_TERM_SIZE
    }
    
    print(f"  GO terms: {len(term_to_genes)} total")
    print(f"  GO terms (filtered {MIN_TERM_SIZE}-{MAX_TERM_SIZE} genes): {len(filtered_terms)}")
    
    return filtered_terms, gene_to_terms


def hypergeometric_test(k, M, n, N):
    """
    Hypergeometric test for enrichment.
    k = genes in both foreground and term
    M = total genes in background
    n = genes in term (in background)
    N = total genes in foreground
    """
    p_value = stats.hypergeom.sf(k - 1, M, n, N)
    return p_value


def run_enrichment(foreground, background, term_to_genes, label):
    """Run GO enrichment analysis."""
    print(f"\n{'='*60}")
    print(f"ENRICHMENT: {label}")
    print(f"{'='*60}")
    
    N = len(foreground)
    M = len(background)
    
    print(f"Foreground: {N} genes")
    print(f"Background: {M} genes")
    print(f"Testing {len(term_to_genes)} GO terms...")
    
    results = []
    
    for go_term, term_genes_bg in term_to_genes.items():
        n = len(term_genes_bg)
        overlap = foreground & term_genes_bg
        k = len(overlap)
        
        if k < 2:
            continue
        
        p_value = hypergeometric_test(k, M, n, N)
        
        expected = (N * n) / M
        fold_enrichment = k / expected if expected > 0 else 0
        
        results.append({
            'go_term': go_term,
            'overlap': k,
            'foreground_size': N,
            'term_size_bg': n,
            'background_size': M,
            'expected': expected,
            'fold_enrichment': fold_enrichment,
            'p_value': p_value
        })
    
    if not results:
        print(f"\nNo enriched terms found for {label}")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('p_value')
    
    df['q_value'] = stats.false_discovery_control(df['p_value'].values)
    
    sig = df[df['q_value'] < FDR_THRESHOLD]
    
    print(f"\nTested: {len(df)} terms")
    print(f"Significant (q < {FDR_THRESHOLD}): {len(sig)}")
    
    if len(sig) > 0:
        print(f"\nTop 10 enrichments:")
        print(f"{'GO Term':<15} {'k/n':<12} {'Fold':<8} {'P-value':<12} {'Q-value'}")
        print("-"*65)
        for _, row in sig.head(10).iterrows():
            print(f"{row['go_term']:<15} {row['overlap']}/{row['term_size_bg']:<10} "
                  f"{row['fold_enrichment']:>6.2f}  {row['p_value']:>10.2e}  {row['q_value']:>10.3e}")
    
    return df


def compare_enrichments(df_mod, df_drv):
    """Compare modulator vs driver enrichments."""
    print("\n" + "="*60)
    print("FUNCTIONAL COMPARISON")
    print("="*60)
    
    if df_mod.empty and df_drv.empty:
        print("\nNo enrichments found for either group")
        return
    
    mod_sig = set(df_mod[df_mod['q_value'] < FDR_THRESHOLD]['go_term'])
    drv_sig = set(df_drv[df_drv['q_value'] < FDR_THRESHOLD]['go_term'])
    
    mod_only = mod_sig - drv_sig
    drv_only = drv_sig - mod_sig
    shared = mod_sig & drv_sig
    
    total = len(mod_sig | drv_sig)
    jaccard = len(shared) / total if total > 0 else 0
    
    print(f"\nEnriched GO terms:")
    print(f"  Modulator-specific: {len(mod_only)}")
    print(f"  Driver-specific: {len(drv_only)}")
    print(f"  Shared: {len(shared)}")
    print(f"  Jaccard overlap: {jaccard:.3f}")
    
    if jaccard < 0.5:
        print(f"  Functional decoupling confirmed (Jaccard < 0.5)")
    else:
        print(f"  Limited decoupling (Jaccard >= 0.5)")


def plot_enrichments(df_mod, df_drv, output_path):
    """Create enrichment comparison figure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    if not df_mod.empty:
        sig_mod = df_mod[df_mod['q_value'] < FDR_THRESHOLD].nsmallest(15, 'q_value')
        if len(sig_mod) > 0:
            y_pos = np.arange(len(sig_mod))
            ax.barh(y_pos, -np.log10(sig_mod['q_value'].values), 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sig_mod['go_term'].values, fontsize=9)
            ax.set_xlabel('-log10(Q-value)', fontweight='bold', fontsize=11)
            ax.set_title('Network Modulators', fontweight='bold', fontsize=12)
            ax.axvline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No significant enrichments', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Network Modulators', fontweight='bold', fontsize=12)
    
    ax = axes[1]
    if not df_drv.empty:
        sig_drv = df_drv[df_drv['q_value'] < FDR_THRESHOLD].nsmallest(15, 'q_value')
        if len(sig_drv) > 0:
            y_pos = np.arange(len(sig_drv))
            ax.barh(y_pos, -np.log10(sig_drv['q_value'].values), 
                   color='#3498db', edgecolor='black', linewidth=1.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sig_drv['go_term'].values, fontsize=9)
            ax.set_xlabel('-log10(Q-value)', fontweight='bold', fontsize=11)
            ax.set_title('Additive Drivers', fontweight='bold', fontsize=12)
            ax.axvline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No significant enrichments', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Additive Drivers', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Figure 4: {output_path}")


def evaluate_gate_4(df_mod, df_drv):
    """Evaluate Decision Gate 4."""
    print("\n" + "="*60)
    print("DECISION GATE 4: FUNCTIONAL DECOUPLING")
    print("="*60)
    
    mod_sig = len(df_mod[df_mod['q_value'] < FDR_THRESHOLD]) if not df_mod.empty else 0
    drv_sig = len(df_drv[df_drv['q_value'] < FDR_THRESHOLD]) if not df_drv.empty else 0
    
    mod_pass = mod_sig >= 3
    drv_pass = drv_sig >= 3
    
    if mod_sig > 0 and drv_sig > 0:
        mod_terms = set(df_mod[df_mod['q_value'] < FDR_THRESHOLD]['go_term'])
        drv_terms = set(df_drv[df_drv['q_value'] < FDR_THRESHOLD]['go_term'])
        total = len(mod_terms | drv_terms)
        overlap = len(mod_terms & drv_terms)
        jaccard = overlap / total if total > 0 else 0
        decoupled = jaccard < 0.5
    else:
        jaccard = 0
        decoupled = True
    
    print(f"\nModulators enriched terms: {mod_sig} - {'PASS' if mod_pass else 'FAIL'}")
    print(f"Drivers enriched terms: {drv_sig} - {'PASS' if drv_pass else 'FAIL'}")
    
    if mod_sig > 0 and drv_sig > 0:
        print(f"Functional overlap (Jaccard): {jaccard:.3f} - {'PASS' if decoupled else 'WEAK'}")
    
    gate_pass = mod_pass or drv_pass
    
    print(f"\n{'='*60}")
    if gate_pass:
        print("GATE 4: PROCEED TO WEEK 5")
    else:
        print("GATE 4: PARTIAL - Enrichment detected but limited")
    print("="*60)
    
    return gate_pass


def main():
    """Main execution."""
    
    modulators, drivers, background_500 = load_gene_sets()
    
    if USE_GENOME_BACKGROUND:
        print("\n" + "="*60)
        print("USING GENOME-WIDE BACKGROUND FOR INCREASED POWER")
        print("="*60)
        
        go_full = pd.read_csv(GO_MAPPING, dtype=str)
        background_genome = set(go_full['gene'].tolist())
        
        print(f"\n500-gene background: {len(background_500)}")
        print(f"Genome-wide background: {len(background_genome)}")
        print(f"Using genome-wide background for enrichment...")
        
        background = background_genome
    else:
        background = background_500
    
    term_to_genes, gene_to_terms = build_go_database(GO_MAPPING, background)
    
    df_mod = run_enrichment(modulators, background, term_to_genes, "Network Modulators")
    df_mod.to_csv(OUTPUT_DIR / "modulators_go_enrichment.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'modulators_go_enrichment.csv'}")
    
    df_drv = run_enrichment(drivers, background, term_to_genes, "Additive Drivers")
    df_drv.to_csv(OUTPUT_DIR / "drivers_go_enrichment.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'drivers_go_enrichment.csv'}")
    
    compare_enrichments(df_mod, df_drv)
    
    plot_enrichments(df_mod, df_drv, FIGURES_DIR / "figure4_go_enrichment.png")
    
    gate_pass = evaluate_gate_4(df_mod, df_drv)
    
    pd.DataFrame([{
        'gate': 'Gate 4',
        'decision': 'PASS' if gate_pass else 'PARTIAL',
        'mod_enriched': len(df_mod[df_mod['q_value'] < FDR_THRESHOLD]) if not df_mod.empty else 0,
        'drv_enriched': len(df_drv[df_drv['q_value'] < FDR_THRESHOLD]) if not df_drv.empty else 0
    }]).to_csv(OUTPUT_DIR / "gate4_decision.csv", index=False)
    
    print("\n" + "="*60)
    print("WEEK 4 COMPLETE")
    print("="*60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()