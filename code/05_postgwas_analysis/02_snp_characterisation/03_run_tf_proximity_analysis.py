#!/usr/bin/env python3
"""
TF binding proximity analysis for influential SNPs.

Tests whether foreground SNPs co-localize with TF binding sites
and identifies enriched TF families relative to matched background controls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import os

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
FOREGROUND = BASE_DIR / "output" / "week2_snp_selection" / "top_influential_snps_pragmatic.csv"
BACKGROUND = BASE_DIR / "output" / "week2_enrichment" / "background_matched_10x.csv"
TF_BINDING = BASE_DIR / "data" / "maize" / "media-10.gz"
OUTPUT_DIR = BASE_DIR / "output" / "week3_tf_binding"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROXIMITY_THRESHOLD = 1000
Z_MIN = 3.0
REQUIRE_ACR = True
DEDUP_BP = 200


def load_snp_sets():
    """Load foreground and background SNP sets."""
    print("="*60)
    print("WEEK 3: TF BINDING PROXIMITY ANALYSIS")
    print("="*60)
    
    print("\nLoading SNP sets...")
    fg = pd.read_csv(FOREGROUND)
    bg = pd.read_csv(BACKGROUND)
    
    print(f"  Foreground columns: {list(fg.columns)}")
    print(f"  Background columns: {list(bg.columns)}")
    
    fg['CHR_NORM'] = fg['chr'].astype(str).str.replace('chr', '', case=False)
    bg['CHR_NORM'] = bg['bg_chr'].astype(str).str.replace('chr', '', case=False)
    
    fg['POS'] = fg['pos'].astype(int)
    bg['POS'] = bg['bg_pos'].astype(int)
    
    fg['SNP_ID'] = fg['snp_id']
    bg['SNP_ID'] = bg['bg_snp_id']
    
    print(f"  Foreground: {len(fg)} SNPs")
    print(f"  Background: {len(bg)} SNPs")
    
    return fg, bg


def load_tf_binding_sites():
    """
    Load TF binding sites from media-10.gz file.
    
    Format: TF_assay   chr:pos   closest_gene   strand   distance   z-score   ACR   class   TF_ID
    Column 2 contains peak summit as 'chr1:17161'
    Column 1 contains TF assay info
    Column 9 contains TF gene ID
    """
    print("\nLoading TF binding sites...")
    
    import gzip
    
    tf_sites = []
    
    with gzip.open(TF_BINDING, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            
            if len(parts) < 9:
                continue
            
            try:
                tf_assay = parts[0]
                peak_loc = parts[1]
                tf_gene_id = parts[8]
                z_score = float(parts[5])
                acr_overlap = int(parts[6])
                
                if ':' not in peak_loc:
                    continue
                
                chr_part, pos_part = peak_loc.split(':')
                chr_val = chr_part.replace('chr', '').replace('Chr', '')
                
                if chr_val not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                    continue
                
                summit_pos = int(pos_part)
                
                tf_sites.append({
                    'chr': chr_val,
                    'summit': summit_pos,
                    'TF_name': tf_gene_id,
                    'TF_assay': tf_assay,
                    'z_score': z_score,
                    'acr_overlap': acr_overlap
                })
            except (ValueError, IndexError) as e:
                continue
    
    df = pd.DataFrame(tf_sites)
    
    if df.empty:
        raise RuntimeError("No TF binding sites loaded. Check file format.")
    
    print(f"  Initial TF sites: {len(df):,}")
    
    if REQUIRE_ACR:
        df = df[df['acr_overlap'] == 1]
        print(f"  After ACR filter: {len(df):,}")
    
    df = df[df['z_score'] >= Z_MIN]
    print(f"  After z-score >= {Z_MIN} filter: {len(df):,}")
    
    df = df.sort_values(['chr', 'summit'])
    df['delta'] = df.groupby('chr')['summit'].diff().fillna(10**9)
    df = df[df['delta'] > DEDUP_BP].drop(columns='delta')
    print(f"  After deduplication (>{DEDUP_BP}bp): {len(df):,}")
    
    print(f"  Final TF binding sites: {len(df):,}")
    print(f"  Chromosomes: {sorted(df['chr'].unique())}")
    print(f"  Unique TFs: {df['TF_name'].nunique()}")
    
    return df


def extract_tf_family(tf_name):
    """
    Extract TF family from TF name.
    Common patterns: WRKY, NAC, bZIP, MYB, GRAS, etc.
    """
    tf_upper = str(tf_name).upper()
    
    families = ['WRKY', 'NAC', 'BZIP', 'MYB', 'GRAS', 'AP2', 'ERF', 
                'MADS', 'HD-ZIP', 'TCP', 'BHLH', 'C2H2', 'DREB', 'ARF']
    
    for family in families:
        if family in tf_upper:
            return family
    
    return 'Other'


def calculate_distances_to_tf(snps, tf_sites):
    """
    For each SNP, calculate distance to nearest TF binding site.
    Fast path: per-chromosome binary search over sorted TF summits.
    """
    print(f"\nCalculating distances to TF binding sites...")
    N_JOBS = max(1, min(os.cpu_count() - 2, 36))
    def _one_chr(chr_id):
        chr_snps = snps[snps['CHR_NORM'] == chr_id].copy()
        chr_tf = tf_sites[tf_sites['chr'] == chr_id].copy()
        if chr_tf.empty or chr_snps.empty:
            if chr_tf.empty:
                print(f"  Warning: No TF sites for chr{chr_id}")
            return pd.DataFrame([])
        print(f"  Processing chr{chr_id}: {len(chr_snps)} SNPs, {len(chr_tf)} TF sites")
        snp_positions = chr_snps['POS'].to_numpy(np.int64)
        tf_summits = np.sort(chr_tf['summit'].to_numpy(np.int64))
        # binary search indices
        idx_right = np.searchsorted(tf_summits, snp_positions, side='left')
        # left neighbour
        left_idx = np.clip(idx_right - 1, 0, len(tf_summits) - 1)
        right_idx = np.clip(idx_right, 0, len(tf_summits) - 1)
        dist_left = np.abs(tf_summits[left_idx] - snp_positions)
        dist_right = np.abs(tf_summits[right_idx] - snp_positions)
        choose_right = dist_right < dist_left
        nearest_idx = np.where(choose_right, right_idx, left_idx)
        min_distance = np.minimum(dist_left, dist_right)
        nearest_tf = chr_tf.iloc[nearest_idx]
        out = pd.DataFrame({
            'snp_id': chr_snps['SNP_ID'].values,
            'chr': chr_id,
            'pos': snp_positions,
            'nearest_tf_distance': min_distance,
            'nearest_tf_name': nearest_tf['TF_name'].to_numpy(object),
            'nearest_tf_family': nearest_tf['TF_name'].map(extract_tf_family).to_numpy(object),
            'is_proximal': (min_distance <= PROXIMITY_THRESHOLD)
        })
        return out
    chroms = list(snps['CHR_NORM'].unique())
    parts = Parallel(n_jobs=N_JOBS, backend='loky', prefer='processes')(delayed(_one_chr)(c) for c in chroms)
    return pd.concat(parts, axis=0, ignore_index=True)


def test_proximity_enrichment(fg_distances, bg_distances):
    """Test enrichment of foreground SNPs near TF binding sites."""
    print("\n" + "="*60)
    print("TEST 1: TF PROXIMITY ENRICHMENT")
    print("="*60)
    
    fg_proximal = fg_distances['is_proximal'].sum()
    fg_total = len(fg_distances)
    fg_pct = (fg_proximal / fg_total) * 100
    
    bg_proximal = bg_distances['is_proximal'].sum()
    bg_total = len(bg_distances)
    bg_pct = (bg_proximal / bg_total) * 100
    
    contingency = np.array([
        [fg_proximal, fg_total - fg_proximal],
        [bg_proximal, bg_total - bg_proximal]
    ])
    
    odds_ratio, p_value = stats.fisher_exact(contingency)
    enrichment = fg_pct / bg_pct if bg_pct > 0 else np.inf
    # RR CI (Katz)
    def _rr_ci(a,b,c,d, alpha=0.05):
        a,b,c,d = float(a),float(b),float(c),float(d)
        if min(a,b,c,d)==0.0: a+=0.5; b+=0.5; c+=0.5; d+=0.5
        n1, n2 = a+b, c+d
        rr = (a/n1)/(c/n2) if c>0 and n2>0 else np.inf
        se = np.sqrt((1.0/a)-(1.0/n1)+(1.0/c)-(1.0/n2))
        z = stats.norm.ppf(1-alpha/2.0)
        return rr, np.exp(np.log(rr)-z*se), np.exp(np.log(rr)+z*se)
    rr, rr_lo, rr_hi = _rr_ci(fg_proximal, fg_total - fg_proximal, bg_proximal, bg_total - bg_proximal)
    
    threshold_kb = PROXIMITY_THRESHOLD / 1000
    print(f"\nForeground proximal to TF (<{threshold_kb:.1f}kb): {fg_proximal}/{fg_total} ({fg_pct:.1f}%)")
    print(f"Background proximal to TF (<{threshold_kb:.1f}kb): {bg_proximal}/{bg_total} ({bg_pct:.1f}%)")
    print(f"Enrichment: {enrichment:.2f}x  |  RR={rr:.2f} [95% CI {rr_lo:.2f}, {rr_hi:.2f}]")
    print(f"Fisher's exact p: {p_value:.3e}")
    print(f"Odds ratio: {odds_ratio:.2f}")
    
    return {
        'test': 'tf_proximity',
        'fg_proximal': fg_proximal,
        'fg_total': fg_total,
        'fg_pct': fg_pct,
        'bg_proximal': bg_proximal,
        'bg_total': bg_total,
        'bg_pct': bg_pct,
        'enrichment_fold': enrichment,
        'enrichment_ci_lower': float(rr_lo),
        'enrichment_ci_upper': float(rr_hi),
        'odds_ratio': odds_ratio,
        'p_value': p_value
    }


def test_tf_family_enrichment(fg_distances, bg_distances):
    """Test enrichment of specific TF families."""
    print("\n" + "="*60)
    print("TEST 2: TF FAMILY ENRICHMENT")
    print("="*60)
    
    fg_proximal = fg_distances[fg_distances['is_proximal']]
    bg_proximal = bg_distances[bg_distances['is_proximal']]
    
    fg_families = fg_proximal['nearest_tf_family'].value_counts()
    bg_families = bg_proximal['nearest_tf_family'].value_counts()
    
    # RR CI helper
    def _rr_ci(a,b,c,d, alpha=0.05):
        a,b,c,d = float(a),float(b),float(c),float(d)
        if min(a,b,c,d)==0.0: a+=0.5; b+=0.5; c+=0.5; d+=0.5
        n1, n2 = a+b, c+d
        rr = (a/n1)/(c/n2) if c>0 and n2>0 else np.inf
        se = np.sqrt((1.0/a)-(1.0/n1)+(1.0/c)-(1.0/n2))
        z = stats.norm.ppf(1-alpha/2.0)
        return rr, np.exp(np.log(rr)-z*se), np.exp(np.log(rr)+z*se)
    
    family_results = []
    
    print(f"\nTesting enrichment for TF families:")
    print(f"{'Family':<15} {'FG%':<8} {'BG%':<8} {'Enrichment':<12} {'P-value'}")
    print("-" * 60)
    
    all_families = set(fg_families.index) | set(bg_families.index)
    
    for family in sorted(all_families):
        fg_count = fg_families.get(family, 0)
        bg_count = bg_families.get(family, 0)
        
        fg_other = len(fg_proximal) - fg_count
        bg_other = len(bg_proximal) - bg_count
        
        if fg_count < 3 or bg_count < 3:
            continue
        
        contingency = np.array([
            [fg_count, fg_other],
            [bg_count, bg_other]
        ])
        
        odds_ratio, p_value = stats.fisher_exact(contingency)
        # RR CI for FG% vs BG%
        rr, rr_lo, rr_hi = _rr_ci(fg_count, fg_other, bg_count, bg_other)
        
        fg_pct = (fg_count / len(fg_proximal)) * 100
        bg_pct = (bg_count / len(bg_proximal)) * 100
        enrichment = fg_pct / bg_pct if bg_pct > 0 else np.inf
        
        print(f"{family:<15} {fg_pct:>6.1f}%  {bg_pct:>6.1f}%  {enrichment:>6.2f}x (RR {rr:.2f} [{rr_lo:.2f},{rr_hi:.2f}])   {p_value:.3e}")
        
        family_results.append({
            'tf_family': family,
            'fg_count': fg_count,
            'bg_count': bg_count,
            'fg_pct': fg_pct,
            'bg_pct': bg_pct,
            'enrichment': enrichment,
            'enrichment_ci_lower': float(rr_lo),
            'enrichment_ci_upper': float(rr_hi),
            'odds_ratio': odds_ratio,
            'p_value': p_value
        })
    
    df_families = pd.DataFrame(family_results)
    
    if not df_families.empty:
        df_families['q_value'] = stats.false_discovery_control(df_families['p_value'].values)
    
    return df_families


def compare_distance_distributions(fg_distances, bg_distances):
    """Compare distance distributions using statistical tests."""
    print("\n" + "="*60)
    print("TEST 3: DISTANCE DISTRIBUTION COMPARISON")
    print("="*60)
    
    fg_dists = fg_distances['nearest_tf_distance'].values
    bg_dists = bg_distances['nearest_tf_distance'].values
    
    ks_stat, ks_p = stats.ks_2samp(fg_dists, bg_dists)
    mw_stat, mw_p = stats.mannwhitneyu(fg_dists, bg_dists, alternative='less')
    
    print(f"\nForeground median distance: {np.median(fg_dists):,.0f} bp")
    print(f"Background median distance: {np.median(bg_dists):,.0f} bp")
    print(f"\nKolmogorov-Smirnov test: D={ks_stat:.3f}, p={ks_p:.3e}")
    print(f"Mann-Whitney U test (one-sided): U={mw_stat:,.0f}, p={mw_p:.3e}")
    
    return {
        'fg_median': np.median(fg_dists),
        'bg_median': np.median(bg_dists),
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p,
        'mw_statistic': mw_stat,
        'mw_p_value': mw_p
    }


def plot_tf_proximity_results(proximity_results, family_results, dist_comparison, output_path):
    """Create comprehensive TF proximity figure."""
    fig = plt.figure(figsize=(18, 6))
    
    fg_pct = proximity_results['fg_pct']
    bg_pct = proximity_results['bg_pct']
    
    threshold_kb = PROXIMITY_THRESHOLD / 1000
    
    ax1 = plt.subplot(131)
    x = ['Foreground', 'Background']
    y = [fg_pct, bg_pct]
    colors = ['#e74c3c', '#95a5a6']
    bars = ax1.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel(f'% within {threshold_kb:.1f}kb of TF', fontweight='bold')
    ax1.set_title('TF Proximity Enrichment', fontweight='bold')
    ax1.set_ylim(0, max(y) * 1.3 if max(y) > 0 else 10)
    
    p_text = f"p = {proximity_results['p_value']:.3f}" if proximity_results['p_value'] >= 0.001 else "p < 0.001"
    ax1.text(0.5, max(y) * 1.15, p_text, ha='center', fontweight='bold')
    
    for bar, val in zip(bars, y):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y)*0.02,
                f'{val:.1f}%', ha='center', va='bottom')
    
    ax2 = plt.subplot(132)
    if not family_results.empty:
        top_families = family_results.nsmallest(8, 'p_value')
        
        y_pos = np.arange(len(top_families))
        enrichments = top_families['enrichment'].values
        p_values = top_families['p_value'].values
        
        colors_fam = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_values]
        
        ax2.barh(y_pos, enrichments, color=colors_fam, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_families['tf_family'].values)
        ax2.set_xlabel('Enrichment Fold', fontweight='bold')
        ax2.set_title('TF Family Enrichment', fontweight='bold')
        ax2.axvline(1.0, color='black', linestyle='--', linewidth=1)
        
        for i, (enr, pval) in enumerate(zip(enrichments, p_values)):
            label = f"{enr:.2f}x" if pval < 0.05 else f"{enr:.2f}x (ns)"
            ax2.text(enr + 0.1, i, label, va='center')
    
    ax3 = plt.subplot(133)
    fg_median = dist_comparison['fg_median']
    bg_median = dist_comparison['bg_median']
    
    x = ['Foreground', 'Background']
    y = [fg_median/1000, bg_median/1000]
    bars = ax3.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Median distance to TF (kb)', fontweight='bold')
    ax3.set_title('Distance Distribution', fontweight='bold')
    
    p_text = f"p = {dist_comparison['mw_p_value']:.3f}" if dist_comparison['mw_p_value'] >= 0.001 else "p < 0.001"
    ax3.text(0.5, max(y) * 1.15, p_text, ha='center', fontweight='bold')
    
    for bar, val in zip(bars, y):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y)*0.02,
                f'{val:.1f}kb', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Figure 3: {output_path}")


def evaluate_gate_3(proximity_results, family_results):
    """Evaluate Decision Gate 3."""
    print("\n" + "="*60)
    print("DECISION GATE 3: TF BINDING ENRICHMENT")
    print("="*60)
    
    prox = proximity_results
    
    prox_pass = (prox['p_value'] < 0.01) and (prox['enrichment_fold'] > 1.3)
    
    sig_families = 0
    if not family_results.empty:
        sig_families = (family_results['p_value'] < 0.05).sum()
    
    family_pass = sig_families >= 1
    
    print(f"\nTF Proximity: {prox['enrichment_fold']:.2f}x, p={prox['p_value']:.3e} - {'PASS' if prox_pass else 'FAIL'}")
    print(f"Significant TF families (p<0.05): {sig_families} - {'PASS' if family_pass else 'FAIL'}")
    
    gate_pass = prox_pass or family_pass
    
    print(f"\n{'='*60}")
    print(f"GATE 3: {'PROCEED TO WEEK 4' if gate_pass else 'REEVALUATE'}")
    print(f"{'='*60}")
    
    return gate_pass


def main():
    """Main execution pipeline."""
    
    fg, bg = load_snp_sets()
    
    tf_sites = load_tf_binding_sites()
    
    fg_distances = calculate_distances_to_tf(fg, tf_sites)
    bg_distances = calculate_distances_to_tf(bg, tf_sites)
    
    fg_distances.to_csv(OUTPUT_DIR / "foreground_tf_distances.csv", index=False)
    bg_distances.to_csv(OUTPUT_DIR / "background_tf_distances.csv", index=False)
    
    proximity_results = test_proximity_enrichment(fg_distances, bg_distances)
    
    family_results = test_tf_family_enrichment(fg_distances, bg_distances)
    if not family_results.empty:
        family_results.to_csv(OUTPUT_DIR / "tf_family_enrichment.csv", index=False)
    
    dist_comparison = compare_distance_distributions(fg_distances, bg_distances)
    
    results_summary = {**proximity_results, **dist_comparison}
    pd.DataFrame([results_summary]).to_csv(OUTPUT_DIR / "tf_enrichment_summary.csv", index=False)
    
    plot_tf_proximity_results(proximity_results, family_results, dist_comparison,
                              FIGURES_DIR / "figure3_tf_proximity.png")
    
    gate_pass = evaluate_gate_3(proximity_results, family_results)
    
    pd.DataFrame([{
        'gate': 'Gate 3',
        'decision': 'PASS' if gate_pass else 'FAIL',
        'tf_proximity_enrich': proximity_results['enrichment_fold'],
        'tf_proximity_p': proximity_results['p_value'],
        'n_sig_families': (family_results['p_value'] < 0.05).sum() if not family_results.empty else 0
    }]).to_csv(OUTPUT_DIR / "gate3_decision.csv", index=False)
    
    print("\nWEEK 3 COMPLETE")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()