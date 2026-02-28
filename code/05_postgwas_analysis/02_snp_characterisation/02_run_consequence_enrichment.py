#!/usr/bin/env python3
"""
Consequence enrichment analysis for foreground vs background SNPs.

Performs vectorized annotation of SNPs with GFF3 genomic features,
tests TSS proximity enrichment, regulatory consequence enrichment,
and combined enrichment using Fisher's exact test.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact, norm
import matplotlib.pyplot as plt
import gzip
from typing import Dict, Tuple, List
import os
import time
from joblib import Parallel, delayed

# Configuration
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
FOREGROUND_FILE = BASE_DIR / "output" / "week2_snp_selection" / "top_influential_snps_pragmatic.csv"
BACKGROUND_FILE = BASE_DIR / "output" / "week2_enrichment" / "background_matched_10x.csv"
GFF3_FILE = BASE_DIR / "data" / "maize" / "Zea_mays.B73_RefGen_v4.gff3.gz"
OUTPUT_DIR = BASE_DIR / "output" / "week2_enrichment"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Multi-core configuration
# Use more cores on 40C machine but leave a couple free for OS/BLAS threads
N_JOBS = max(1, min(os.cpu_count() - 2, 36))

# Consequence hierarchy
CONSEQUENCE_HIERARCHY = {
    "5_prime_UTR": "regulatory",
    "3_prime_UTR": "regulatory", 
    "upstream": "regulatory",
    "downstream": "regulatory",
    "exon": "coding",
    "CDS": "coding",
    "intron": "intronic",
    "intergenic": "intergenic"
}


def load_snp_sets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load foreground and background SNP sets."""
    print("Loading SNP sets...")
    
    fg = pd.read_csv(FOREGROUND_FILE)
    bg = pd.read_csv(BACKGROUND_FILE)
    
    # Standardize columns
    fg = fg.rename(columns={'snp_id': 'snp', 'chr': 'chr', 'pos': 'pos', 'dist_to_tss': 'dist_to_tss'})
    bg = bg.rename(columns={'bg_snp_id': 'snp', 'bg_chr': 'chr', 'bg_pos': 'pos', 'bg_abs_dist_to_tss': 'dist_to_tss'})
    
    fg['set'] = 'foreground'
    bg['set'] = 'background'
    
    fg['chr'] = fg['chr'].astype(str).str.replace('chr', '', case=False).str.replace('Chr', '')
    bg['chr'] = bg['chr'].astype(str).str.replace('chr', '', case=False).str.replace('Chr', '')
    fg['pos'] = pd.to_numeric(fg['pos'], errors='coerce')
    bg['pos'] = pd.to_numeric(bg['pos'], errors='coerce')
    
    print(f"  Foreground: {len(fg)} SNPs")
    print(f"  Background: {len(bg)} SNPs")
    
    return fg, bg


def normalize_feature_type(feat_type: str) -> str:
    """Normalize feature type to standard names."""
    feat_lower = feat_type.lower()
    
    # Gene
    if feat_lower == 'gene':
        return 'gene'
    
    # Exon/CDS
    if feat_lower in ['exon', 'cds', 'coding_sequence']:
        return 'exon'
    
    # UTRs - accept many variants
    if any(x in feat_lower for x in ['five_prime_utr', '5_prime_utr', '5utr', 'five_utr']):
        return '5_prime_UTR'
    
    if any(x in feat_lower for x in ['three_prime_utr', '3_prime_utr', '3utr', 'three_utr']):
        return '3_prime_UTR'
    
    # Transcript/mRNA (for hierarchy only)
    if feat_lower in ['mrna', 'transcript']:
        return 'transcript'
    
    return feat_type


def odds_ratio_and_ci_from_contingency(cont: np.ndarray, alpha: float = 0.05):
    """
    cont = [[a, b],
            [c, d]]
    Returns: (or_val, ci_low, ci_high)
    Applies 0.5 continuity correction if any cell is zero.
    """
    a, b, c, d = (float(x) for x in cont.ravel())
    if min(a, b, c, d) == 0.0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
    z = norm.ppf(1 - alpha/2.0)
    lo = np.exp(np.log(or_val) - z * se)
    hi = np.exp(np.log(or_val) + z * se)
    return float(or_val), float(lo), float(hi)


def risk_ratio_and_ci_from_2x2(cont: np.ndarray, alpha: float = 0.05):
    """
    Katz log method CI for the risk/enrichment ratio:
      RR = (a/(a+b)) / (c/(c+d))
    Applies 0.5 continuity correction if any cell is zero.
    Returns: (rr, ci_low, ci_high)
    """
    a, b, c, d = (float(x) for x in cont.ravel())
    if min(a, b, c, d) == 0.0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    n1 = a + b
    n2 = c + d
    p1 = a / n1
    p2 = c / n2
    rr = (p1 / p2) if p2 > 0 else np.inf
    z = norm.ppf(1 - alpha/2.0)
    # SE[ln(RR)] = sqrt(1/a - 1/n1 + 1/c - 1/n2)
    se = np.sqrt((1.0/a) - (1.0/n1) + (1.0/c) - (1.0/n2))
    lo = np.exp(np.log(rr) - z * se)
    hi = np.exp(np.log(rr) + z * se)
    return float(rr), float(lo), float(hi)


def _tick(label, t0=[None]):
    """Track timing between phases."""
    if t0[0] is None:
        t0[0] = time.time()
        return
    dt = time.time() - t0[0]
    print(f"[TIMER] {label} in {dt/60:.1f} min")
    t0[0] = None


def _interval_index(df, feature_type):
    """Build interval index for fast overlap queries."""
    sub = df[df['feature_type'] == feature_type]
    if sub.empty:
        return None, None
    iv = pd.IntervalIndex.from_arrays(sub['start'].to_numpy(np.int64),
                                      sub['end'].to_numpy(np.int64),
                                      closed='both')
    genes = sub['gene_id'].to_numpy(object)
    return iv, genes


def _assign_hits(snps_chr, iv, genes, label, only_if='intergenic'):
    """Assign consequence to SNPs overlapping intervals (handles overlaps)."""
    if iv is None:
        return snps_chr
    pos = snps_chr['pos'].to_numpy(np.int64)
    
    # Handle overlapping intervals - take first match per position
    idx = np.full(len(pos), -1, dtype=np.int64)
    for i, p in enumerate(pos):
        matches = iv.contains(p)
        if matches.any():
            idx[i] = np.flatnonzero(matches)[0]
    
    cur = snps_chr['consequence'].to_numpy(object)
    mask = (idx >= 0) & ((cur == only_if) if only_if else np.ones(len(cur), bool))
    if not np.any(mask):
        return snps_chr
    snps_chr.loc[snps_chr.index[mask], 'consequence'] = label
    snps_chr.loc[snps_chr.index[mask], 'nearest_gene'] = genes[idx[mask]]
    return snps_chr


def _annotate_chr(snps_chr: pd.DataFrame, feats_chr: pd.DataFrame) -> pd.DataFrame:
    """Annotate SNPs on a single chromosome using interval indices."""
    if snps_chr.empty or feats_chr.empty:
        return snps_chr

    snps_chr = snps_chr.copy()
    snps_chr['consequence'] = 'intergenic'
    snps_chr['nearest_gene'] = None

    iv_5utr, g_5utr = _interval_index(feats_chr, '5_prime_UTR')
    iv_3utr, g_3utr = _interval_index(feats_chr, '3_prime_UTR')
    iv_exon, g_exon = _interval_index(feats_chr, 'exon')
    iv_gene, g_gene = _interval_index(feats_chr, 'gene')

    genes = feats_chr[feats_chr['feature_type'] == 'gene']
    if not genes.empty:
        start = genes['start'].to_numpy(np.int64)
        end   = genes['end'].to_numpy(np.int64)
        strand= genes['strand'].to_numpy(object)
        gid   = genes['gene_id'].to_numpy(object)

        tss = np.where(strand == '+', start, end)
        tes = np.where(strand == '+', end,   start)

        prom_start = np.where(strand == '+', tss - 2000, tss)
        prom_end   = np.where(strand == '+', tss,        tss + 2000)
        iv_prom = pd.IntervalIndex.from_arrays(prom_start, prom_end, closed='left')
        iv_down = pd.IntervalIndex.from_arrays(
            np.where(strand == '+', tes, tes - 500),
            np.where(strand == '+', tes + 500, tes),
            closed='left'
        )
        g_prom = gid
        g_down = gid
    else:
        iv_prom = iv_down = g_prom = g_down = None

    snps_chr = _assign_hits(snps_chr, iv_5utr, g_5utr, '5_prime_UTR', 'intergenic')
    snps_chr = _assign_hits(snps_chr, iv_3utr, g_3utr, '3_prime_UTR', 'intergenic')
    snps_chr = _assign_hits(snps_chr, iv_exon, g_exon, 'exon', 'intergenic')
    snps_chr = _assign_hits(snps_chr, iv_gene, g_gene, 'intron', 'intergenic')
    snps_chr = _assign_hits(snps_chr, iv_prom, g_prom, 'upstream', 'intergenic')
    snps_chr = _assign_hits(snps_chr, iv_down, g_down, 'downstream', 'intergenic')

    return snps_chr


def load_gene_annotations() -> pd.DataFrame:
    """Load gene features from GFF3 with robust parsing."""
    print("\nLoading gene annotations from GFF3...")
    
    features = []
    line_count = 0
    
    with gzip.open(GFF3_FILE, 'rt') as f:
        for line in f:
            line_count += 1
            
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            seqid, source, feature_type, start, end, score, strand, phase, attributes = parts
            
            # Normalize feature type
            norm_type = normalize_feature_type(feature_type)
            
            # Only keep relevant features
            if norm_type not in ['gene', 'exon', '5_prime_UTR', '3_prime_UTR', 'transcript']:
                continue
            
            # Extract gene/parent ID
            gene_id = None
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('ID='):
                    gene_id = attr.split('=')[1].replace('gene:', '').replace('transcript:', '')
                elif attr.startswith('Parent='):
                    gene_id = attr.split('=')[1].replace('gene:', '').replace('transcript:', '')
                
                if gene_id:
                    break
            
            # Normalize chromosome
            chr_norm = seqid.replace('chr', '').replace('Chr', '')
            
            # Only keep main chromosomes
            if chr_norm not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                continue
            
            features.append({
                'chr': chr_norm,
                'start': int(start),
                'end': int(end),
                'feature_type': norm_type,
                'strand': strand,
                'gene_id': gene_id
            })
    
    df = pd.DataFrame(features)
    
    if df.empty:
        raise RuntimeError(f"ERROR: No features parsed from GFF3 after {line_count} lines!")
    
    print(f"  Parsed {line_count:,} GFF3 lines")
    print(f"  Extracted {len(df):,} relevant features")
    
    # Feature inventory
    print("\n  Feature type inventory:")
    for feat, count in df['feature_type'].value_counts().items():
        print(f"    {feat}: {count:,}")
    
    # Chromosome inventory
    print("\n  Chromosome distribution:")
    for chr_id, count in sorted(df['chr'].value_counts().items()):
        print(f"    chr{chr_id}: {count:,} features")
    
    # Critical checks
    n_genes = (df['feature_type'] == 'gene').sum()
    if n_genes == 0:
        raise RuntimeError("ERROR: Zero 'gene' features found! Check GFF3 format.")
    
    print(f"\n  QC: {n_genes:,} genes found - OK to proceed")
    
    return df


def vectorized_overlap_annotation(snps: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Parallel interval-indexed annotation (per chromosome)."""
    print(f"\nAnnotating SNPs with parallel interval index (n_jobs={N_JOBS})...")
    by_chr_feats = dict(tuple(features.groupby('chr', sort=False)))
    tasks = []
    for chr_id, snps_chr in snps.groupby('chr', sort=False):
        feats_chr = by_chr_feats.get(chr_id, pd.DataFrame(columns=features.columns))
        print(f"  Queue chr{chr_id}: {len(snps_chr)} SNPs, {len(feats_chr)} features")
        tasks.append((chr_id, snps_chr, feats_chr))

    def _run(task):
        _, snps_chr, feats_chr = task
        return _annotate_chr(snps_chr, feats_chr)

    annotated_parts = Parallel(n_jobs=N_JOBS, backend='loky', prefer='processes')(
        delayed(_run)(t) for t in tasks
    )
    snps_out = pd.concat(annotated_parts, axis=0).sort_index()

    snps_out['consequence_category'] = snps_out['consequence'].map(CONSEQUENCE_HIERARCHY)
    snps_out['consequence_category'] = snps_out['consequence_category'].fillna('intergenic')

    print("\n  Annotation summary:")
    for cons, count in snps_out['consequence'].value_counts().items():
        pct = 100 * count / len(snps_out)
        print(f"    {cons}: {count} ({pct:.1f}%)")

    n_intergenic = (snps_out['consequence'] == 'intergenic').sum()
    if n_intergenic == len(snps_out):
        raise RuntimeError("ERROR: All SNPs annotated as intergenic! Feature overlap failed.")
    return snps_out


def assign_proximity_bins(snps: pd.DataFrame) -> pd.DataFrame:
    """Assign proximity bins."""
    snps = snps.copy()
    abs_dist = snps['dist_to_tss'].abs()
    
    snps['proximity_bin'] = 'distal_100-500kb'
    snps.loc[abs_dist < 20000, 'proximity_bin'] = 'proximal_0-20kb'
    snps.loc[(abs_dist >= 20000) & (abs_dist < 100000), 'proximity_bin'] = 'intermediate_20-100kb'
    
    return snps


def test_proximity_enrichment(fg: pd.DataFrame, bg: pd.DataFrame) -> Dict:
    """Test TSS proximity enrichment."""
    print("\n" + "="*60)
    print("TEST 1: TSS PROXIMITY ENRICHMENT")
    print("="*60)
    
    fg_proximal = (fg['proximity_bin'] == 'proximal_0-20kb').sum()
    fg_distal = (fg['proximity_bin'] != 'proximal_0-20kb').sum()
    bg_proximal = (bg['proximity_bin'] == 'proximal_0-20kb').sum()
    bg_distal = (bg['proximity_bin'] != 'proximal_0-20kb').sum()
    
    contingency = np.array([[fg_proximal, fg_distal],
                            [bg_proximal, bg_distal]])
    odds_ratio, p_value = fisher_exact(contingency, alternative='greater')
    or_ci, or_lo, or_hi = odds_ratio_and_ci_from_contingency(contingency)
    
    fg_prox_pct = 100 * fg_proximal / len(fg)
    bg_prox_pct = 100 * bg_proximal / len(bg)
    enrichment = fg_prox_pct / bg_prox_pct if bg_prox_pct > 0 else 1.0
    
    rr, rr_lo, rr_hi = risk_ratio_and_ci_from_2x2(contingency)
    print(f"\nForeground proximal: {fg_proximal}/{len(fg)} ({fg_prox_pct:.1f}%)")
    print(f"Background proximal: {bg_proximal}/{len(bg)} ({bg_prox_pct:.1f}%)")
    print(f"Enrichment: {enrichment:.2f}x  |  RR={rr:.2f} [95% CI {rr_lo:.2f}, {rr_hi:.2f}]")
    print(f"Fisher's exact p: {p_value:.3e}")
    print(f"Proximity OR={odds_ratio:.2f} [95% CI {or_lo:.2f}, {or_hi:.2f}]")
    
    return {
        'proximity': {
            'fg_proximal': int(fg_proximal),
            'fg_distal': int(fg_distal),
            'bg_proximal': int(bg_proximal),
            'bg_distal': int(bg_distal),
            'fg_proximal_pct': fg_prox_pct,
            'bg_proximal_pct': bg_prox_pct,
            'enrichment_fold': enrichment,
            'enrichment_ci_lower': float(rr_lo),
            'enrichment_ci_upper': float(rr_hi),
            'odds_ratio': float(odds_ratio),
            'ci_lower': float(or_lo),
            'ci_upper': float(or_hi),
            'p_value': float(p_value)
        }
    }


def test_consequence_enrichment(fg: pd.DataFrame, bg: pd.DataFrame) -> Dict:
    """Test regulatory consequence enrichment."""
    print("\n" + "="*60)
    print("TEST 2: REGULATORY CONSEQUENCE ENRICHMENT")
    print("="*60)
    
    fg_reg = (fg['consequence_category'] == 'regulatory').sum()
    fg_other = (fg['consequence_category'] != 'regulatory').sum()
    bg_reg = (bg['consequence_category'] == 'regulatory').sum()
    bg_other = (bg['consequence_category'] != 'regulatory').sum()
    
    contingency = np.array([[fg_reg, fg_other],
                            [bg_reg, bg_other]])
    odds_ratio, p_value = fisher_exact(contingency, alternative='greater')
    or_ci, or_lo, or_hi = odds_ratio_and_ci_from_contingency(contingency)
    
    fg_reg_pct = 100 * fg_reg / len(fg)
    bg_reg_pct = 100 * bg_reg / len(bg)
    enrichment = fg_reg_pct / bg_reg_pct if bg_reg_pct > 0 else 1.0
    
    rr, rr_lo, rr_hi = risk_ratio_and_ci_from_2x2(contingency)
    print(f"\nForeground regulatory: {fg_reg}/{len(fg)} ({fg_reg_pct:.1f}%)")
    print(f"Background regulatory: {bg_reg}/{len(bg)} ({bg_reg_pct:.1f}%)")
    print(f"Enrichment: {enrichment:.2f}x  |  RR={rr:.2f} [95% CI {rr_lo:.2f}, {rr_hi:.2f}]")
    print(f"Fisher's exact p: {p_value:.3e}")
    print(f"Regulatory OR={odds_ratio:.2f} [95% CI {or_lo:.2f}, {or_hi:.2f}]")
    
    return {
        'regulatory': {
            'fg_regulatory': int(fg_reg),
            'fg_other': int(fg_other),
            'bg_regulatory': int(bg_reg),
            'bg_other': int(bg_other),
            'fg_regulatory_pct': fg_reg_pct,
            'bg_regulatory_pct': bg_reg_pct,
            'enrichment_fold': enrichment,
            'enrichment_ci_lower': float(rr_lo),
            'enrichment_ci_upper': float(rr_hi),
            'odds_ratio': float(odds_ratio),
            'ci_lower': float(or_lo),
            'ci_upper': float(or_hi),
            'p_value': float(p_value)
        }
    }


def test_combined_enrichment(fg: pd.DataFrame, bg: pd.DataFrame) -> Dict:
    """Test combined proximal + regulatory enrichment."""
    print("\n" + "="*60)
    print("TEST 3: COMBINED (PROXIMAL + REGULATORY)")
    print("="*60)
    
    fg_both = ((fg['proximity_bin'] == 'proximal_0-20kb') & 
               (fg['consequence_category'] == 'regulatory')).sum()
    fg_other = len(fg) - fg_both
    
    bg_both = ((bg['proximity_bin'] == 'proximal_0-20kb') & 
               (bg['consequence_category'] == 'regulatory')).sum()
    bg_other = len(bg) - bg_both
    
    contingency = np.array([[fg_both, fg_other],
                            [bg_both, bg_other]])
    odds_ratio, p_value = fisher_exact(contingency, alternative='greater')
    or_ci, or_lo, or_hi = odds_ratio_and_ci_from_contingency(contingency)
    
    fg_both_pct = 100 * fg_both / len(fg)
    bg_both_pct = 100 * bg_both / len(bg)
    enrichment = fg_both_pct / bg_both_pct if bg_both_pct > 0 else 1.0
    
    rr, rr_lo, rr_hi = risk_ratio_and_ci_from_2x2(contingency)
    print(f"\nForeground proximal+regulatory: {fg_both}/{len(fg)} ({fg_both_pct:.1f}%)")
    print(f"Background proximal+regulatory: {bg_both}/{len(bg)} ({bg_both_pct:.1f}%)")
    print(f"Enrichment: {enrichment:.2f}x  |  RR={rr:.2f} [95% CI {rr_lo:.2f}, {rr_hi:.2f}]")
    print(f"Fisher's exact p: {p_value:.3e}")
    print(f"Combined OR={odds_ratio:.2f} [95% CI {or_lo:.2f}, {or_hi:.2f}]")
    
    return {
        'combined': {
            'fg_prox_reg': int(fg_both),
            'fg_other': int(fg_other),
            'bg_prox_reg': int(bg_both),
            'bg_other': int(bg_other),
            'fg_prox_reg_pct': fg_both_pct,
            'bg_prox_reg_pct': bg_both_pct,
            'enrichment_fold': enrichment,
            'enrichment_ci_lower': float(rr_lo),
            'enrichment_ci_upper': float(rr_hi),
            'odds_ratio': float(odds_ratio),
            'ci_lower': float(or_lo),
            'ci_upper': float(or_hi),
            'p_value': float(p_value)
        }
    }


def plot_enrichment(results: Dict, output_path: Path):
    """Generate Figure 2."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#e74c3c', '#95a5a6']
    x = ['Foreground', 'Background']
    
    # Test 1
    ax = axes[0]
    prox = results['proximity']
    y = [prox['fg_proximal_pct'], prox['bg_proximal_pct']]
    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('% within 20kb of TSS', fontweight='bold')
    ax.set_title('Proximity Enrichment', fontweight='bold')
    ax.set_ylim(0, max(y) * 1.3)
    
    p_text = f"p = {prox['p_value']:.3f}" if prox['p_value'] >= 0.001 else f"p < 0.001"
    ax.text(0.5, max(y) * 1.15, p_text, ha='center', fontweight='bold')
    
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y)*0.02,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # Test 2
    ax = axes[1]
    reg = results['regulatory']
    y = [reg['fg_regulatory_pct'], reg['bg_regulatory_pct']]
    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('% regulatory', fontweight='bold')
    ax.set_title('Regulatory Enrichment', fontweight='bold')
    ax.set_ylim(0, max(max(y) * 1.3, 5))
    
    p_text = f"p = {reg['p_value']:.3f}" if reg['p_value'] >= 0.001 else f"p < 0.001"
    ax.text(0.5, max(max(y) * 1.15, 2.5), p_text, ha='center', fontweight='bold')
    
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height() + max(y)*0.02, 0.2),
                f'{val:.1f}%', ha='center', va='bottom')
    
    # Test 3
    ax = axes[2]
    comb = results['combined']
    y = [comb['fg_prox_reg_pct'], comb['bg_prox_reg_pct']]
    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('% proximal + regulatory', fontweight='bold')
    ax.set_title('Combined Enrichment', fontweight='bold')
    ax.set_ylim(0, max(max(y) * 1.3, 5))
    
    p_text = f"p = {comb['p_value']:.3f}" if comb['p_value'] >= 0.001 else f"p < 0.001"
    ax.text(0.5, max(max(y) * 1.15, 2.5), p_text, ha='center', fontweight='bold')
    
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height() + max(y)*0.02, 0.2),
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Figure 2: {output_path}")


def evaluate_gate_2(results: Dict) -> bool:
    """Evaluate Decision Gate 2."""
    print("\n" + "="*60)
    print("DECISION GATE 2: REGULATORY ENRICHMENT")
    print("="*60)
    
    prox = results['proximity']
    reg = results['regulatory']
    
    prox_pass = (prox['p_value'] < 0.05) and (prox['enrichment_fold'] > 1.2)
    reg_pass = (reg['p_value'] < 0.05) and (reg['enrichment_fold'] > 1.5)
    
    print(f"\nProximity: {prox['enrichment_fold']:.2f}x, p={prox['p_value']:.3e} - {'PASS' if prox_pass else 'FAIL'}")
    print(f"Regulatory: {reg['enrichment_fold']:.2f}x, p={reg['p_value']:.3e} - {'PASS' if reg_pass else 'FAIL'}")
    
    gate_pass = prox_pass or reg_pass
    
    print(f"\n{'='*60}")
    print(f"GATE 2: {'PROCEED TO WEEK 3' if gate_pass else 'REEVALUATE'}")
    print(f"{'='*60}")
    
    return gate_pass


def main():
    """Main execution."""
    print("="*60)
    print("WEEK 2: CONSEQUENCE ENRICHMENT")
    print("="*60)
    
    fg, bg = load_snp_sets()
    features = load_gene_annotations()
    
    print("\nAnnotating foreground...")
    _tick("start")
    fg_ann = vectorized_overlap_annotation(fg, features)
    fg_ann = assign_proximity_bins(fg_ann)
    _tick("foreground")
    
    print("\nAnnotating background...")
    _tick("start")
    bg_ann = vectorized_overlap_annotation(bg, features)
    bg_ann = assign_proximity_bins(bg_ann)
    _tick("background")
    
    fg_ann.to_csv(OUTPUT_DIR / "foreground_annotated.csv", index=False)
    bg_ann.to_csv(OUTPUT_DIR / "background_annotated.csv", index=False)
    
    results = {}
    results.update(test_proximity_enrichment(fg_ann, bg_ann))
    results.update(test_consequence_enrichment(fg_ann, bg_ann))
    results.update(test_combined_enrichment(fg_ann, bg_ann))
    
    pd.DataFrame([results]).to_csv(OUTPUT_DIR / "enrichment_results.csv", index=False)
    
    plot_enrichment(results, FIGURES_DIR / "figure2_enrichment.png")
    
    gate_pass = evaluate_gate_2(results)
    
    pd.DataFrame([{
        'gate': 'Gate 2',
        'decision': 'PASS' if gate_pass else 'FAIL',
        'prox_enrich': results['proximity']['enrichment_fold'],
        'prox_p': results['proximity']['p_value'],
        'reg_enrich': results['regulatory']['enrichment_fold'],
        'reg_p': results['regulatory']['p_value']
    }]).to_csv(OUTPUT_DIR / "gate2_decision.csv", index=False)
    
    print("\nWEEK 2 COMPLETE")


if __name__ == "__main__":
    main()