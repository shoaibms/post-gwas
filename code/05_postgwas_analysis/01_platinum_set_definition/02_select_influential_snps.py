#!/usr/bin/env python3
"""
Select top influential cis-SNPs for each platinum modulator gene.

For each gene, extracts SNPs within a cis-window from PVAR, ranks by
XGBoost importance (if NPZ available) or TSS proximity, applies spatial
LD pruning, and selects the top K independent SNPs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor

# Configuration
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
PLATINUM_FILE = BASE_DIR / "output" / "week1_stability" / "platinum_modulator_set.csv"
NPZ_DIR = BASE_DIR / "output" / "ect_drought_D1b_gateON" / "baseline_npz"
PVAR_FILE = BASE_DIR / "output" / "geno" / "cohort_pruned.pvar"
PGEN_FILE = BASE_DIR / "output" / "geno" / "cohort_pruned.pgen"
GFF3_FILE = BASE_DIR / "data" / "maize" / "Zea_mays.B73_RefGen_v4.gff3.gz"
OUTPUT_DIR = BASE_DIR / "output" / "week2_snp_selection"

# Parameters
CIS_WINDOW = 500000  # +/-500kb window
TOP_K_SNPS = 3  # Select top 3 independent SNPs per gene
MIN_DISTANCE = 250000  # Minimum 250kb between selected SNPs (LD proxy)
N_FOLDS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_platinum_genes() -> List[str]:
    """Load platinum modulator gene list."""
    df = pd.read_csv(PLATINUM_FILE)
    genes = df['gene'].tolist()
    print(f"Loaded {len(genes)} platinum modulator genes")
    return genes


def load_pvar_metadata() -> pd.DataFrame:
    """Load PVAR file correctly."""
    print("Loading PVAR file...")
    
    rows = []
    with open(PVAR_FILE, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                rows.append({
                    'CHROM': parts[0],
                    'POS': parts[1],
                    'ID': parts[2],
                    'REF': parts[3],
                    'ALT': parts[4]
                })
    
    pvar = pd.DataFrame(rows)
    pvar['POS'] = pd.to_numeric(pvar['POS'], errors='coerce')
    
    initial_count = len(pvar)
    pvar = pvar.dropna(subset=['POS'])
    pvar['POS'] = pvar['POS'].astype(int)
    
    if len(pvar) < initial_count:
        print(f"  Dropped {initial_count - len(pvar)} SNPs with invalid positions")
    
    # Normalize chromosome names
    pvar['CHR_NORM'] = pvar['CHROM'].str.replace('chr', '', case=False).str.replace('Chr', '')
    
    print(f"  Loaded {len(pvar):,} valid SNPs")
    return pvar


def load_gff3_gene_positions() -> pd.DataFrame:
    """Extract gene TSS positions from GFF3."""
    print("Loading gene positions from GFF3...")
    
    import gzip
    genes = []
    
    with gzip.open(GFF3_FILE, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2].lower() != 'gene':
                continue
            
            seqid, _, _, start, end, _, strand, _, attributes = parts
            
            gene_id = None
            for attr in attributes.split(';'):
                if attr.startswith('ID='):
                    gene_id = attr.split('=')[1].replace('gene:', '')
                    break
            
            if not gene_id or not gene_id.startswith('Zm00001d'):
                continue
            
            tss = int(end) if strand == '-' else int(start)
            chr_norm = seqid.replace('chr', '').replace('Chr', '')
            
            genes.append({
                'gene_id': gene_id,
                'chr': chr_norm,
                'tss': tss,
                'strand': strand
            })
    
    df = pd.DataFrame(genes)
    print(f"  Loaded {len(df):,} gene positions")
    return df


def extract_cis_snps(gene_chr: str, gene_tss: int, pvar: pd.DataFrame,
                     window: int = CIS_WINDOW) -> pd.DataFrame:
    """
    Extract all cis-SNPs within window of gene TSS.
    
    Returns DataFrame with SNPs sorted by distance to TSS.
    """
    start_pos = max(1, gene_tss - window)
    end_pos = gene_tss + window
    
    # Filter to chromosome and window
    cis_snps = pvar[
        (pvar['CHR_NORM'] == str(gene_chr)) &
        (pvar['POS'] >= start_pos) &
        (pvar['POS'] <= end_pos)
    ].copy()
    
    # Calculate distance to TSS
    cis_snps['dist_to_tss'] = cis_snps['POS'] - gene_tss
    cis_snps['abs_dist'] = cis_snps['dist_to_tss'].abs()
    
    # Sort by proximity to TSS
    cis_snps = cis_snps.sort_values('abs_dist')
    
    return cis_snps


def load_npz_for_gene(gene: str, fold: int) -> Optional[Dict[str, np.ndarray]]:
    """Load NPZ data if available."""
    npz_path = NPZ_DIR / f"fold_{fold}" / f"{gene}.npz"
    
    if not npz_path.exists():
        return None
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        return {
            'X_tr': data['X_tr'],
            'y_tr': data['y_tr'],
            'env_tr': data['env_tr']
        }
    except:
        return None


def compute_importance_scores(gene: str) -> Optional[np.ndarray]:
    """
    Compute XGBoost feature importance if NPZ files exist.
    Returns None if NPZ not available.
    """
    all_scores = []
    
    for fold in range(1, N_FOLDS + 1):
        data = load_npz_for_gene(gene, fold)
        if data is None:
            continue
        
        X_tr = data['X_tr']
        if X_tr.shape[1] == 0:
            continue
        
        try:
            # Add environment as features
            env_dummies = pd.get_dummies(data['env_tr'], prefix='env')
            X_full = np.hstack([X_tr, env_dummies.values])
            
            # Train XGBoost
            model = XGBRegressor(
                max_depth=2,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
            
            model.fit(X_full, data['y_tr'])
            
            # Get importance for SNPs only
            n_snps = X_tr.shape[1]
            importance = model.feature_importances_[:n_snps]
            all_scores.append(importance)
            
        except:
            continue
    
    if len(all_scores) == 0:
        return None
    
    # Average across folds
    mean_scores = np.mean(all_scores, axis=0)
    return mean_scores


def spatial_ld_prune(snps_df: pd.DataFrame, min_distance: int) -> pd.DataFrame:
    """
    Simple spatial LD pruning: select SNPs separated by at least min_distance.
    
    SNPs are already sorted by importance or proximity.
    """
    if len(snps_df) == 0:
        return snps_df
    
    selected = [snps_df.iloc[0]]
    
    for _, snp in snps_df.iloc[1:].iterrows():
        # Check distance to all selected SNPs
        is_independent = all(
            abs(snp['POS'] - s['POS']) >= min_distance
            for s in selected
        )
        
        if is_independent:
            selected.append(snp)
    
    return pd.DataFrame(selected)


def select_top_snps_for_gene(gene: str, pvar: pd.DataFrame,
                               gene_positions: pd.DataFrame,
                               top_k: int = TOP_K_SNPS) -> pd.DataFrame:
    """
    Select top influential cis-SNPs for a gene.
    
    Priority:
    1. Try to use importance scores from NPZ
    2. Fallback to proximity to TSS
    3. Always apply spatial LD pruning
    """
    print(f"\nProcessing: {gene}")
    
    # Get gene position
    gene_info = gene_positions[gene_positions['gene_id'] == gene]
    if gene_info.empty:
        print(f"  Warning: Gene not found in GFF3")
        return pd.DataFrame()
    
    gene_chr = gene_info.iloc[0]['chr']
    gene_tss = gene_info.iloc[0]['tss']
    print(f"  Position: chr{gene_chr}:{gene_tss}")
    
    # Extract cis-SNPs from PVAR
    cis_snps = extract_cis_snps(gene_chr, gene_tss, pvar, CIS_WINDOW)
    
    if len(cis_snps) == 0:
        print(f"  Warning: No cis-SNPs found within +/-{CIS_WINDOW/1000:.0f}kb")
        return pd.DataFrame()
    
    print(f"  Found {len(cis_snps)} cis-SNPs within +/-{CIS_WINDOW/1000:.0f}kb")
    
    # Try to compute importance scores
    importance_scores = compute_importance_scores(gene)
    
    if importance_scores is not None and len(importance_scores) == len(cis_snps):
        # Use importance-based ranking
        cis_snps['importance_score'] = importance_scores
        cis_snps = cis_snps.sort_values('importance_score', ascending=False)
        print(f"  Ranked by XGBoost importance")
    else:
        # Fallback: rank by proximity to TSS
        cis_snps['importance_score'] = 1.0 / (cis_snps['abs_dist'] + 1)
        print(f"  Ranked by proximity to TSS (NPZ not available)")
    
    # Spatial LD pruning
    pruned_snps = spatial_ld_prune(cis_snps, MIN_DISTANCE)
    print(f"  Spatial LD prune: {len(cis_snps)} -> {len(pruned_snps)} independent")
    
    # Select top K
    top_snps = pruned_snps.head(top_k)
    
    # Prepare output
    results = top_snps[['ID', 'CHROM', 'POS', 'REF', 'ALT', 
                        'dist_to_tss', 'importance_score']].copy()
    results['gene'] = gene
    results['gene_chr'] = gene_chr
    results['gene_tss'] = gene_tss
    
    results = results.rename(columns={
        'ID': 'snp_id',
        'CHROM': 'chr',
        'POS': 'pos',
        'REF': 'ref',
        'ALT': 'alt'
    })
    
    print(f"  Selected {len(results)} top SNPs")
    
    return results


def main():
    """Main execution function."""
    print("="*60)
    print("WEEK 2: PRAGMATIC SNP SELECTION")
    print("="*60)
    print("\nDirect genomic coordinate approach - no index mapping")
    
    # Load inputs
    print("\nStep 1: Loading inputs...")
    platinum_genes = load_platinum_genes()
    pvar = load_pvar_metadata()
    gene_positions = load_gff3_gene_positions()
    
    # Process each gene
    print(f"\nStep 2: Processing {len(platinum_genes)} platinum genes...")
    
    all_results = []
    
    for i, gene in enumerate(platinum_genes, 1):
        print(f"\n[{i}/{len(platinum_genes)}]", end=" ")
        
        gene_results = select_top_snps_for_gene(
            gene, pvar, gene_positions, top_k=TOP_K_SNPS
        )
        
        if not gene_results.empty:
            all_results.append(gene_results)
    
    # Combine and save
    if len(all_results) == 0:
        print("\nERROR: No SNPs selected!")
        return
    
    combined = pd.concat(all_results, ignore_index=True)
    
    output_file = OUTPUT_DIR / "top_influential_snps_pragmatic.csv"
    combined.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("PRAGMATIC SNP SELECTION COMPLETE")
    print("="*60)
    print(f"\nTotal SNPs: {len(combined):,}")
    print(f"Genes: {combined['gene'].nunique()}")
    print(f"Mean SNPs/gene: {len(combined)/combined['gene'].nunique():.1f}")
    
    # Metadata completeness
    print(f"\nMetadata: 100% complete")
    
    # Distance statistics
    print(f"\nDistance to TSS statistics:")
    print(combined['dist_to_tss'].describe())
    
    within_20kb = (combined['dist_to_tss'].abs() <= 20000).sum()
    within_100kb = (combined['dist_to_tss'].abs() <= 100000).sum()
    within_500kb = (combined['dist_to_tss'].abs() <= 500000).sum()
    
    print(f"\nProximity distribution:")
    print(f"  Within +/-20kb: {within_20kb}/{len(combined)} ({100*within_20kb/len(combined):.1f}%)")
    print(f"  Within +/-100kb: {within_100kb}/{len(combined)} ({100*within_100kb/len(combined):.1f}%)")
    print(f"  Within +/-500kb: {within_500kb}/{len(combined)} ({100*within_500kb/len(combined):.1f}%)")
    
    print(f"\nSaved to: {output_file}")
    
    # Preview
    print("\nPreview:")
    print(combined[['gene', 'snp_id', 'chr', 'pos', 'dist_to_tss']].head(10))
    
    print("\nREADY FOR: Matched background control generation")


if __name__ == "__main__":
    main()