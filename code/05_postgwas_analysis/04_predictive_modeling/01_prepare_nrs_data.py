"""
Data preparation for network resilience score (NRS) analysis.

Handles transposed expression files (rows=samples, columns=genes),
extracts genotypes from VCF, and aligns samples across data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import gzip

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "output" / "week5_nrs"
DATA_DIR = BASE_DIR / "data" / "maize"
GENO_DIR = BASE_DIR / "output" / "geno"
FILTERED_DIR = BASE_DIR / "output" / "data_filtered"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_snp_extract_list():
    """Create list of SNPs to extract from genotype data."""
    print("Creating SNP extraction list...")
    
    snp_file = BASE_DIR / "output" / "week2_snp_selection" / "top_influential_snps_pragmatic.csv"
    snps = pd.read_csv(snp_file)
    
    snp_ids = snps['chr'].astype(str) + '_' + snps['pos'].astype(str)
    
    extract_file = OUTPUT_DIR / "snps_to_extract.txt"
    with open(extract_file, 'w') as f:
        for snp_id in snp_ids:
            f.write(f"{snp_id}\n")
    
    print(f"  Created extraction list: {len(snp_ids)} SNPs")
    print(f"  Saved to: {extract_file}")
    
    return extract_file, snps


def extract_genotypes_vcf(snp_info):
    """Extract genotypes directly from VCF."""
    print("\nExtracting genotypes from VCF...")
    
    vcf_path = DATA_DIR / "zea_mays_miss0.6_maf0.05.recode.vcf.gz"
    
    snp_positions = set(zip(snp_info['chr'].astype(str), snp_info['pos'].astype(int)))
    
    genotypes = []
    samples = None
    snp_ids = []
    
    print("  Parsing VCF (this may take several minutes)...")
    
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('##'):
                continue
            
            if line.startswith('#CHROM'):
                samples = line.strip().split('\t')[9:]
                print(f"  Found {len(samples)} samples in VCF")
                continue
            
            fields = line.strip().split('\t')
            chrom = fields[0].replace('chr', '')
            pos = int(fields[1])
            
            if (chrom, pos) in snp_positions:
                snp_id = f"{chrom}_{pos}"
                genotype_calls = fields[9:]
                
                gt_values = []
                for call in genotype_calls:
                    gt = call.split(':')[0]
                    if gt == './.':
                        gt_values.append(np.nan)
                    elif gt == '0/0':
                        gt_values.append(0)
                    elif gt == '0/1' or gt == '1/0':
                        gt_values.append(1)
                    elif gt == '1/1':
                        gt_values.append(2)
                    else:
                        gt_values.append(np.nan)
                
                genotypes.append(gt_values)
                snp_ids.append(snp_id)
                
                if len(snp_ids) % 10 == 0:
                    print(f"  Extracted {len(snp_ids)} SNPs...", end='\r')
    
    print(f"\n  Extracted {len(snp_ids)} SNPs total")
    
    genotype_df = pd.DataFrame(
        np.array(genotypes).T,
        index=samples,
        columns=snp_ids
    )
    
    missing_rate = genotype_df.isna().sum().sum() / genotype_df.size * 100
    print(f"  Missing rate: {missing_rate:.2f}%")
    
    genotype_df = genotype_df.fillna(genotype_df.mean())
    
    return genotype_df, samples


def load_expression_data_transposed():
    """
    Load expression matrices - CORRECTLY handling transposed format.
    
    Files have:
    - Rows = Samples (CML134, etc.)
    - Columns = Genes (Zm00001d######)
    
    We need to TRANSPOSE so:
    - Rows = Genes
    - Columns = Samples
    """
    print("\nLoading expression data (transposed format)...")
    
    expr_files = {
        'WW': FILTERED_DIR / "WW_209-Uniq_FPKM.agpv4.txt.gz",
        'WS1': FILTERED_DIR / "WS1_208-uniq_FPKM.agpv4.txt.gz",
        'WS2': FILTERED_DIR / "WS2_210-uniq_FPKM.agpv4.txt.gz"
    }
    
    expr_data = {}
    for cond, fpath in expr_files.items():
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found")
            continue
        
        print(f"  Loading {cond}...")
        df = pd.read_csv(fpath, sep='\t', compression='gzip', index_col=0)
        
        print(f"    Before transpose: {df.shape[0]} rows x {df.shape[1]} columns")
        
        df_transposed = df.T
        
        print(f"    After transpose: {df_transposed.shape[0]} rows (genes) x {df_transposed.shape[1]} columns (samples)")
        print(f"    Gene IDs (first 3): {df_transposed.index.tolist()[:3]}")
        print(f"    Sample IDs (first 3): {df_transposed.columns.tolist()[:3]}")
        
        expr_data[cond] = df_transposed
    
    return expr_data


def compute_phenotypes(expr_data):
    """Compute drought response phenotype."""
    print("\nComputing phenotypes...")
    
    if 'WW' not in expr_data or 'WS2' not in expr_data:
        print("  ERROR: Missing required expression data (WW or WS2)")
        return None, None, None
    
    platinum_file = BASE_DIR / "output" / "week1_stability" / "platinum_modulator_set.csv"
    platinum_genes = pd.read_csv(platinum_file)['gene'].tolist()
    
    print(f"  Using {len(platinum_genes)} platinum modulator genes")
    
    ww_expr = expr_data['WW']
    ws2_expr = expr_data['WS2']
    
    print(f"  WW: {ww_expr.shape[0]} genes x {ww_expr.shape[1]} samples")
    print(f"  WS2: {ws2_expr.shape[0]} genes x {ws2_expr.shape[1]} samples")
    
    common_samples = list(set(ww_expr.columns) & set(ws2_expr.columns))
    common_genes = list(set(platinum_genes) & set(ww_expr.index) & set(ws2_expr.index))
    
    print(f"  Common samples across WW and WS2: {len(common_samples)}")
    print(f"  Available platinum genes: {len(common_genes)}/{len(platinum_genes)}")
    
    if len(common_genes) == 0:
        print("  ERROR: No platinum genes found in expression data!")
        return None, None, None
    
    if len(common_samples) == 0:
        print("  ERROR: No common samples between WW and WS2!")
        return None, None, None
    
    ww_sub = ww_expr.loc[common_genes, common_samples]
    ws2_sub = ws2_expr.loc[common_genes, common_samples]
    
    delta_expr = ws2_sub - ww_sub
    
    phenotype = delta_expr.mean(axis=0)
    phenotype.name = 'drought_response'
    
    print(f"  Phenotype computed:")
    print(f"    Mean: {phenotype.mean():.3f}")
    print(f"    Std: {phenotype.std():.3f}")
    print(f"    Min: {phenotype.min():.3f}")
    print(f"    Max: {phenotype.max():.3f}")
    
    return phenotype, common_samples, common_genes


def align_and_save_data(genotype_df, phenotype, snp_info, vcf_samples):
    """Align genotype and phenotype data and save."""
    print("\nAligning data...")
    
    print(f"  Genotype samples (VCF): {len(genotype_df.index)}")
    print(f"    First 5: {genotype_df.index.tolist()[:5]}")
    
    print(f"  Phenotype samples (Expression): {len(phenotype.index)}")
    print(f"    First 5: {phenotype.index.tolist()[:5]}")
    
    common_samples = list(set(genotype_df.index) & set(phenotype.index))
    print(f"  Samples with both genotype and phenotype: {len(common_samples)}")
    
    if len(common_samples) < 50:
        print("  WARNING: Very few samples with complete data!")
        print("  Checking for potential name variations...")
        
        geno_set = set(genotype_df.index)
        pheno_set = set(phenotype.index)
        
        geno_only = list(geno_set - pheno_set)[:5]
        pheno_only = list(pheno_set - geno_set)[:5]
        
        print(f"  Genotype-only (first 5): {geno_only}")
        print(f"  Phenotype-only (first 5): {pheno_only}")
    
    if len(common_samples) == 0:
        print("\nERROR: No overlapping samples!")
        print("Cannot proceed with analysis.")
        return None
    
    geno_aligned = genotype_df.loc[common_samples]
    pheno_aligned = phenotype.loc[common_samples]
    
    geno_aligned.to_csv(OUTPUT_DIR / "genotype_matrix.csv")
    pheno_aligned.to_csv(OUTPUT_DIR / "phenotype_vector.csv")
    
    print(f"\n  Saved genotype matrix: {geno_aligned.shape}")
    print(f"  Saved phenotype vector: {len(pheno_aligned)}")
    
    platinum_genes = pd.read_csv(
        BASE_DIR / "output" / "week1_stability" / "platinum_modulator_set.csv"
    )['gene'].tolist()
    
    snp_info['is_modulator'] = snp_info['gene'].isin(platinum_genes)
    snp_info.to_csv(OUTPUT_DIR / "snp_annotations.csv", index=False)
    
    print(f"  Saved SNP annotations")
    
    summary = {
        'n_samples': len(common_samples),
        'n_snps_total': geno_aligned.shape[1],
        'n_snps_modulator': snp_info['is_modulator'].sum(),
        'n_snps_driver': (~snp_info['is_modulator']).sum(),
        'phenotype_mean': pheno_aligned.mean(),
        'phenotype_std': pheno_aligned.std(),
        'phenotype_min': pheno_aligned.min(),
        'phenotype_max': pheno_aligned.max()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "data_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return summary


def main():
    """Main execution."""
    print("="*60)
    print("WEEK 5: DATA PREPARATION")
    print("="*60)
    
    extract_file, snp_info = create_snp_extract_list()
    
    genotype_df, vcf_samples = extract_genotypes_vcf(snp_info)
    
    if genotype_df is None:
        print("\nERROR: Failed to extract genotypes")
        sys.exit(1)
    
    expr_data = load_expression_data_transposed()
    
    if not expr_data:
        print("\nERROR: Failed to load expression data")
        sys.exit(1)
    
    phenotype, common_samples, common_genes = compute_phenotypes(expr_data)
    
    if phenotype is None:
        print("\nERROR: Failed to compute phenotypes")
        sys.exit(1)
    
    summary = align_and_save_data(genotype_df, phenotype, snp_info, vcf_samples)
    
    if summary is None:
        print("\nERROR: Failed to align data")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nPrepared data saved to: {OUTPUT_DIR}")
    
    if summary['n_samples'] >= 50:
        print("\nREADY for NRS analysis")
        print("\nNext step: Run week5_nrs_simple.py")
    else:
        print("\nWARNING: Sample count is low")
        print("  Review sample alignment before proceeding")


if __name__ == "__main__":
    main()