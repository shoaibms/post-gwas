#!/usr/bin/env python3
"""
Enhanced Data Preparation for Environment-Conditional eQTL Analysis
===================================================================
Implements both baseline and drought-responsive gene selection strategies
with comprehensive logging and reproducible outputs.

Key Features:
- One-switch toggle between selection methods via GENE_SELECTOR environment variable
- Drought-responsive selection uses environment-specific expression variance
- Standardized AGPv4 gene ID formatting across all outputs
- Comprehensive provenance tracking and reproducibility controls
- Robust error handling and data validation

Author: ECT Analysis Pipeline
Date: 2025-01-09
Usage: 
    GENE_SELECTOR=baseline python prepare_data.py    # Current approach
    GENE_SELECTOR=drought python prepare_data.py     # Drought-responsive
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Iterable
import logging
import json
from datetime import datetime
import subprocess
import hashlib
import platform
import argparse
import gzip
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from pandas_plink import read_plink
from sklearn.preprocessing import StandardScaler
from scipy import stats
from itertools import combinations
from tqdm.auto import tqdm

# === NORMALIZATION HELPERS ===

def _norm_chr(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x)
    for k in ("chr","Chr","CHR","chromosome_","Chromosome_"):
        s = s.replace(k, "")
    if s in {"Mt","Pt","mitochondrion","chloroplast"}:
        return None
    try:
        i = int(s)
        return str(i) if 1 <= i <= 10 else None
    except ValueError:
        return None

def _ordered_unique(seq):  # stable de-dup
    return list(OrderedDict.fromkeys(seq))

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class PrepConfig:
    """Fixed configuration with correct file paths."""
    
    # Environment selection strategy
    GENE_SELECTOR = os.getenv("GENE_SELECTOR", "drought")  # "baseline" or "drought"
    STRICT_MODE = True
    
    # Data paths
    BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    
    EXPRESSION_FILES = {
    'WW': OUTPUT_DIR / "data_filtered" / "WW_209-Uniq_FPKM.agpv4.txt.gz",
    'WS1': OUTPUT_DIR / "data_filtered" / "WS1_208-uniq_FPKM.agpv4.txt.gz", 
    'WS2': OUTPUT_DIR / "data_filtered" / "WS2_210-uniq_FPKM.agpv4.txt.gz"
    }

    GENOTYPE_VCF = DATA_DIR / "maize" / "zea_mays_miss0.6_maf0.05.recode.vcf.gz"
    
    GFF3_FILE = DATA_DIR / "maize" / "Zea_mays.B73_RefGen_v4.gff3.gz"
    
    PVAR_FILE = OUTPUT_DIR / "geno" / "cohort_pruned.pvar"
    PGEN_FILE = OUTPUT_DIR / "geno" / "cohort_pruned.pgen"
    PSAM_FILE = OUTPUT_DIR / "geno" / "cohort_pruned.psam"
    
    # Output structure
    TRANSFORMER_DATA_DIR = OUTPUT_DIR / "transformer_data"
    COHORT_DIR = OUTPUT_DIR / "cohort"
    GENE_METADATA_DIR = OUTPUT_DIR / "gene_metadata"
    
    # Analysis parameters
    N_TARGET_GENES = 500
    N_PCS = 10
    TOP_K_CIS_SNPS = 16
    CIS_WINDOW_MB = 1.0
    LOG_TRANSFORM_OFFSET = 1e-6
    
    # Quality control thresholds
    MIN_EXPRESSION_SAMPLES = 50
    MIN_SNP_MAF = 0.05
    MAX_MISSING_RATE = 0.1
    
    # Reproducibility
    RANDOM_SEED = 42
    N_THREADS = os.cpu_count() or 1

# =============================================================================
# LOGGING SETUP
# =============================================================================

class _AsciiConsole(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        # Replace common symbols with ASCII
        return (msg.replace("→", "->")
                   .replace("≥", ">=")
                   .replace("≤", "<=")
                   .replace("±", "+/-"))

def setup_logging(config: PrepConfig) -> logging.Logger:
    """Setup comprehensive logging with provenance tracking."""
    log_dir = config.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prepare_data_{config.GENE_SELECTOR}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_AsciiConsole('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"DATA PREPARATION PIPELINE - {config.GENE_SELECTOR.upper()} MODE")
    logger.info("=" * 80)
    logger.info(f"Gene selector: {config.GENE_SELECTOR}")
    logger.info(f"Target genes: {config.N_TARGET_GENES}")
    logger.info(f"Random seed: {config.RANDOM_SEED}")
    logger.info(f"Log file: {log_file}")
    
    return logger

# =============================================================================
# GENE SELECTION STRATEGIES
# =============================================================================

class GeneSelector:
    """Unified gene selection interface with multiple strategies."""
    
    def __init__(self, config: PrepConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.rng = np.random.RandomState(config.RANDOM_SEED)
    
    def select_genes(self, expr_df: pd.DataFrame, samples_df: pd.DataFrame) -> List[str]:
        """Main entry point for gene selection."""
        if self.config.GENE_SELECTOR == "baseline":
            return self._select_baseline_genes(expr_df, samples_df)
        elif self.config.GENE_SELECTOR == "drought":
            return self._select_drought_responsive_genes(expr_df, samples_df)
        else:
            raise ValueError(f"Unknown GENE_SELECTOR: {self.config.GENE_SELECTOR}")
    
    def _select_baseline_genes(self, expr_df: pd.DataFrame, samples_df: pd.DataFrame) -> List[str]:
        """
        Baseline selection: high variance and biological relevance.
        
        This reproduces the existing gene selection to maintain consistency
        with previously published results.
        """
        self.logger.info("Using BASELINE gene selection strategy")

        # Load existing gene list if available
        filename = f"transformer_gene_set_{self.config.N_TARGET_GENES}.csv"
        existing_gene_file = self.config.OUTPUT_DIR / filename
        if existing_gene_file.exists():
            self.logger.info(f"Loading existing baseline gene set: {existing_gene_file}")
            baseline_genes = pd.read_csv(existing_gene_file)
            if 'gene' in baseline_genes.columns:
                gene_list = baseline_genes['gene'].astype(str).tolist()
            elif 'gene_name' in baseline_genes.columns:
                gene_list = baseline_genes['gene_name'].astype(str).tolist()
            else:
                gene_list = baseline_genes.iloc[:, 0].astype(str).tolist()
            
            # Validate genes exist in expression data
            available_genes = set(expr_df.columns)
            valid_genes = [g for g in gene_list if g in available_genes]
            self.logger.info(f"Validated {len(valid_genes)}/{len(gene_list)} baseline genes")
            
            if len(valid_genes) >= self.config.N_TARGET_GENES:
                return valid_genes[:self.config.N_TARGET_GENES]
        
        # Fallback: variance-based selection
        self.logger.info("Fallback to variance-based selection")
        return self._variance_based_selection(expr_df)
    
    def _select_drought_responsive_genes(self, expr_df: pd.DataFrame, samples_df: pd.DataFrame) -> List[str]:
        """
        Drought-responsive selection: WW-baseline stress-consistent.
        
        Methodology:
        1. Identify genes present (median expression > 0) in all environments.
        2. Calculate robust standard deviation (MAD-based) for each gene.
        3. Compute standardized expression shifts in stress vs. baseline (Z-scores).
        4. Filter for genes with a consistent direction of change in both stress conditions.
        5. Filter for genes with a minimum magnitude of change (|Z| >= TAU_SD).
        6. Rank candidates by a composite stress score (sum of absolute Z-scores).
        7. Select top N genes, backfilling if necessary from direction-consistent genes.
        """
        self.logger.info("Using WW-baseline STRESS-CONSISTENT gene selection strategy")
        
        # Validate environment labels
        if 'env' not in samples_df.columns:
            raise ValueError("samples_df must contain 'env' column")
        
        expected_envs = {'WW', 'WS1', 'WS2'}
        available_envs = set(samples_df['env'].unique())
        if not expected_envs.issubset(available_envs):
            missing = expected_envs - available_envs
            raise ValueError(f"Missing environments: {missing}")

        env = samples_df["env"].values
        genes = expr_df.columns

        def robust_sd(col):
            # 1.4826 * MAD; returns float
            x = col.values.astype(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            sd = 1.4826 * mad
            return sd if sd > 0 else np.nan

        # Split by env
        expr_WW  = expr_df.loc[env == "WW"]
        expr_WS1 = expr_df.loc[env == "WS1"]
        expr_WS2 = expr_df.loc[env == "WS2"]

        # Medians per env (gene-wise)
        med_WW  = expr_WW.median(axis=0)
        med_WS1 = expr_WS1.median(axis=0)
        med_WS2 = expr_WS2.median(axis=0)

        # Presence gate with configurable threshold
        TAU_EXPR = float(os.getenv("SELECTION_TAU_EXPR", "1e-3"))
        present_all = (med_WW > TAU_EXPR) & (med_WS1 > TAU_EXPR) & (med_WS2 > TAU_EXPR)

        # Robust SD per gene (pooled across all samples)
        sd_all = expr_df.apply(robust_sd, axis=0).replace(0, np.nan)

        # WW-baseline shifts and standardized effect sizes
        d1 = med_WS1 - med_WW
        d2 = med_WS2 - med_WW
        z1 = d1 / sd_all
        z2 = d2 / sd_all

        # Consistent stress response: same direction in WS1 and WS2 vs WW
        same_dir = np.sign(z1.fillna(0)) == np.sign(z2.fillna(0))

        # Magnitude threshold (in robust SD units)
        TAU_SD = float(os.getenv("SELECTION_TAU_SD", 0.25))  # 0.25 is a conservative value
        mag_ok = (z1.abs() >= TAU_SD) & (z2.abs() >= TAU_SD)

        # Primary candidate mask
        mask = present_all & same_dir & mag_ok

        # Stress score for ranking (bigger = stronger & more consistent)
        stress_score = z1.abs().fillna(0) + z2.abs().fillna(0)

        # Build stats table for auditing
        sel_stats = pd.DataFrame({
            "gene_id": genes,
            "med_WW": med_WW.values, "med_WS1": med_WS1.values, "med_WS2": med_WS2.values,
            "d_WS1_vs_WW": d1.values, "d_WS2_vs_WW": d2.values,
            "z_WS1_vs_WW": z1.values, "z_WS2_vs_WW": z2.values,
            "present_all": present_all.values,
            "same_dir": same_dir.values,
            "mag_ok": mag_ok.values,
            "stress_score": stress_score.values,
        })

        # Rank by stress_score (desc), first within mask; if not enough, backfill by score
        TARGET = self.config.N_TARGET_GENES

        # Ensure sel_stats is indexed by gene_id for aligned boolean masking
        if "gene_id" in sel_stats.columns:
            sel_stats = sel_stats.set_index("gene_id", drop=False)
        else:
            sel_stats.index.name = "gene_id"

        # Build aligned masks (index = gene_id)
        mask_primary = (present_all & same_dir & mag_ok).reindex(sel_stats.index).fillna(False)
        mask_fallback = ((present_all & same_dir) & (~mag_ok)).reindex(sel_stats.index).fillna(False)

        # Select using .loc with aligned masks
        primary = sel_stats.loc[mask_primary].sort_values("stress_score", ascending=False)
        if TARGET - len(primary) > 0:
            need = TARGET - len(primary)
            # Exclude primary genes from the fallback pool to prevent duplicates
            fallback_candidates = sel_stats.loc[mask_fallback & ~sel_stats.index.isin(primary.index)]
            fallback = fallback_candidates.sort_values("stress_score", ascending=False).head(need)
            chosen = pd.concat([primary, fallback])
        else:
            chosen = primary.head(TARGET)

        if len(chosen) < TARGET:
            self.logger.warning(f"Could only select {len(chosen)} genes; target={TARGET}. Consider lowering SELECTION_TAU_SD.")

        # Final list uses the index (gene_id)
        selected_genes = list(chosen.head(TARGET).index)
        
        # Log summary
        self.logger.info("Drought-responsive selection statistics (WW-baseline):")
        self.logger.info(f"  Genes evaluated: {expr_df.shape[1]}")
        self.logger.info(f"  Presence (all envs) pass: {int(present_all.sum())}")
        self.logger.info(f"  Direction-consistent pass: {int((present_all & same_dir).sum())}")
        self.logger.info(f"  Magnitude (|z|>={TAU_SD}) pass: {int(mask.sum())}")
        self.logger.info(f"  Selected genes: {len(selected_genes)}")
        if not chosen.empty:
            self.logger.info(f"  Stress-score range (selected): {chosen.head(TARGET)['stress_score'].min():.3f} - {chosen.head(TARGET)['stress_score'].max():.3f}")

        # Save stats (auditable)
        gene_stats_file = self.config.GENE_METADATA_DIR / "drought_gene_statistics.csv"
        self.config.GENE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
        # Save audit table with gene_id as a column
        sel_stats.to_csv(gene_stats_file, index=False)
        self.logger.info(f"Saved gene statistics: {gene_stats_file}")

        return selected_genes
    
    def _variance_based_selection(self, expr_df: pd.DataFrame) -> List[str]:
        """Fallback variance-based selection for baseline mode."""
        gene_vars = expr_df.var(axis=0, skipna=True)
        top_var_genes = gene_vars.nlargest(self.config.N_TARGET_GENES)
        self.logger.info(f"Selected top {len(top_var_genes)} genes by variance")
        return top_var_genes.index.tolist()

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def load_and_process_expression(config: PrepConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process RNA-seq expression data from all environments.
    
    Returns:
        expr_df: Samples × genes expression matrix (log-transformed)
        samples_df: Sample metadata with environment labels
    """
    logger.info("Loading expression data from all environments")
    
    expr_data = []
    sample_metadata = []
    
    for env, filepath in config.EXPRESSION_FILES.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Expression file not found: {filepath}")
        
        logger.info(f"  Loading {env}: {filepath}")
        
        # Load expression data
        if filepath.suffix == '.gz':
            df = pd.read_csv(filepath, sep='\t', compression='gzip')
        else:
            df = pd.read_csv(filepath, sep='\t')
        
        # Convert from long to wide format if needed
        if 'sample' in df.columns and 'gene' in df.columns:
            # Long format: pivot to wide
            expr_wide = df.pivot(index='sample', columns='gene', values='FPKM')
        else:
            # Assume already in wide format (genes as columns)
            expr_wide = df.set_index(df.columns[0])  # First column as sample ID
        
        # Log transformation
        expr_log = np.log1p(expr_wide.fillna(0) + config.LOG_TRANSFORM_OFFSET)
        
        # Create sample metadata
        samples = pd.DataFrame({
            'sample_id': expr_log.index,
            'env': env,
            'accession': [s.split('_')[0] if '_' in str(s) else str(s) for s in expr_log.index]
        })
        samples.set_index('sample_id', inplace=True)
        
        expr_data.append(expr_log)
        sample_metadata.append(samples)
        
        logger.info(f"    Samples: {len(expr_log)}, Genes: {len(expr_log.columns)}")
    
    # Combine all environments
    expr_combined = pd.concat(expr_data, axis=0)
    samples_combined = pd.concat(sample_metadata, axis=0)
    
    # Remove genes with excessive missing values
    missing_rates = expr_combined.isnull().mean(axis=0)
    good_genes = missing_rates[missing_rates <= config.MAX_MISSING_RATE].index
    expr_filtered = expr_combined[good_genes].copy()
    
    logger.info(f"Combined expression data: {len(expr_filtered)} samples × {len(expr_filtered.columns)} genes")
    logger.info(f"Removed {len(expr_combined.columns) - len(good_genes)} genes with >10% missing values")
    
    # Quality control logging
    logger.info("Expression data quality metrics:")
    logger.info(f"  Mean expression range: {expr_filtered.mean().min():.3f} - {expr_filtered.mean().max():.3f}")
    logger.info(f"  Expression variance range: {expr_filtered.var().min():.3f} - {expr_filtered.var().max():.3f}")
    logger.info(f"  Samples per environment: {samples_combined['env'].value_counts().to_dict()}")
    
    return expr_filtered, samples_combined

def process_genotypes(config: PrepConfig, logger: logging.Logger, samples_df: pd.DataFrame,
                      expected_samples: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Force-generate a filtered PLINK2 BED set, load it, and perform
    high-performance imputation using NumPy before creating the final DataFrame.
    """

    bed_prefix = Path(r"C:\Users\ms\Desktop\gwas\output\geno\cohort_pruned_bed")
    plink2_exe = Path(r"C:\Users\ms\Desktop\gwas\code\data_reading\plink2.exe")
    pfile_prefix = Path(r"C:\Users\ms\Desktop\gwas\output\geno\cohort_pruned")

    # Always regenerate filtered BED
    logger.info("Running PLINK2 to regenerate filtered BED set...")
    plink2_cmd = [
        str(plink2_exe),
        "--pfile", str(pfile_prefix),
        "--maf", str(config.MIN_SNP_MAF),
        "--geno", str(config.MAX_MISSING_RATE),
        "--max-alleles", "2",
        "--make-bed",
        "--out", str(bed_prefix)
    ]
    # Using capture_output=True and text=True to log stdout/stderr from PLINK
    result = subprocess.run(plink2_cmd, check=True, capture_output=True, text=True)
    logger.info("PLINK2 stdout:\n" + result.stdout)
    if result.stderr:
        logger.warning("PLINK2 stderr:\n" + result.stderr)

    # Load with pandas-plink
    logger.info(f"Loading genotypes with pandas-plink from: {bed_prefix}")
    (bim, fam, G) = read_plink(str(bed_prefix))
    logger.info("Genotype data loaded into Dask array. Now computing to NumPy array...")

    # === High-Performance Imputation ===
    # Step 1: Convert Dask array to an in-memory NumPy array.
    # Note: this is a memory-intensive operation.
    geno_np = G.compute().T.astype(np.float32)  # Shape: samples x SNPs
    logger.info(f"Computed genotype matrix into NumPy array with shape: {geno_np.shape}")

    # Count missing values
    n_total_genotypes = geno_np.size
    n_missing_genotypes = np.isnan(geno_np).sum()
    percent_missing = (n_missing_genotypes / n_total_genotypes) * 100 if n_total_genotypes > 0 else 0
    
    logger.info("=" * 30 + " IMPUTATION REPORT " + "=" * 30)
    logger.info(f"  Total genotype values: {n_total_genotypes:,}")
    logger.info(f"  Missing values to be imputed: {n_missing_genotypes:,}")
    logger.info(f"  Percentage of data to be imputed: {percent_missing:.4f}%")
    logger.info("=" * 80)

    # Step 2: Impute missing values using fast NumPy functions.
    logger.info("Imputing missing genotypes using NumPy...")
    col_means = np.nanmean(geno_np, axis=0)
    
    # Find the indices of NaN values
    nan_indices = np.where(np.isnan(geno_np))
    
    # Replace NaNs with the mean of their respective column
    # np.take is an efficient way to look up the correct mean for each NaN
    geno_np[nan_indices] = np.take(col_means, nan_indices[1])
    logger.info("NumPy imputation complete.")

    # Step 3: Now, create the pandas DataFrame from the *fully imputed* NumPy array.
    geno = pd.DataFrame(
        geno_np,
        index=fam.iid.astype(str),
        columns=bim.snp.astype(str)
    )
    logger.info(f"Created imputed genotype DataFrame. Final shape: {geno.shape}")

    # Harmonize IDs with expression accessions
    def _harmonize_ids(ids: pd.Index, cohort_accessions: set) -> pd.Series:
        orig = pd.Series(ids, index=ids, name="original_id")
        cand = orig.copy()
        strategies = [
            lambda s: s.str.replace(r"^0_", "", regex=True),
            lambda s: s.str.replace(r"^[0-9_]+(?=[A-Za-z])", "", regex=True),
            lambda s: s
        ]
        for fn in strategies:
            tmp = fn(cand)
            if tmp.isin(cohort_accessions).sum() >= cand.isin(cohort_accessions).sum():
                cand = tmp
        mapping = pd.DataFrame({"original_id": orig, "harmonized_id": cand})
        mapping_file = config.COHORT_DIR / "genotype_id_mapping.csv"
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(mapping_file, index=False)
        logger.info(f"Saved genotype ID harmonization map: {mapping_file}")
        return cand

    geno.index = _harmonize_ids(geno.index.astype(str), set(samples_df["accession"]))

    # Subset to expected samples if provided
    if expected_samples is not None:
        overlap = [s for s in expected_samples if s in geno.index]
        geno = geno.loc[overlap]
        logger.info(f"Subset to {len(overlap)} expected samples")

    pcs_df = calculate_pcs_robust(geno, config, logger)

    return geno, pcs_df

def calculate_pcs_robust(geno_df, config, logger):
    """Robust PCA calculation with proper error handling."""
    logger.info("Calculating PCs from genotype data")
    
    if geno_df.empty:
        logger.warning("Genotype data is empty, cannot compute PCs.")
        return pd.DataFrame(index=geno_df.index)
    
    n_pcs = min(config.N_PCS, geno_df.shape[0] - 1, geno_df.shape[1])
    pcs_df = pd.DataFrame(index=geno_df.index)  # Initialize empty DataFrame
    
    try:
        # Try fbpca first (faster)
        logger.info("Using fbpca for faster PCA calculation.")
        import fbpca
        U, s, Vt = fbpca.pca(geno_df.values, k=n_pcs, raw=True)
        pcs = U * s  # Principal components
        logger.info("fbpca PCA completed successfully.")
        
    except ImportError:
        logger.info("fbpca not available, using sklearn PCA.")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=config.RANDOM_SEED)
            pcs = pca.fit_transform(geno_df.values)
            logger.info("sklearn PCA completed successfully.")
        except Exception as e:
            logger.error(f"sklearn PCA failed: {e}")
            pcs = None
            
    except Exception as e:
        logger.error(f"fbpca PCA failed: {e}")
        logger.info("Falling back to sklearn PCA.")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=config.RANDOM_SEED)
            pcs = pca.fit_transform(geno_df.values)
            logger.info("sklearn PCA fallback completed successfully.")
        except Exception as e2:
            logger.error(f"sklearn PCA fallback also failed: {e2}")
            pcs = None
    
    # Create DataFrame from successful PCA
    if pcs is not None:
        pcs_df = pd.DataFrame(pcs, index=geno_df.index, 
                            columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
        logger.info(f"PCA completed: {pcs.shape[1]} components calculated")
    else:
        logger.warning("All PCA methods failed. Returning empty PC DataFrame.")
    
    return pcs_df

def create_cis_mapping(selected_genes: List[str], geno_df: pd.DataFrame, 
                      config: PrepConfig, logger: logging.Logger) -> Dict[str, List[int]]:
    """
    Coordinate-based cis SNP mapping.
    """
    logger.info("Creating cis-regulatory mapping with corrected file paths")
    pvar = _load_snp_coordinates(config, logger)
    genes = _load_gene_coordinates(config, logger)
    if pvar is None or pvar.empty or genes is None or genes.empty:
        logger.error("Missing SNP or gene coordinates; cannot map cis.")
        return {}
    mapping = _perform_cis_mapping(selected_genes, geno_df, pvar, genes, config, logger)
    return mapping


def _load_snp_coordinates(config: PrepConfig, logger: logging.Logger) -> pd.DataFrame:
    # Prefer PVAR generated by plink2
    if config.PVAR_FILE.exists():
        logger.info(f"Loading SNP coordinates from PVAR: {config.PVAR_FILE}")
        pvar = pd.read_csv(
            config.PVAR_FILE,
            sep=r"\s+|\t", engine="python", comment="#",
            header=None, names=["CHR","POS","ID","REF","ALT"],
            usecols=[0,1,2,3,4], dtype={"CHR":str, "POS":int, "ID":str}
        )
    else:
        # Fallback: parse VCF directly
        logger.info(f"PVAR missing, parsing VCF: {config.GENOTYPE_VCF}")
        fh = gzip.open(config.GENOTYPE_VCF, "rt") if str(config.GENOTYPE_VCF).endswith(".gz") else open(config.GENOTYPE_VCF, "r")
        recs = []
        with fh as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 5: 
                    continue
                chr_raw = c[0]
                pos = int(c[1])
                vid = c[2] if c[2] != "." else f"{chr_raw}_{pos}"
                recs.append((chr_raw, pos, vid, c[3], c[4]))
        pvar = pd.DataFrame(recs, columns=["CHR","POS","ID","REF","ALT"])

    # Filter out SNPs with missing IDs ('.')
    n_before_filter = len(pvar)
    pvar = pvar[pvar["ID"] != "."].copy()
    n_after_filter = len(pvar)
    logger.info(f"Filtered out {n_before_filter - n_after_filter} SNPs with missing IDs ('.')")

    # Normalize chromosomes and filter to {1..10}
    pvar["chr_norm"] = pvar["CHR"].map(_norm_chr)
    pvar = pvar[pvar["chr_norm"].notna()].copy()
    pvar["POS"] = pvar["POS"].astype(int)

    logger.info(f"Loaded {len(pvar):,} SNPs with valid IDs after chr normalization (nuclear 1..10)")
    return pvar


def _load_gene_coordinates(config: PrepConfig, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading gene coordinates from GFF3: {config.GFF3_FILE}")
    open_fn = gzip.open if str(config.GFF3_FILE).endswith(".gz") else open
    rows = []
    with open_fn(config.GFF3_FILE, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, _, ftype, start, end, _, strand, _, attrs = parts
            if ftype != "gene":
                continue
            # parse AGPv4 gene ID
            gene_id = None
            for kv in attrs.split(";"):
                if kv.startswith("ID="):
                    val = kv.split("=",1)[1]
                    if val.startswith("gene:"):
                        val = val[5:]
                    gene_id = val
                    break
            if (gene_id is None) or (not gene_id.startswith("Zm00001d")):
                continue  # enforce AGPv4
            chr_norm = _norm_chr(seqid)
            if chr_norm is None:
                continue
            start_i, end_i = int(start), int(end)
            tss = end_i if strand == "-" else start_i  # 1-based consistent with PVAR
            rows.append((gene_id, chr_norm, tss))
    genes = pd.DataFrame(rows, columns=["gene_id","chr_norm","tss"])
    logger.info(f"Found {len(genes):,} gene records in GFF3 (AGPv4; chr 1..10)")
    return genes


def _perform_cis_mapping(selected_genes: List[str],
                         geno_df: pd.DataFrame,
                         pvar: pd.DataFrame,
                         genes: pd.DataFrame,
                         config: PrepConfig,
                         logger: logging.Logger) -> Dict[str, List[int]]:
    """
    Selects TOP_K unique cis-SNPs by de-duplicating within the cis-window
    and then sorting by distance to TSS.
    """
    window_bp = int(config.CIS_WINDOW_MB * 1_000_000)
    genes_sel = genes[genes["gene_id"].isin(selected_genes)].copy()
    logger.info(f"Mapping diverse cis-SNPs for {len(genes_sel)}/{len(selected_genes)} genes found in GFF3.")

    id_to_col = {str(s): j for j, s in enumerate(map(str, geno_df.columns))}
    pvar_in_G = pvar[pvar["ID"].astype(str).isin(id_to_col)].copy()

    mapping: Dict[str, List[int]] = {}
    pvar_by_chr = {c: df for c, df in pvar_in_G.groupby("chr_norm")}
    
    first_gene_processed = False
    for _, row in tqdm(genes_sel.iterrows(), total=len(genes_sel), desc="Mapping diverse cis-SNPs"):
        gid, chr_n, tss = row["gene_id"], row["chr_norm"], int(row["tss"])
        pv = pvar_by_chr.get(chr_n, pd.DataFrame())

        if pv.empty:
            mapping[gid] = []
            continue

        # 1. Window filter
        w = pv[(pv["POS"] >= tss - window_bp) & (pv["POS"] <= tss + window_bp)].copy()

        if w.empty:
            mapping[gid] = []
            continue

        # 2. Ensure we only consider unique SNPs within the window
        w["dist"] = (w["POS"] - tss).abs()
        
        # First, sort by distance so that when we drop duplicates, we keep the closest one
        w.sort_values("dist", inplace=True)
        
        # Now, drop any duplicate SNP IDs, keeping the first occurrence (which is the closest)
        unique_snps_in_window = w.drop_duplicates(subset=['ID'], keep='first')
        
        # 3. Select the top K from this cleaned list of *unique* SNPs
        top_k_snps = unique_snps_in_window.head(config.TOP_K_CIS_SNPS)
        chosen_ids = top_k_snps["ID"].astype(str).tolist()

        # 4. Map the chosen SNP IDs back to their column indices
        mapping[gid] = [id_to_col[sid] for sid in chosen_ids if sid in id_to_col]

    # --- Reporting Section ---
    cov_rows = []
    for _, row in genes_sel.iterrows():
        gid, chr_n, tss = row["gene_id"], row["chr_norm"], int(row["tss"])
        pv = pvar_by_chr.get(chr_n, pd.DataFrame())
        n_in_window = 0
        if not pv.empty:
            w_report = pv[(pv["POS"] >= tss - window_bp) & (pv["POS"] <= tss + window_bp)]
            n_in_window = len(w_report)
        cov_rows.append((gid, chr_n, tss, n_in_window))

    cov = pd.DataFrame(cov_rows, columns=["gene_id","chr_norm","tss","n_cis_snps"])
    n_pos = int((cov["n_cis_snps"] > 0).sum())
    pct = n_pos / max(1, len(cov))
    logger.info(f"Cis-mapping complete. Genes with >=1 SNP in window: {n_pos}/{len(cov)} ({pct*100:.1f}%)")
    
    return mapping


def align_and_integrate_data(
    expr_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    geno_df: pd.DataFrame,
    pcs_df: pd.DataFrame,
    selected_genes: list[str],
    logger: logging.Logger,
    config: PrepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- helpers ---
    def _std_index(df: pd.DataFrame, name: str, allow_dupes: bool) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.astype(str).str.strip()
        if not allow_dupes and not df.index.is_unique:
            n_dup = df.index.duplicated(keep="first").sum()
            logger.warning(f"{name}: index not unique -> dropping {n_dup} duplicate rows (keep=first)")
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _report_id_diff(a: set, b: set, name_a: str, name_b: str, path: Path):
        only_a = sorted(a - b)[:20]
        only_b = sorted(b - a)[:20]
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            f.write(f"Only in {name_a} (first 20): {only_a}\n")
            f.write(f"Only in {name_b} (first 20): {only_b}\n")
        logger.info(f"ID difference report saved to {path}")

    # sanitize indices: keep dupes for expr/samples, enforce unique for geno/pcs
    expr_df    = _std_index(expr_df,    "expr",    allow_dupes=True)
    samples_df = _std_index(samples_df, "samples", allow_dupes=True)
    geno_df    = _std_index(geno_df,    "geno",    allow_dupes=False)
    pcs_df     = _std_index(pcs_df,     "pcs",     allow_dupes=False)

    # Compute overlap by accession (genotypes are per accession, expression per sample)
    n_expr = samples_df.shape[0]
    n_acc_expr = samples_df["accession"].nunique()
    n_acc_geno = geno_df.index.nunique()
    n_acc_overlap = len(set(samples_df["accession"]).intersection(set(geno_df.index)))

    logger.info(f"ID counts -> expr_samples:{n_expr} | expr_accessions:{n_acc_expr} | geno_accessions:{n_acc_geno}")
    logger.info(f"Accession overlap (expression ∩ genotype): {n_acc_overlap}")

    # if a 'sample_id' column exists, reindex samples_df onto it (without losing duplicates)
    if "sample_id" in samples_df.columns:
        s = samples_df["sample_id"].astype(str).str.strip()
        # keep original order; set_index will preserve duplicates
        samples_df = samples_df.set_index(s)
        samples_df.index.name = "sample_id"

    # ---------- fast path: align by sample_id, anchored on expr_df ----------
    # mask has same length/order as expr_df
    mask = expr_df.index.isin(samples_df.index) & expr_df.index.isin(geno_df.index) & expr_df.index.isin(pcs_df.index)

    if mask.any():
        # positions where all modalities have this sample_id
        pos = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)
        # labels in the same order as expr rows
        labels_in_order = expr_df.index[pos].astype(str)

        # 1) select expr & samples by **position** (avoids label-based inflation on duplicate indices)
        expr_aligned    = expr_df.iloc[pos][selected_genes].copy()
        samples_aligned = samples_df.iloc[pos].copy()

        # 2) select geno & PCs by **label reindex** to match the expr order (duplicates are replicated as needed)
        geno_aligned = geno_df.reindex(labels_in_order).copy()
        pcs_aligned  = pcs_df.reindex(labels_in_order).copy()

        order = labels_in_order.tolist()
    else:
        # ---------- accession fallback (for real VCFs indexed by accession) ----------
        logger.warning("No common rows by sample_id; attempting accession-based alignment")
        if "accession" not in samples_df.columns:
            raise ValueError("Fallback requires 'accession' column in samples_df.")
        # keep expression rows whose accession exists in geno & pcs
        acc = samples_df["accession"].astype(str).str.strip()
        ok = acc.isin(geno_df.index) & acc.isin(pcs_df.index)
        if not ok.any():
            raise ValueError("No common accessions across samples/genotypes/PCs.")
        order = expr_df.index[expr_df.index.isin(samples_df.index[ok])]
        samples_sub = samples_df.loc[order]
        expr_aligned    = expr_df.loc[order, selected_genes].copy()
        # duplicate geno/pcs per sample using accession lookup
        acc_order = samples_sub["accession"].astype(str).str.strip().tolist()
        geno_aligned = geno_df.loc[acc_order].copy()
        pcs_aligned  = pcs_df.loc[acc_order].copy()
        # relabel indices to sample_id
        geno_aligned.index = order
        pcs_aligned.index  = order
        samples_aligned    = samples_sub.copy()
        logger.info("Accession fallback aligned samples: %d", len(order))

    n = len(order)
    # final sanity
    if expr_aligned.shape[0] != n or samples_aligned.shape[0] != n or geno_aligned.shape[0] != n or pcs_aligned.shape[0] != n:
        logger.error("Alignment failed. Reporting ID differences before raising error.")
        _report_id_diff(set(expr_df.index), set(geno_df.index), "expr", "geno", config.COHORT_DIR / "id_diff_expr_vs_geno.txt")
        _report_id_diff(set(samples_df.index), set(geno_df.index), "samples", "geno", config.COHORT_DIR / "id_diff_samples_vs_geno.txt")
        raise ValueError(
            f"Alignment mismatch: expr={expr_aligned.shape[0]}, samples={samples_aligned.shape[0]}, "
            f"geno={geno_aligned.shape[0]}, pcs={pcs_aligned.shape[0]}, expected={n}"
        )
    logger.info(f"Aligned dataset -> {n} sample rows (genotypes replicated per sample) x {len(selected_genes)} genes; genotypes: {geno_aligned.shape[1]} SNPs; PCs: {pcs_aligned.shape[1]}")
    return expr_aligned, samples_aligned, geno_aligned, pcs_aligned


# =============================================================================
# PYTORCH DATASET CREATION
# =============================================================================

def create_pytorch_dataset(integrated_data: Dict, gene_snp_mapping: Dict[str, List[int]],
                          config: PrepConfig, logger: logging.Logger, output_file: str) -> str:
    """
    Create PyTorch tensor dataset for transformer training.
    
    Returns path to saved dataset file.
    """
    logger.info("Creating PyTorch tensor dataset")
    
    # Extract data components
    Y = integrated_data['expression'].values.astype(np.float32)
    G = integrated_data['genotypes'].values.astype(np.float32) 
    PCs = integrated_data['pcs'].values.astype(np.float32)
    E = integrated_data['env_encoded'].astype(np.int64)
    
    # Create cis-regulatory mask
    n_genes = len(integrated_data['selected_genes'])
    n_snps = G.shape[1]
    
    gene_snp_mask = torch.zeros((n_genes, n_snps), dtype=torch.bool)
    gene_snp_indices = []
    
    for i, gene in enumerate(integrated_data['selected_genes']):
        if gene in gene_snp_mapping:
            snp_indices = gene_snp_mapping[gene]

            # Enforce uniqueness and bounds (guards duplicate “same SNP x8” cases)
            uniq = sorted({idx for idx in snp_indices if 0 <= idx < n_snps})
            if len(uniq) != len(snp_indices):
                logger.warning(f"[cis-map] {gene}: {len(snp_indices)-len(uniq)} duplicate/out-of-bounds SNP indices removed")

            if uniq:
                gene_snp_mask[i, uniq] = True
                gene_snp_indices.append(uniq)
            else:
                gene_snp_indices.append([])
                logger.warning(f"[cis-map] {gene}: no valid cis-SNPs after filtering")
        else:
            gene_snp_indices.append([])
    
    # Convert to tensors with gene_names inclusion
    tensors = {
        'Y': torch.from_numpy(Y),
        'G': torch.from_numpy(G), 
        'PCs': torch.from_numpy(PCs),
        'E': torch.from_numpy(E),
        'gene_snp_mask': gene_snp_mask,
        'gene_snp_indices': gene_snp_indices,
        'gene_names': [str(g) for g in integrated_data['selected_genes']],
        'sample_names': integrated_data['common_samples'],
        'env_mapping': integrated_data['env_mapping']
    }
    
    # Store gene IDs at top-level for downstream script compatibility
    tensors['gene_ids'] = [str(g) for g in integrated_data['selected_genes']]

    # Add metadata
    metadata = {
        'n_samples': Y.shape[0],
        'n_genes': Y.shape[1], 
        'n_snps': G.shape[1],
        'n_pcs': PCs.shape[1],
        'gene_selector': config.GENE_SELECTOR,
        'creation_timestamp': datetime.now().isoformat(),
        'random_seed': config.RANDOM_SEED,
    }
    
    metadata['gene_ids'] = tensors['gene_ids']
    tensors['metadata'] = metadata
    
    # Save dataset
    dataset_file = output_file
    Path(dataset_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, dataset_file)
    
    logger.info(f"SUCCESS: Data bundle saved to: {dataset_file}")
    logger.info(f"Dataset shape: {Y.shape[0]} samples x {Y.shape[1]} genes")
    logger.info(f"Genotype shape: {G.shape}")
    logger.info(f"PC shape: {PCs.shape}")
    logger.info(f"Gene names included: {len(tensors['gene_names'])} genes")
    
    return str(dataset_file)

# =============================================================================
# SIDE-CAR EXPORT: cis stats per gene (Meff via LD-eigen, MAF summary, genotype-class counts)
# =============================================================================
def compute_and_write_cis_sidecar(integrated_data: Dict,
                                  gene_snp_mapping: Dict[str, List[int]],
                                  config: PrepConfig,
                                  logger: logging.Logger) -> str:
    """Write a side-car CSV with per-gene cis stats without changing core outputs.
    Columns:
      gene_id, n_cis_snps, meff_ld_eigen, maf_min, maf_median, maf_max,
      count0_WW, count1_WW, count2_WW, count0_WS1, count1_WS1, count2_WS1, count0_WS2, count1_WS2, count2_WS2,
      rep_snp_col (index), rep_snp_maf
    Notes:
      • Meff computed from eigenvalues of the SNP correlation matrix: Meff = p^2 / sum(lambda_i^2) (Cheverud/Li–Ji style).
      • Genotype-class counts are computed for a representative SNP (highest MAF within cis window, for balance).
    """
    try:
        geno_df = integrated_data['genotypes']
        samples_df = integrated_data['samples']
        env_order = list(integrated_data.get('env_mapping', {}).keys()) or ['WW','WS1','WS2']
        genes = list(integrated_data['selected_genes'])
        
        out_rows = []
        for gid in genes:
            snp_idx = gene_snp_mapping.get(gid, []) or []
            n_cis = len(snp_idx)
            meff = float('nan')
            maf_min = maf_med = maf_max = float('nan')
            rep_col = -1
            rep_maf = float('nan')
            counts = {f"{g}{env}": 0 for env in env_order for g in ('count0_','count1_','count2_')}
            
            if n_cis > 0:
                X = geno_df.iloc[:, snp_idx].to_numpy(copy=False)
                X = np.clip(X, 0.0, 2.0)
                p = np.mean(X, axis=0) / 2.0
                maf = np.minimum(p, 1.0 - p)
                if maf.size > 0:
                    maf_min = float(np.min(maf))
                    maf_med = float(np.median(maf))
                    maf_max = float(np.max(maf))
                    rep_col_local = int(np.argmax(maf))
                    rep_col = int(snp_idx[rep_col_local])
                    rep_maf = float(maf[rep_col_local])
                
                col_var = np.var(X, axis=0)
                keep = col_var > 1e-8
                k = int(np.sum(keep))
                if k >= 1:
                    Xk = X[:, keep]
                    with np.errstate(all='ignore'):
                        R = np.corrcoef(Xk, rowvar=False)
                    if np.any(~np.isfinite(R)):
                        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                        np.fill_diagonal(R, 1.0)
                    w = np.linalg.eigvalsh(R)
                    denom = float(np.sum(w**2))
                    meff_raw = (k * k) / denom if denom > 0 else float(k)
                    meff = float(np.clip(meff_raw, 1.0, float(k)))
                
                if rep_col >= 0:
                    g = geno_df.iloc[:, rep_col].to_numpy(copy=False)
                    g = np.rint(np.clip(g, 0.0, 2.0)).astype(int)
                    env_series = samples_df['env'].astype(str).values
                    for env in env_order:
                        mask = (env_series == env)
                        if mask.any():
                            vals, freqs = np.unique(g[mask], return_counts=True)
                            for v, c in zip(vals, freqs):
                                if v in (0,1,2):
                                    counts[f'count{v}_{env}'] = int(c)
            
            out_rows.append({
                'gene_id': gid,
                'n_cis_snps': int(n_cis),
                'meff_ld_eigen': float(meff),
                'maf_min': float(maf_min),
                'maf_median': float(maf_med),
                'maf_max': float(maf_max),
                **counts,
                'rep_snp_col': int(rep_col),
                'rep_snp_maf': float(rep_maf),
            })
        
        sidecar_df = pd.DataFrame(out_rows)
        sidecar_path = config.OUTPUT_DIR / f"cis_sidecar_stats_{config.GENE_SELECTOR}.csv"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_df.to_csv(sidecar_path, index=False)
        
        n_rows = len(sidecar_df)
        mean_n = float(sidecar_df['n_cis_snps'].mean()) if n_rows else 0.0
        med_meff = float(sidecar_df['meff_ld_eigen'].median()) if n_rows else float('nan')
        logger.info(f"[side-car] Wrote cis stats for {n_rows} genes → {sidecar_path}")
        logger.info(f"[side-car] mean n_cis_snps={mean_n:.2f}; median Meff={med_meff:.2f}")
        try:
            logger.info("[side-car] head(3):\n" + sidecar_df.head(3).to_string(index=False))
        except Exception:
            pass
        
        return str(sidecar_path)
    except Exception as e:
        logger.error(f"[side-car] Failed to compute/write cis stats: {e}", exc_info=True)
        return ""

# =============================================================================
# OUTPUT STANDARDIZATION
# =============================================================================

def save_standardized_outputs(integrated_data: Dict, config: PrepConfig, logger: logging.Logger):
    """
    Save standardized output files with consistent formatting.
    """
    logger.info("Saving standardized output files")
    
    # Gene list with standardized format
    gene_df = pd.DataFrame({
        'gene_name': integrated_data['selected_genes']  # Use gene_name for consistency
    })
    gene_file = config.OUTPUT_DIR / f"transformer_gene_set_{config.N_TARGET_GENES}_{config.GENE_SELECTOR}.csv"
    assert len(integrated_data['selected_genes']) == config.N_TARGET_GENES, "Gene list not at target N prior to save."
    gene_df.to_csv(gene_file, index=False)
    logger.info(f"Saved gene list: {gene_file}")

    # Version the gene set with a content hash
    try:
        gid_hash = hashlib.md5((";".join(gene_df['gene_name'].astype(str))).encode()).hexdigest()[:10]
        vpath = config.OUTPUT_DIR / f"transformer_gene_set_{config.N_TARGET_GENES}_{config.GENE_SELECTOR}_{gid_hash}.csv"
        gene_df.to_csv(vpath, index=False)
        logger.info(f"Saved versioned gene list: {vpath}")
    except Exception as e:
        logger.warning(f"Could not create versioned gene list: {e}")
    
    # Sample metadata
    samples_output = integrated_data['samples'].copy()
    samples_output['sample_name'] = samples_output.index  # Standardize sample ID column
    samples_file = config.COHORT_DIR / f"samples_metadata_{config.GENE_SELECTOR}.csv" 
    config.COHORT_DIR.mkdir(parents=True, exist_ok=True)
    samples_output.to_csv(samples_file, index=True)
    logger.info(f"Saved sample metadata: {samples_file}")
    
    # Expression matrix (samples × genes, standardized column names)
    expr_output = integrated_data['expression'].copy()
    expr_output.columns.name = 'gene_name'
    expr_file = config.OUTPUT_DIR / f"expression_matrix_{config.GENE_SELECTOR}.csv"
    expr_output.to_csv(expr_file)
    logger.info(f"Saved expression matrix: {expr_file}")

# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_input_files(config: PrepConfig, logger: logging.Logger) -> bool:
    """Validate that all required input files exist."""
    
    logger.info("Validating input files...")
    
    missing_files = []
    
    # Check expression files
    for env, file_path in config.EXPRESSION_FILES.items():
        if not file_path.exists():
            missing_files.append(f"Expression {env}: {file_path}")
    
    # Check genotype file
    if not config.GENOTYPE_VCF.exists():
        missing_files.append(f"Genotype VCF: {config.GENOTYPE_VCF}")
    
    # Check GFF3 file
    if not config.GFF3_FILE.exists():
        missing_files.append(f"GFF3 annotation: {config.GFF3_FILE}")
    
    if missing_files:
        logger.error("Missing required input files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        return False
    
    logger.info("All required input files found")
    return True

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_data_preparation(args: argparse.Namespace):
    """Main pipeline for data preparation."""
    
    # Initialize configuration and logging
    config = PrepConfig()
    config.CIS_WINDOW_MB = args.cis_window_kb / 1000.0
    logger = setup_logging(config)
    
    try:
        # Validate input files first
        if not validate_input_files(config, logger):
            raise FileNotFoundError("Required input files missing - check file paths")
        
        # Validate configuration
        logger.info("Validating configuration...")
        if config.GENE_SELECTOR not in ['baseline', 'drought']:
            raise ValueError(f"Invalid GENE_SELECTOR: {config.GENE_SELECTOR}. Must be 'baseline' or 'drought'")
        
        # Create output directories
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.TRANSFORMER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.GENE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load expression data
        logger.info("STEP 1: Loading expression data")
        expr_df, samples_df = load_and_process_expression(config, logger)
        
        # Step 2: Gene selection
        logger.info("STEP 2: Gene selection") 
        selector = GeneSelector(config, logger)
        selected_genes = selector.select_genes(expr_df, samples_df)
        
        # --- Enforce target number of genes, all present in AGPv4 GFF3 ---
        genes_gff = _load_gene_coordinates(config, logger)
        genes_in_gff = set(genes_gff["gene_id"])

        # Start with your selected list, but (i) dedup, (ii) drop anything not in GFF3
        selected_genes = _ordered_unique(selected_genes)
        selected_in = [g for g in selected_genes if g in genes_in_gff]

        target_n = config.N_TARGET_GENES
        missing_n = target_n - len(selected_in)

        if missing_n > 0:
            backfill = []
            # For drought, use the ranked stats file for deterministic backfill
            if config.GENE_SELECTOR == 'drought':
                stats_path = config.GENE_METADATA_DIR / "drought_gene_statistics.csv"
                if stats_path.exists():
                    stats_df = pd.read_csv(stats_path)

                    # Column compatibility (new selector vs old)
                    gene_col  = "gene_id" if "gene_id" in stats_df.columns else ("gene" if "gene" in stats_df.columns else None)
                    score_col = "stress_score" if "stress_score" in stats_df.columns else ("composite_score" if "composite_score" in stats_df.columns else None)
                    if gene_col is None or score_col is None:
                        raise RuntimeError(f"Gene stats missing required columns; have: {list(stats_df.columns)}")

                    ranked = (
                        stats_df
                        .sort_values(score_col, ascending=False)[gene_col]
                        .astype(str)
                        .tolist()
                    )
                    logger.info(f"Backfill ranking uses {score_col} from {gene_col} (n={len(ranked)}).")
                    pool = [g for g in ranked if (g in genes_in_gff) and (g not in selected_in)]
                    backfill = pool[:missing_n]
                else:
                    logger.warning(f"Drought stats file not found for backfilling: {stats_path}")

            # If backfill is still not enough (or for other modes), grab any valid gene from GFF3
            if len(backfill) < missing_n:
                needed = missing_n - len(backfill)
                current_genes = set(selected_in + backfill)
                extra_pool = [g for g in genes_gff["gene_id"].tolist() if g not in current_genes]
                backfill.extend(extra_pool[:needed])

            selected_genes = _ordered_unique(selected_in + backfill)

        # Final hard check
        if len(selected_genes) != target_n:
            raise RuntimeError(f"Backfill failed: have {len(selected_genes)} genes; target={target_n}")

        logger.info(f"Final gene list fixed at N={len(selected_genes)} (all AGPv4-valid).")
        
        # Step 3: Process genotypes
        logger.info("STEP 3: Processing genotype data")
        expected_samples = list(samples_df['accession'].astype(str).unique())
        geno_df, pcs_df = process_genotypes(config, logger, samples_df, expected_samples=expected_samples)
        
        # Step 4: Create cis-regulatory mapping
        logger.info("STEP 4: Creating cis-regulatory mapping")
        gene_snp_mapping = create_cis_mapping(selected_genes, geno_df, config, logger)
        
        # Validate cis-mapping results
        if selected_genes: # Avoid division by zero
            genes_with_snps = len(gene_snp_mapping)
            zero_snp_rate = (len(selected_genes) - genes_with_snps) / len(selected_genes)
            
            logger.info(f"Cis-mapping validation:")
            logger.info(f"  Genes with cis-SNPs: {genes_with_snps}/{len(selected_genes)} ({100-zero_snp_rate*100:.1f}%)")
            logger.info(f"  Zero-SNP rate: {zero_snp_rate*100:.1f}%")
            
            if zero_snp_rate > 0.2:  # More than 20% genes have no SNPs
                logger.warning(f"High zero-SNP rate detected! This will hurt LMM performance.")
                logger.warning("Consider:")
                logger.warning("1. Increasing CIS_WINDOW_MB")
                logger.warning("2. Checking gene ID format consistency")
                logger.warning("3. Verifying coordinate system alignment")
        
        # Step 5: Align and integrate data
        logger.info("STEP 5: Data integration and alignment")
        (
            expr_aligned,
            samples_aligned,
            geno_aligned,
            pcs_aligned,
        ) = align_and_integrate_data(
            expr_df, samples_df, geno_df, pcs_df, selected_genes, logger, config
        )
        
        # Re-create integrated_data dictionary for downstream functions
        assert not expr_aligned.isnull().any().any(), "Missing values in expression data"
        expr_aligned.columns = [str(col) for col in expr_aligned.columns] # Standardize gene names
        
        env_mapping = {'WW': 0, 'WS1': 1, 'WS2': 2}
        env_encoded = samples_aligned['env'].map(env_mapping).values
        
        logger.info(f"  Environment distribution: {pd.Series(env_encoded).value_counts().to_dict()}")
        
        integrated_data = {
            'expression': expr_aligned,
            'samples': samples_aligned,
            'genotypes': geno_aligned,
            'pcs': pcs_aligned,
            'env_encoded': env_encoded,
            'env_mapping': env_mapping,
            'selected_genes': selected_genes,
            'common_samples': samples_aligned.index.tolist(),
        }
        
        # Persist environment label mapping
        env_map_file = config.COHORT_DIR / "environment_label_map.json"
        with open(env_map_file, "w") as f:
            json.dump(env_mapping, f, indent=2)
        logger.info(f"Saved environment label map: {env_map_file}")

        # Save aligned sample list for audit
        aligned_samples_df = pd.DataFrame({
            "sample_id": integrated_data['common_samples'],
            "accession": samples_aligned["accession"].values,
            "env": samples_aligned["env"].values
        })
        aligned_samples_file = config.COHORT_DIR / f"aligned_samples_{config.GENE_SELECTOR}.csv"
        aligned_samples_df.to_csv(aligned_samples_file, index=False)
        logger.info(f"Saved aligned sample list: {aligned_samples_file}")
        
        # Step 6: Create PyTorch dataset
        logger.info("STEP 6: Creating PyTorch dataset")
        dataset_file = create_pytorch_dataset(integrated_data, gene_snp_mapping, config, logger, args.output_file)
        
        # Step 6.5: Write cis side-car stats (Meff, MAF summary, genotype-class counts)
        logger.info("STEP 6.5: Writing cis side-car stats")
        sidecar_file = compute_and_write_cis_sidecar(integrated_data, gene_snp_mapping, config, logger)
        if sidecar_file:
            logger.info(f"STEP 6.5 OK: side-car saved at {sidecar_file}")
        else:
            logger.warning("STEP 6.5 skipped or failed: no side-car produced")
        
        # Step 7: Save standardized outputs
        logger.info("STEP 7: Saving standardized outputs")
        save_standardized_outputs(integrated_data, config, logger)
        
        # Step 8: Generate summary report
        logger.info("STEP 8: Generating summary report")
        
        summary_report = {
            'pipeline_version': '2.0_drought_enhanced',
            'gene_selector': config.GENE_SELECTOR,
            'completion_timestamp': datetime.now().isoformat(),
            'random_seed': config.RANDOM_SEED,
            'dataset_stats': {
                'n_samples': len(integrated_data['common_samples']),
                'n_genes': len(selected_genes),
                'n_snps': len(geno_df.columns),
                'n_pcs': config.N_PCS,
                'environments': list(integrated_data['env_mapping'].keys())
            },
            'output_files': {
                'pytorch_dataset': dataset_file,
                'gene_list': str(config.OUTPUT_DIR / f"transformer_gene_set_{config.N_TARGET_GENES}_{config.GENE_SELECTOR}.csv"),
                'expression_matrix': str(config.OUTPUT_DIR / f"expression_matrix_{config.GENE_SELECTOR}.csv"),
                'sample_metadata': str(config.COHORT_DIR / f"samples_metadata_{config.GENE_SELECTOR}.csv")
            }
        }
        
        summary_file = config.OUTPUT_DIR / f"preparation_summary_{config.GENE_SELECTOR}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Step 9: Generate provenance manifest
        logger.info("STEP 9: Generating provenance manifest")
        def _md5(p: Path) -> Optional[str]:
            if not p.exists(): return None
            h = hashlib.md5()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1<<20), b""):
                    h.update(chunk)
            return h.hexdigest()

        plink_binary_prefix = Path(r"C:\Users\ms\Desktop\gwas\output\geno\cohort_pruned")
        pgen_file = plink_binary_prefix.with_suffix('.pgen')
        pvar_file = plink_binary_prefix.with_suffix('.pvar')
        psam_file = plink_binary_prefix.with_suffix('.psam')

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "selector": config.GENE_SELECTOR,
            "selector_params": {
                "SELECTION_TAU_SD": float(os.getenv("SELECTION_TAU_SD", 0.25)),
                "SELECTION_TAU_EXPR": float(os.getenv("SELECTION_TAU_EXPR", "1e-3")),
            },
            "n_samples": len(integrated_data['common_samples']),
            "n_genes": len(selected_genes),
            "n_snps": geno_aligned.shape[1],
            "pcs": config.N_PCS,
            "inputs": {
                "expression_files": {
                    k: {"path": str(v), "size": v.stat().st_size if v.exists() else None, "md5": _md5(v)}
                    for k, v in config.EXPRESSION_FILES.items()
                },
                "plink_pfiles": {
                    "pgen": {"path": str(pgen_file), "md5": _md5(pgen_file)},
                    "pvar": {"path": str(pvar_file), "md5": _md5(pvar_file)},
                    "psam": {"path": str(psam_file), "md5": _md5(psam_file)},
                },
                "code": {
                    "script_md5": _md5(Path(__file__)),
                    "python": sys.version,
                    "platform": platform.platform()
                }
            }
        }
        
        manifest_file = config.OUTPUT_DIR / "preparation_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Provenance manifest saved: {manifest_file}")

        logger.info("=" * 80)
        logger.info("DATA PREPARATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Gene selector: {config.GENE_SELECTOR}")
        logger.info(f"Final dataset: {summary_report['dataset_stats']}")
        logger.info(f"PyTorch dataset: {dataset_file}")
        logger.info(f"Summary report: {summary_file}")
        logger.info(f"Provenance manifest: {manifest_file}")
        
        return summary_report
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Data Preparation for Environment-Conditional eQTL Analysis")
    parser.add_argument("--cis-window-kb", type=int, default=1000,
                        help="Cis-regulatory window size in kilobases (e.g., 1000 for +/- 1Mb).")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Full path for the output .pt file (e.g., .../transformer_data_win1Mb.pt).")
    args = parser.parse_args()
    run_data_preparation(args)