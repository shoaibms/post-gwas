"""
Data Loader for GWAS Manuscript Figures
========================================
Flexible column mapping with clear error messages.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
import gzip
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

DEBUG_MODE = False

class GWASDataLoader:
    """Robust data loader with flexible column mapping and diagnostics."""
    
    # Column name aliases - maps expected name to possible variations
    COLUMN_ALIASES = {
        # Gene identifiers
        'gene': ['gene', 'gene_id', 'gene_name', 'symbol', 'Gene', 'GENE'],
        
        # Classification labels
        'label': ['label', 'class', 'category', 'classification', 'gene_class'],
        
        # Model metrics
        'h_statistic': ['h_statistic', 'H', 'h', 'hscore', 'h_stat', 'H-statistic',
                        'interaction_strength', 'h_score'],
        'gxe_variance_fraction': ['gxe_variance_fraction', 'gxe_fraction', 'gxe_var_fraction',
                                  'gxe_var', 'GxE_fraction', 'gxe_variance', 'var_gxe'],
        'ws2_attention': ['ws2_attention', 'attention_ws2', 'attn_ws2', 'WS2_attention',
                         'attention_WS2', 'ws2_attn'],
        'delta_r2': ['delta_r2', 'deltaR2', 'ΔR2', 'delta_r_squared', 'r2_improvement'],
        'r2_add': ['r2_add', 'r2_additive', 'r_squared_additive', 'additive_r2'],
        
        # SNP info
        'snp_id': ['snp_id', 'SNP', 'snp', 'variant_id', 'rsid'],
        'chr': ['chr', 'chromosome', 'chrom', 'Chr', 'CHR'],
        'pos': ['pos', 'position', 'bp', 'POS', 'Position'],
        'consequence': ['consequence', 'annotation', 'effect', 'variant_class'],
        'distance_to_tss': ['distance_to_tss', 'dist_tss', 'tss_distance', 'distance_TSS'],
        'importance_score': ['importance_score', 'importance', 'weight', 'effect_size'],
        
        # TF binding
        'tf_overlap': ['tf_overlap', 'TF_overlap', 'has_tf', 'overlaps_tf'],
        'distance_to_tf': ['distance_to_tf', 'tf_distance', 'dist_to_tf'],
        
        # ieQTL
        'q_value': ['q_value', 'qvalue', 'FDR', 'fdr', 'padj', 'adjusted_p'],
        'p_value': ['p_value', 'pvalue', 'P', 'p', 'pval'],
        'beta_gxe': ['beta_gxe', 'βGxE', 'effect_gxe', 'gxe_effect']
    }
    
    def __init__(self, base_dir: str = r"C:\Users\ms\Desktop\gwas"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "output"
        self.data_dir = self.base_dir / "data" / "maize"
        self._cache = {}
        
        logger.info(f"GWAS Data Loader initialized: {self.base_dir}")
        if DEBUG_MODE:
            logger.info("DEBUG MODE ENABLED - Will show detailed diagnostics")
    
    def _map_columns(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Map actual column names to expected names using aliases.
        
        Args:
            df: DataFrame to map
            context: Description for error messages
        
        Returns:
            DataFrame with standardized column names
        """
        if DEBUG_MODE:
            logger.debug(f"[{context}] Original columns: {list(df.columns)}")
        
        rename_map = {}
        
        for expected, aliases in self.COLUMN_ALIASES.items():
            for actual_col in df.columns:
                if actual_col in aliases:
                    rename_map[actual_col] = expected
                    break
        
        if rename_map:
            df = df.rename(columns=rename_map)
            if DEBUG_MODE:
                logger.debug(f"[{context}] Mapped columns: {rename_map}")
        
        return df
    
    def _diagnose_missing_columns(self, df: pd.DataFrame, required: Set[str], 
                                  context: str, file_path: Path):
        """Print helpful diagnostic info when columns are missing."""
        missing = required - set(df.columns)
        
        if not missing:
            return

        print("\n" + "="*70)
        print(f"COLUMN MISMATCH in {context}")
        print("="*70)
        print(f"\n  File: {file_path}")
        print(f"\n  Missing columns: {missing}")
        print(f"\n  Available columns: {list(df.columns)}")

        print("\n  Possible solutions:")
        for missing_col in missing:
            if missing_col in self.COLUMN_ALIASES:
                aliases = self.COLUMN_ALIASES[missing_col]
                matches = [col for col in df.columns if col.lower() in [a.lower() for a in aliases]]
                if matches:
                    print(f"    '{missing_col}' might be: {matches}")
                else:
                    print(f"    '{missing_col}' aliases: {aliases[:5]}... (not found)")

        print("\n  How to fix:")
        print(f"    Option 1: Add missing columns to {file_path.name}")
        print(f"    Option 2: Update COLUMN_ALIASES in data_loader if column exists with different name")
        print(f"    Option 3: Make column optional (use np.nan if not critical)")
        print("="*70 + "\n")
    
    # ===================================================================
    # CORE GENE SETS
    # ===================================================================
    
    def load_platinum_modulators(self) -> pd.DataFrame:
        """Load 31 platinum modulator genes."""
        if 'platinum' in self._cache:
            return self._cache['platinum'].copy()
        
        file_path = self.output_dir / "week1_stability" / "platinum_modulator_set.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Platinum modulators file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, "platinum_modulators")
        
        if 'gene' not in df.columns:
            self._diagnose_missing_columns(df, {'gene'}, "platinum_modulators", file_path)
            raise ValueError("Cannot proceed without 'gene' column")
        
        self._cache['platinum'] = df
        logger.info(f"Loaded {len(df)} platinum modulators")
        return df.copy()
    
    def load_decouple_labels(self, window: str = "500kb") -> pd.DataFrame:
        """
        Load gene classifications with flexible column mapping.
        
        Args:
            window: "500kb", "1Mb", or "2Mb"
        """
        cache_key = f'decouple_{window}'
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        window_map = {"500kb": "500kb", "1Mb": "1Mb", "2Mb": "2Mb"}
        file_path = self.output_dir / "ect_alt" / "integrated" / f"decouple_labels_{window_map[window]}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Decouple labels file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        original_cols = set(df.columns)
        
        # Map columns
        df = self._map_columns(df, f"decouple_labels_{window}")
        
        # Check required vs optional columns
        required_cols = {'gene', 'label'}  # Minimum required
        optional_cols = {'h_statistic', 'gxe_variance_fraction', 'ws2_attention', 'delta_r2', 'r2_add'}
        
        missing_required = required_cols - set(df.columns)
        if missing_required:
            self._diagnose_missing_columns(df, required_cols, f"decouple_{window}", file_path)
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Handle optional columns gracefully
        missing_optional = optional_cols - set(df.columns)
        if missing_optional:
            logger.warning(f"[decouple_{window}] Optional columns missing: {missing_optional}")
            logger.warning(f"   These will be set to NaN. Figure quality may be affected.")
            for col in missing_optional:
                df[col] = np.nan
        
        self._cache[cache_key] = df
        logger.info(f"Loaded {len(df)} genes for window {window} (decouple labels).")
        
        if DEBUG_MODE:
            logger.debug(f"  Columns present: {[c for c in optional_cols if c in original_cols]}")
            logger.debug(f"  Columns added as NaN: {[c for c in missing_optional]}")
        
        return df.copy()
    
    def load_window_stability_metrics(self) -> pd.DataFrame:
        """Load cross-window stability metrics."""
        if 'stability' in self._cache:
            return self._cache['stability'].copy()
        
        file_path = self.output_dir / "week1_stability" / "window_stability_metrics.csv"
        
        if not file_path.exists():
            logger.warning(f"Window stability file not found: {file_path}")
            logger.warning("Creating empty DataFrame with expected structure")
            return pd.DataFrame(columns=['metric', 'value', '500kb_1Mb', '500kb_2Mb', '1Mb_2Mb'])
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, "window_stability")
        
        self._cache['stability'] = df
        logger.info(f"Loaded window stability metrics")
        return df.copy()
    
    # ===================================================================
    # SNP DATA
    # ===================================================================
    
    def load_influential_snps(self) -> pd.DataFrame:
        """Load influential SNPs with flexible column handling."""
        if 'snps' in self._cache:
            return self._cache['snps'].copy()
        
        file_path = self.output_dir / "week2_snp_selection" / "top_influential_snps_pragmatic.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"SNPs file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, "influential_snps")
        
        # Required columns
        required_cols = {'snp_id', 'gene', 'chr', 'pos'}
        missing_required = required_cols - set(df.columns)
        
        if missing_required:
            self._diagnose_missing_columns(df, required_cols, "influential_snps", file_path)
            raise ValueError(f"Missing required SNP columns: {missing_required}")
        
        # Optional columns
        optional_cols = {'consequence', 'distance_to_tss', 'importance_score'}
        for col in optional_cols:
            if col not in df.columns:
                logger.warning(f"[SNPs] Missing optional column '{col}', setting to default")
                if col == 'consequence':
                    df[col] = 'unknown'
                else:
                    df[col] = np.nan
        
        self._cache['snps'] = df
        logger.info(f"Loaded {len(df)} influential SNPs")
        return df.copy()
    
    def load_background_snps(self) -> pd.DataFrame:
        """Load background matched SNPs."""
        if 'background' in self._cache:
            return self._cache['background'].copy()
        
        file_path = self.output_dir / "week2_enrichment" / "background_matched_10x.csv"
        
        if not file_path.exists():
            logger.warning(f"Background SNPs not found: {file_path}")
            logger.warning("Returning empty DataFrame")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, "background_snps")
        
        self._cache['background'] = df
        logger.info(f"Loaded {len(df)} background SNPs")
        return df.copy()
    
    # ===================================================================
    # TF BINDING DATA
    # ===================================================================
    
    def load_tf_binding_sites(self) -> pd.DataFrame:
        """Load TF binding peaks from media-10.gz."""
        if 'tf_sites' in self._cache:
            return self._cache['tf_sites'].copy()
        
        file_path = self.data_dir / "media-10.gz"
        
        if not file_path.exists():
            raise FileNotFoundError(f"TF binding file not found: {file_path}")
        
        tf_sites = []
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                try:
                    peak_loc = parts[1]
                    if ':' not in peak_loc:
                        continue
                    
                    chr_part, pos_part = peak_loc.split(':')
                    chr_val = chr_part.replace('chr', '').replace('Chr', '')
                    
                    if chr_val not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                        continue
                    
                    tf_sites.append({
                        'chr': chr_val,
                        'summit': int(pos_part),
                        'TF_name': parts[8],
                        'TF_assay': parts[0],
                        'z_score': float(parts[5]),
                        'acr_overlap': int(parts[6])
                    })
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(tf_sites)
        
        # Apply filters
        df = df[df['acr_overlap'] == 1]
        df = df[df['z_score'] >= 3.0]
        
        # Deduplicate
        df = df.sort_values(['chr', 'summit'])
        df['delta'] = df.groupby('chr')['summit'].diff().fillna(10**9)
        df = df[df['delta'] > 200].drop(columns='delta')
        
        self._cache['tf_sites'] = df
        logger.info(f"Loaded {len(df)} high-quality TF binding sites")
        return df.copy()
    
    def load_tf_proximity_results(self) -> pd.DataFrame:
        """Load TF proximity analysis results."""
        if 'tf_proximity' in self._cache:
            return self._cache['tf_proximity'].copy()
        
        file_path = self.output_dir / "week3_tf_binding" / "snp_tf_proximity_results.csv"
        
        if not file_path.exists():
            logger.warning(f"TF proximity results not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, "tf_proximity")
        
        self._cache['tf_proximity'] = df
        logger.info(f"Loaded TF proximity results")
        return df.copy()
    
    # ===================================================================
    # GO ENRICHMENT
    # ===================================================================
    
    def load_go_enrichment(self, gene_set: str = "modulators") -> pd.DataFrame:
        """Load GO enrichment results."""
        cache_key = f'go_{gene_set}'
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        file_path = self.output_dir / "week4_go_enrichment" / f"go_enrichment_{gene_set}.csv"
        
        if not file_path.exists():
            logger.warning(f"GO enrichment not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, f"go_{gene_set}")
        
        # Filter to significant if q_value exists
        if 'q_value' in df.columns:
            df = df[df['q_value'] < 0.05]
        
        self._cache[cache_key] = df
        logger.info(f"Loaded GO enrichment for {gene_set}")
        return df.copy()
    
    # ===================================================================
    # IEQTL DATA
    # ===================================================================
    
    def load_ieqtl_results(self) -> pd.DataFrame:
        """Load interaction eQTL results."""
        if 'ieqtl' in self._cache:
            return self._cache['ieqtl'].copy()
        
        file_path = self.output_dir / "week5_ieqtl" / "ieqtl_results_complete.csv"
        
        if not file_path.exists():
            logger.warning(f"ieQTL results not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df = self._map_columns(df, "ieqtl")
        
        self._cache['ieqtl'] = df
        logger.info(f"Loaded ieQTL results for {len(df)} gene-SNP pairs")
        return df.copy()
    
    # ===================================================================
    # EXPRESSION DATA
    # ===================================================================
    
    def load_expression_matrix(self, condition: str = "WW") -> pd.DataFrame:
        """Load expression matrix for given condition."""
        cache_key = f'expr_{condition}'
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        file_map = {
            "WW": "WW_209-Uniq_FPKM.agpv4.txt.gz",
            "WS1": "WS1_208-uniq_FPKM.agpv4.txt.gz",
            "WS2": "WS2_210-uniq_FPKM.agpv4.txt.gz"
        }
        
        file_path = self.output_dir / "data_filtered" / file_map[condition]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Expression file not found: {file_path}")
        
        df = pd.read_csv(file_path, sep='\t', compression='gzip', index_col=0)
        
        self._cache[cache_key] = df
        logger.info(f"Loaded expression matrix for {condition}: {df.shape}")
        return df.copy()
    
    def load_expression_for_genes(self, genes: List[str]) -> Dict[str, pd.DataFrame]:
        """Load expression across all conditions for specific genes."""
        result = {}
        for condition in ['WW', 'WS1', 'WS2']:
            expr = self.load_expression_matrix(condition)
            available_genes = [g for g in genes if g in expr.index]
            result[condition] = expr.loc[available_genes]
        
        logger.info(f"Loaded expression for {len(genes)} genes across 3 conditions")
        return result
    
    # ===================================================================
    # GENOME ANNOTATION
    # ===================================================================
    
    def load_gene_positions(self) -> pd.DataFrame:
        """Load gene positions from GFF3."""
        if 'gene_positions' in self._cache:
            return self._cache['gene_positions'].copy()
        
        file_path = self.data_dir / "Zea_mays.B73_RefGen_v4.gff3.gz"
        
        if not file_path.exists():
            raise FileNotFoundError(f"GFF3 file not found: {file_path}")
        
        genes = []
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene':
                    continue
                
                attrs = dict(item.split('=') for item in parts[8].split(';') if '=' in item)
                
                if 'ID' not in attrs:
                    continue
                
                gene_id = attrs['ID'].replace('gene:', '')
                
                genes.append({
                    'gene_id': gene_id,
                    'chr': parts[0].replace('chr', '').replace('Chr', ''),
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'strand': parts[6]
                })
        
        df = pd.DataFrame(genes)
        df = df[df['chr'].isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])]
        df['tss'] = df.apply(lambda row: row['start'] if row['strand'] == '+' else row['end'], axis=1)
        
        self._cache['gene_positions'] = df
        logger.info(f"Loaded positions for {len(df)} genes")
        return df.copy()
    
    # ===================================================================
    # FIGURE-SPECIFIC BUNDLES
    # ===================================================================
    
    def load_figure1_data(self) -> Dict:
        """Bundle all data for Figure 1."""
        return {
            'platinum': self.load_platinum_modulators(),
            'decouple_500kb': self.load_decouple_labels("500kb"),
            'decouple_1Mb': self.load_decouple_labels("1Mb"),
            'decouple_2Mb': self.load_decouple_labels("2Mb"),
            'stability': self.load_window_stability_metrics()
        }
    
    def load_figure2_data(self) -> Dict:
        """Bundle all data for Figure 2."""
        return {
            'snps': self.load_influential_snps(),
            'background': self.load_background_snps(),
            'tf_sites': self.load_tf_binding_sites(),
            'tf_proximity': self.load_tf_proximity_results(),
            'gene_positions': self.load_gene_positions(),
            'platinum': self.load_platinum_modulators()
        }
    
    # ===================================================================
    # UTILITIES
    # ===================================================================
    
    def validate_all_files_exist(self) -> Dict[str, bool]:
        """Check which expected files exist."""
        checks = {
            'platinum_modulators': (self.output_dir / "week1_stability" / "platinum_modulator_set.csv").exists(),
            'influential_snps': (self.output_dir / "week2_snp_selection" / "top_influential_snps_pragmatic.csv").exists(),
            'tf_binding': (self.data_dir / "media-10.gz").exists(),
            'ieqtl_results': (self.output_dir / "week5_ieqtl" / "ieqtl_results_complete.csv").exists(),
            'expression_ww': (self.output_dir / "data_filtered" / "WW_209-Uniq_FPKM.agpv4.txt.gz").exists(),
            'gff3': (self.data_dir / "Zea_mays.B73_RefGen_v4.gff3.gz").exists()
        }
        
        missing = [k for k, v in checks.items() if not v]
        
        if missing:
            logger.warning(f"Missing files: {missing}")
        else:
            logger.info("All expected files present")
        
        return checks
    
    def clear_cache(self):
        """Clear cached data."""
        self._cache = {}
        logger.info("Cache cleared")


# ========================================================================
# GLOBAL INSTANCE
# ========================================================================

data_loader = GWASDataLoader()


# ========================================================================
# TESTING
# ========================================================================

if __name__ == "__main__":
    print("="*60)
    print("DATA LOADER TEST SUITE")
    print("="*60)
    
    loader = GWASDataLoader()
    
    print("\n1. Checking file availability...")
    checks = loader.validate_all_files_exist()
    for name, exists in checks.items():
        status = "OK" if exists else "MISSING"
        print(f"   [{status}] {name}")
    
    print("\n2. Testing core data loads...")
    try:
        plat = loader.load_platinum_modulators()
        print(f"   [OK] Platinum: {len(plat)} genes")
    except Exception as e:
        print(f"   [FAIL] Platinum: {e}")

    try:
        snps = loader.load_influential_snps()
        print(f"   [OK] SNPs: {len(snps)} variants")
    except Exception as e:
        print(f"   [FAIL] SNPs: {e}")

    try:
        tf = loader.load_tf_binding_sites()
        print(f"   [OK] TF sites: {len(tf)} peaks")
    except Exception as e:
        print(f"   [FAIL] TF sites: {e}")
    
    print("\n3. Testing figure bundles...")
    try:
        fig1_data = loader.load_figure1_data()
        print(f"   [OK] Figure 1: {len(fig1_data)} data components")
    except Exception as e:
        print(f"   [FAIL] Figure 1: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
