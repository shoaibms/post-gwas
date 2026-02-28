"""
Create "literature" gene lists from differential expression analysis.

Performs WS2 vs WW paired t-test to identify drought-responsive genes,
then saves them as reference lists for enrichment testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

BASE = Path(r"C:\Users\ms\Desktop\gwas")
FILTERED = BASE / "output" / "data_filtered"
LIT_DIR = BASE / "data" / "maize" / "external" / "literature"
LIT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("CREATING LITERATURE GENE LISTS FROM DE ANALYSIS")
print("="*60)

print("\n1. Loading expression data...")

def load_expression(filename):
    """Load and transpose expression data."""
    df = pd.read_csv(FILTERED / filename, sep="\t", compression="gzip", index_col=0)
    return df.T  # Transpose: rows=genes, cols=samples

ww = load_expression("WW_209-Uniq_FPKM.agpv4.txt.gz")
ws2 = load_expression("WS2_210-uniq_FPKM.agpv4.txt.gz")

print(f"  WW: {ww.shape[0]} genes x {ww.shape[1]} samples")
print(f"  WS2: {ws2.shape[0]} genes x {ws2.shape[1]} samples")

print("\n2. Aligning samples and genes...")

common_samples = ww.columns.intersection(ws2.columns)
ww = ww[common_samples]
ws2 = ws2[common_samples]

common_genes = ww.index.intersection(ws2.index)
ww = ww.loc[common_genes]
ws2 = ws2.loc[common_genes]

print(f"  Common: {len(common_genes)} genes x {len(common_samples)} samples")

print("\n3. Running paired t-test (WS2 vs WW)...")

# Log transform: log2(1 + FPKM)
x = np.log2(1 + ww.values)
y = np.log2(1 + ws2.values)

# Paired t-test across samples (axis=1)
t_stats, p_values = ttest_rel(y, x, axis=1, nan_policy='omit')

# Log2 fold change
log2fc = y.mean(axis=1) - x.mean(axis=1)

# Variance filter to avoid degenerate tests
keep = (ww.var(axis=1) > 1e-6) & (ws2.var(axis=1) > 1e-6)
p = pd.Series(p_values, index=common_genes)
p = p[keep].dropna()

# FDR correction only on valid p-values
q = pd.Series(np.nan, index=common_genes)
q.loc[p.index] = multipletests(p.values, method='fdr_bh')[1]

print(f"  Computed statistics for {len(common_genes)} genes")
print(f"  Variance filtered: {keep.sum()} genes kept, {(~keep).sum()} low-variance removed")

print("\n4. Identifying DE genes...")

log2fc = pd.Series(log2fc, index=common_genes)

results = pd.DataFrame({
    'gene': common_genes,
    'log2FC': log2fc,
    't_stat': t_stats,
    'p': p_values,
    'q': q
})

# Define gene sets with milder effect size threshold
upregulated = q.index[(q < 0.05) & (log2fc > 0.25)]
downregulated = q.index[(q < 0.05) & (log2fc < -0.25)]
all_de = q.index[q < 0.05]

print(f"  Upregulated (q<0.05, log2FC>0.25): {len(upregulated)}")
print(f"  Downregulated (q<0.05, log2FC<-0.25): {len(downregulated)}")
print(f"  All DE (q<0.05): {len(all_de)}")

print("\n5. Saving gene lists...")

# Convert Index to Series for saving
pd.Series(upregulated).drop_duplicates().to_csv(LIT_DIR / "WS2_up_AGPv4.txt", 
                                                 index=False, header=False)
pd.Series(downregulated).drop_duplicates().to_csv(LIT_DIR / "WS2_down_AGPv4.txt",
                                                   index=False, header=False)
pd.Series(all_de).drop_duplicates().to_csv(LIT_DIR / "WS2_allDE_AGPv4.txt",
                                            index=False, header=False)

print(f"  Saved to: {LIT_DIR}")
print(f"    - WS2_up_AGPv4.txt ({len(upregulated)} genes)")
print(f"    - WS2_down_AGPv4.txt ({len(downregulated)} genes)")
print(f"    - WS2_allDE_AGPv4.txt ({len(all_de)} genes)")

print("\n6. Summary statistics...")

print(f"\n  Mean log2FC (all genes): {results['log2FC'].mean():.3f}")
print(f"  Median p-value: {results['p'].median():.2e}")
print(f"  Significant genes (q<0.05): {(results['q'] < 0.05).sum()}")

# Top upregulated
print("\n  Top 5 upregulated genes:")
top_up = results.nlargest(5, 'log2FC')[['gene', 'log2FC', 'q']]
for _, row in top_up.iterrows():
    print(f"    {row['gene']}: log2FC={row['log2FC']:.2f}, q={row['q']:.2e}")

# Top downregulated
print("\n  Top 5 downregulated genes:")
top_down = results.nsmallest(5, 'log2FC')[['gene', 'log2FC', 'q']]
for _, row in top_down.iterrows():
    print(f"    {row['gene']}: log2FC={row['log2FC']:.2f}, q={row['q']:.2e}")

# Save full results for reference
results_file = LIT_DIR / "DE_analysis_results.csv"
results.to_csv(results_file, index=False)
print(f"\n  Full results saved: {results_file}")

print("\n" + "="*60)
print("LITERATURE LISTS CREATED")
print("="*60)

print("\nNOTE: These lists are from your own data (WS2 vs WW).")
print("Testing for enrichment will show if your 31 platinum modulators")
print("are enriched among drought-responsive genes.")
print("\nThis is somewhat circular but still informative:")
print("  - Tests if modulators are enriched in DE genes")
print("  - Different analysis method (t-test vs your models)")
print("  - Validates biological relevance")

print("\n" + "="*60)
print("NEXT STEP")
print("="*60)
print("\nRun Week 5 workflow to test enrichment:")
print("  python code\\postgwas\\week5_option_b_complete.py")