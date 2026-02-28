#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Delta-model ieQTL discovery.

Tests: delta_expr (WS2 - WW) ~ G (+ genotype PCs) with robust SNP mapping,
per-SNP genotype QC, SNP-level and gene-level BH FDR, and lambdaGC
computed on upper-tail p-values.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from scipy.stats import chi2

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUT  = BASE / "output" / "week6_ieqtl"
OUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("DELTA-MODEL QC: expr_WS2 - expr_WW ~ G (+ PCs)")
print("="*60)

# ---------------------------------------------------------------------
# 1) Load expression (long) and build delta (WS2 - WW) per line x gene
# ---------------------------------------------------------------------
long = pd.read_csv(OUT / "expr_long_ww_ws2.csv")  # columns: gene,line,env,expr
req_cols = {"gene", "line", "env", "expr"}
if not req_cols.issubset(long.columns):
    raise SystemExit(f"expr_long_ww_ws2.csv missing columns: {req_cols - set(long.columns)}")

ww  = long[long.env == "WW"].pivot(index="line", columns="gene", values="expr")
ws2 = long[long.env == "WS2"].pivot(index="line", columns="gene", values="expr")
common_lines = ww.index.intersection(ws2.index)
ww  = ww.loc[common_lines]
ws2 = ws2.loc[common_lines]
delta = (ws2 - ww)  # delta expression
print(f"Delta matrix: {delta.shape[0]} lines x {delta.shape[1]} genes")

# ---------------------------------------------------------------------
# 2) Load genotype matrix, SNP meta, platinum genes
# ---------------------------------------------------------------------
geno = pd.read_csv(BASE / "output" / "week5_nrs" / "genotype_matrix.csv", index_col=0)  # lines x SNPs
geno = geno.loc[common_lines]  # align to expression lines
print(f"Genotype matrix aligned: {geno.shape[0]} lines x {geno.shape[1]} SNPs")

meta = pd.read_csv(BASE / "output" / "week2_snp_selection" / "top_influential_snps_pragmatic.csv")
if "gene" not in meta.columns:
    raise SystemExit("SNP meta must contain 'gene' column.")

plat_genes = pd.read_csv(BASE / "output" / "week1_stability" / "platinum_modulator_set.csv")["gene"].unique()
genes = [g for g in plat_genes if g in delta.columns]
print(f"Platinum genes in delta: {len(genes)}")

# ---------------------------------------------------------------------
# 3) Robust mapping: genotype columns <-> (chr,pos); meta -> (chr,pos)
# ---------------------------------------------------------------------
def parse_colname(c):
    """
    Parse genotype col like '8_28568160' or '8:28568160' -> ('8', 28568160)
    Returns None if not parseable.
    """
    m = re.match(r"^(\d+)[_:](\d+)$", str(c))
    if m:
        return (m.group(1), int(m.group(2)))
    return None

# map (chr,pos) -> actual genotype column name
col_map = {}
for c in geno.columns:
    key = parse_colname(c)
    if key:
        col_map[key] = c

# normalize meta to provide chr/pos
meta = meta.copy()
if {"chr", "pos"}.issubset(meta.columns):
    meta["chr"] = meta["chr"].astype(str).str.replace("^chr", "", regex=True)
    meta["pos"] = meta["pos"].astype(int)
elif "snp_id" in meta.columns:
    # derive chr/pos from '8:28568160' or '8_28568160'
    snp_txt = meta["snp_id"].astype(str).str.replace("_", ":")
    parts = snp_txt.str.split(":", expand=True)
    if parts.shape[1] != 2:
        raise SystemExit("Cannot parse snp_id into chr:pos.")
    meta["chr"] = parts[0].str.replace("^chr", "", regex=True)
    meta["pos"] = parts[1].astype(int)
else:
    raise SystemExit("SNP meta must have (chr,pos) or 'snp_id'.")

# attach matching genotype column name
meta["geno_col"] = meta.apply(lambda r: col_map.get((str(r["chr"]), int(r["pos"]))), axis=1)
mapped = meta.dropna(subset=["geno_col"]).copy()
print(f"Meta SNPs mapped to genotype columns: {len(mapped)}/{len(meta)}")

# helpful lookup for readable IDs
if "snp_id" in mapped.columns:
    id_lookup = dict(zip(mapped["geno_col"], mapped["snp_id"]))
else:
    id_lookup = dict(zip(mapped["geno_col"], mapped["geno_col"]))

# per-gene genotype column list
gene2snps = mapped.groupby("gene")["geno_col"].apply(list).to_dict()

# ---------------------------------------------------------------------
# 4) Genotype PCs (optional but helpful for lambdaGC)
# ---------------------------------------------------------------------
Xg = geno.copy()
Xg = Xg.apply(lambda col: col.fillna(col.mean()), axis=0)
# standardize
Xstd = (Xg - Xg.mean()) / Xg.std(ddof=0)
ncols = Xstd.shape[1]
k = int(min(5, max(0, ncols - 1)))
if k >= 1:
    pca = PCA(n_components=k, svd_solver="full", random_state=42)
    PCS = pd.DataFrame(pca.fit_transform(Xstd), index=Xstd.index, columns=[f"PC{i+1}" for i in range(k)])
    print(f"Computed PCs: {k} (explained var first 3 ~ {pca.explained_variance_ratio_[:3].sum():.2f})")
else:
    PCS = pd.DataFrame(index=Xstd.index)
    print("Computed PCs: 0 (not enough SNPs to compute PCs)")

# ---------------------------------------------------------------------
# 5) Test delta_expr ~ G (+ PCs) with HC3 robust SE
# ---------------------------------------------------------------------
rows = []
diag = []

def good_genotype_spread(gseries, min_per_class=8, min_lines=80):
    # require at least two genotype classes with min_per_class lines
    counts = gseries.groupby(gseries).size()
    if (counts >= min_per_class).sum() < 2:
        return False, counts
    if gseries.shape[0] < min_lines:
        return False, counts
    return True, counts

for gene in genes:
    if gene not in delta.columns:
        continue
    y = delta[gene].dropna()
    L = y.index

    for snp_col in gene2snps.get(gene, []):
        if snp_col not in geno.columns:
            continue
        g = geno.loc[L, snp_col].astype(float)

        # Build analysis DataFrame with PCs
        df = pd.concat([y.rename("dy"), g.rename("G"), PCS.loc[L]], axis=1).dropna()

        ok, counts = good_genotype_spread(df["G"])
        if not ok or df["G"].var() == 0:
            diag.append({
                "gene": gene, "snp": id_lookup.get(snp_col, snp_col),
                "n": df.shape[0],
                "n_G0": int(counts.get(0, 0)), "n_G1": int(counts.get(1, 0)), "n_G2": int(counts.get(2, 0)),
                "reason": "insufficient_genotype_spread_or_variance"
            })
            continue

        # Prepare design matrix: intercept + G + PCs
        X = pd.concat([df[["G"]], df.filter(like="PC")], axis=1)
        X = sm.add_constant(X, has_constant="add")

        try:
            fit = sm.OLS(df["dy"], X).fit(cov_type="HC3")
            beta = fit.params.get("G", np.nan)
            se   = fit.bse.get("G", np.nan)
            pval = fit.pvalues.get("G", np.nan)
        except Exception as e:
            diag.append({
                "gene": gene, "snp": id_lookup.get(snp_col, snp_col),
                "n": df.shape[0],
                "n_G0": int(counts.get(0, 0)), "n_G1": int(counts.get(1, 0)), "n_G2": int(counts.get(2, 0)),
                "reason": f"fit_error:{str(e)[:80]}"
            })
            continue

        maf = df["G"].mean() / 2.0  # assumes 0/1/2 dosage
        rows.append({
            "gene": gene,
            "snp": id_lookup.get(snp_col, snp_col),
            "geno_col": snp_col,
            "beta_G": beta,
            "se_G": se,
            "p_G": pval,
            "maf": maf,
            "n": df.shape[0],
            "n_G0": int(counts.get(0, 0)), "n_G1": int(counts.get(1, 0)), "n_G2": int(counts.get(2, 0))
        })

res = pd.DataFrame(rows)
diag = pd.DataFrame(diag)

print(f"Tests run: {len(res)}")

# Save diagnostics early
diag.to_csv(OUT / "ieqtl_delta_diagnostics.csv", index=False)

if len(res) == 0:
    print("No tests ran -- likely mapping or QC prevented all models. "
          "Check 'ieqtl_delta_diagnostics.csv' and mapping stats.")
    # still write empty result shell
    res.to_csv(OUT / "ieqtl_delta_results.csv", index=False)
    raise SystemExit(0)

# ---------------------------------------------------------------------
# 6) FDR: SNP-level and gene-level (minP per gene)
# ---------------------------------------------------------------------
res["q_G"] = multipletests(res["p_G"].values, method="fdr_bh")[1]

gmin = res.groupby("gene")["p_G"].min()
gq   = multipletests(gmin.values, method="fdr_bh")[1]
gene_fdr = pd.DataFrame({"gene": gmin.index, "p_min": gmin.values, "q_gene": gq})

# ---------------------------------------------------------------------
# 7) lambdaGC on upper-tail p-values
# ---------------------------------------------------------------------
p = res["p_G"].values
mask = p > 0.1
p_tail = p[mask] if mask.sum() >= 20 else p  # fallback if too few
chi = chi2.isf(p_tail, 1)
lam = np.median(chi) / chi2.ppf(0.5, 1)
print(f"lambdaGC(>0.1 tail): {lam:.2f}")

# ---------------------------------------------------------------------
# 8) Save outputs
# ---------------------------------------------------------------------
res.sort_values(["gene", "p_G"]).to_csv(OUT / "ieqtl_delta_results.csv", index=False)

sig_snp = res[res["q_G"] <= 0.10].sort_values("p_G")
sig_snp.to_csv(OUT / "ieqtl_delta_sig_snps_q10.csv", index=False)

gene_fdr.sort_values("q_gene").to_csv(OUT / "ieqtl_delta_gene_fdr.csv", index=False)

print(f"SNP q<=0.10: {len(sig_snp)} across genes {sig_snp['gene'].nunique()}")
print(f"Gene q<=0.10: {(gene_fdr['q_gene']<=0.10).sum()} / {len(gene_fdr)}")

# ---------------------------------------------------------------------
# 9) Decision and guidance
# ---------------------------------------------------------------------
n_gene_q10 = int((gene_fdr["q_gene"] <= 0.10).sum())
decision = "USE_DELTA_SET" if (n_gene_q10 >= 10 and lam <= 1.20) else "USE_INTERSECTION"
print("\nDecision rule:")
print(f"  Genes (q_gene<=0.10) = {n_gene_q10}")
print(f"  lambdaGC(>0.1 tail) = {lam:.2f}")
print(f"  => {decision}")
with open(OUT / "ieqtl_delta_decision.txt", "w", encoding="utf-8") as f:
    f.write(f"genes_q10={n_gene_q10}\n")
    f.write(f"lambda_gc={lam:.3f}\n")
    f.write(f"decision={decision}\n")

print("\nDone.")
