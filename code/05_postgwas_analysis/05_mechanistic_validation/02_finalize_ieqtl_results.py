#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalize ieQTL analysis results from the delta-model (v2).

Filters by gene-level and SNP-level q<=0.10, adds genomic coordinates,
MAF and direction, writes CSV + BED, optionally overlaps with Week 3
filtered TF peaks, and runs Fisher's exact test for TF overlap enrichment.
"""

import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUT  = BASE / "output" / "week6_ieqtl"
OUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("FINALIZING ieQTL RESULTS")
print("="*60)

# -----------------------------
# 1) Load delta-model results
# -----------------------------
print("\n1. Loading delta-model results...")
res_delta = pd.read_csv(OUT / "ieqtl_delta_results.csv")       # has columns incl. gene, snp, p_G, q_G, n_G0..2
gene_fdr  = pd.read_csv(OUT / "ieqtl_delta_gene_fdr.csv")      # has columns: gene, p_min, q_gene

print(f"   Total tests: {len(res_delta)}")
print(f"   Genes tested: {res_delta['gene'].nunique()}")

# Keep genes with q_gene <= 0.10
sig_genes = set(gene_fdr.loc[gene_fdr["q_gene"] <= 0.10, "gene"])
print(f"\n2. Genes with q_gene <= 0.10: {len(sig_genes)}")

# Keep SNPs with q_G <= 0.10 within those genes
final_snps = res_delta[(res_delta["gene"].isin(sig_genes)) & (res_delta["q_G"] <= 0.10)].copy()
print(f"   SNPs with q <= 0.10: {len(final_snps)}")

# -----------------------------
# 2) Add genomic coordinates
# -----------------------------
print("\n3. Adding genomic coordinates...")
meta = pd.read_csv(BASE / "output" / "week2_snp_selection" / "top_influential_snps_pragmatic.csv")

# Ensure chr/pos present; if not, derive from snp_id
meta = meta.copy()
if not {"chr","pos"}.issubset(meta.columns):
    if "snp_id" not in meta.columns:
        raise SystemExit("Need 'chr'+'pos' or 'snp_id' in top_influential_snps_pragmatic.csv")
    tmp = meta["snp_id"].astype(str).str.replace("_",":")
    parts = tmp.str.split(":", expand=True)
    if parts.shape[1] != 2:
        raise SystemExit("Cannot parse snp_id into chr:pos")
    meta["chr"] = parts[0].str.replace("^chr","",regex=True)
    meta["pos"] = parts[1].astype(int)

coord_cols = ["gene","snp_id","chr","pos","ref","alt"]
meta_coords = meta[[c for c in coord_cols if c in meta.columns]].copy()

final = final_snps.merge(meta_coords, left_on=["gene","snp"], right_on=["gene","snp_id"], how="left")
final["chr"] = final["chr"].astype(str).str.replace("^chr","",regex=True)
final["pos"] = final["pos"].astype("Int64")

matched = final["chr"].notna().sum()
print(f"   Matched coordinates: {matched}/{len(final)}")

# -----------------------------
# 3) Compute MAF & direction
# -----------------------------
print("\n4. Calculating MAF...")

def calc_maf(row):
    n0 = int(row.get("n_G0", 0) or 0)
    n1 = int(row.get("n_G1", 0) or 0)
    n2 = int(row.get("n_G2", 0) or 0)
    total = n0 + n1 + n2
    if total == 0: return None
    return (n1 + 2*n2) / (2*total)

final["maf"] = final.apply(calc_maf, axis=1)
final["effect_direction"] = final["beta_G"].apply(lambda b: "WS2>WW" if b > 0 else "WW>WS2")

# -----------------------------
# 5) Save main CSV + gene summary
# -----------------------------
print("\n5. Saving results...")

cols = [
    "gene","snp","chr","pos","ref","alt",
    "beta_G","se_G","p_G","q_G","effect_direction","maf",
    "n","n_G0","n_G1","n_G2"
]
final = final[[c for c in cols if c in final.columns]].sort_values("p_G")

main_file = OUT / "ieqtl_final_results.csv"
final.to_csv(main_file, index=False)
print(f"   Main results: {main_file}")

gene_summary = (final.groupby("gene")
                     .agg(n_snps_sig=("snp","count"),
                          p_min=("p_G","min"),
                          q_min=("q_G","min"))
                     .reset_index())
gene_summary = gene_summary.merge(gene_fdr[["gene","q_gene"]], on="gene", how="left").sort_values("q_gene")

gene_file = OUT / "ieqtl_gene_summary.csv"
gene_summary.to_csv(gene_file, index=False)
print(f"   Gene summary: {gene_file}")

# -----------------------------
# 6) Write BED (0-based, 1-bp)
# -----------------------------
print("\n6. Creating BED file...")
bed = final.dropna(subset=["chr","pos"]).copy()
bed["start"] = bed["pos"].astype(int) - 1
bed["end"]   = bed["pos"].astype(int)
bed["name"]  = bed["snp"].astype(str)
bed_out = bed[["chr","start","end","name","gene"]]
bed_file = OUT / "ieqtl_final_snps.bed"
bed_out.to_csv(bed_file, sep="\t", header=False, index=False)
print(f"   BED file: {bed_file}")

# -----------------------------
# 7) Optional: TF-peak overlap
# -----------------------------
print("\n7. Checking TF peak overlaps...")
tf_bed = BASE / "output" / "week3_tf_binding" / "filtered_tf_sites.bed"

# Always predefine to avoid NameError
overlaps = []
snps_in_peaks = pd.DataFrame()

if tf_bed.exists():
    peaks = pd.read_csv(tf_bed, sep="\t", header=None)
    if peaks.shape[1] < 3:
        print("   TF bed has <3 columns; skipping.")
    else:
        cols = ["chr","start","end"] + [f"col{i}" for i in range(3, peaks.shape[1])]
        peaks.columns = cols
        peaks["chr"] = peaks["chr"].astype(str).str.replace("^chr","",regex=True)

        # quick overlap by chromosome; 16 SNPs x ~13k peaks is fine to loop
        for _, s in bed.iterrows():
            chr_peaks = peaks[peaks["chr"] == str(s["chr"])]
            hits = chr_peaks[(chr_peaks["start"] <= s["pos"]) & (s["pos"] < chr_peaks["end"])]
            if len(hits) > 0:
                # collect TF labels if present
                tf_cols = [c for c in hits.columns if c.startswith("col")]
                tfs = []
                for c in tf_cols:
                    tfs += list(hits[c].dropna().astype(str).unique())
                tfs = [t for t in tfs if t and t.lower() != "nan"]
                overlaps.append({
                    "gene": s["gene"],
                    "snp": s["name"],
                    "chr": s["chr"],
                    "pos": int(s["pos"]),
                    "n_tf_peaks": int(len(hits)),
                    "tf_names": ";".join(sorted(set(tfs))) if tfs else ""
                })

        if overlaps:
            snps_in_peaks = pd.DataFrame(overlaps)
            tf_out = OUT / "ieqtl_snps_overlapping_tf_peaks.csv"
            snps_in_peaks.to_csv(tf_out, index=False)
            print(f"   SNPs overlapping TF peaks: {len(snps_in_peaks)}/{len(final)}")
            print(f"   Saved: {tf_out}")

            # ---------------------------------------------------------
            # 7b) Fisher's exact test: ieQTL lead SNP TF-overlap
            #     enrichment vs matched background
            # ---------------------------------------------------------
            print("\n   --- 7b) Fisher's exact: ieQTL TF overlap ---")

            n_ieqtl_overlap = len(snps_in_peaks)   # e.g. 9
            n_ieqtl_total   = len(bed)              # e.g. 16

            # Compute TF overlap for matched background (930 SNPs)
            bg_file = BASE / "output" / "week2_enrichment" / "background_matched_10x.csv"
            if bg_file.exists():
                bg_snps = pd.read_csv(bg_file)

                # Background file uses bg_chr/bg_pos/bg_snp_id naming
                rename_map = {c: c.replace("bg_","") for c in bg_snps.columns if c.startswith("bg_")}
                if rename_map:
                    bg_snps = bg_snps.rename(columns=rename_map)

                # Fallback: parse from snp_id if chr/pos still missing
                if not {"chr","pos"}.issubset(bg_snps.columns):
                    if "snp_id" in bg_snps.columns:
                        tmp = bg_snps["snp_id"].astype(str).str.replace("_",":")
                        parts = tmp.str.split(":", expand=True)
                        bg_snps["chr"] = parts[0].str.replace("^chr","",regex=True)
                        bg_snps["pos"] = parts[1].astype(int)
                bg_snps["chr"] = bg_snps["chr"].astype(str).str.replace("^chr","",regex=True)
                bg_snps["pos"] = bg_snps["pos"].astype(int)

                # Deduplicate: file has one row per fg-bg pair; keep unique bg positions
                bg_snps = bg_snps.drop_duplicates(subset=["chr","pos"])

                # Same BED overlap loop as ieQTL SNPs above
                bg_overlap_count = 0
                bg_total_tested  = 0
                for _, row in bg_snps.iterrows():
                    row_chr = str(row.get("chr", ""))
                    row_pos = row.get("pos", None)
                    if not row_chr or pd.isna(row_pos):
                        continue
                    row_pos = int(row_pos)
                    bg_total_tested += 1
                    chr_pk = peaks[peaks["chr"] == row_chr]
                    if chr_pk[(chr_pk["start"] <= row_pos) & (row_pos < chr_pk["end"])].shape[0] > 0:
                        bg_overlap_count += 1

                bg_no_overlap = bg_total_tested - bg_overlap_count

                print(f"   ieQTL lead SNPs:  {n_ieqtl_overlap}/{n_ieqtl_total}"
                      f" ({100*n_ieqtl_overlap/n_ieqtl_total:.1f}%) overlap TF peaks")
                print(f"   Background SNPs:  {bg_overlap_count}/{bg_total_tested}"
                      f" ({100*bg_overlap_count/bg_total_tested:.1f}%) overlap TF peaks")

                # 2x2 table: [ieQTL overlap, ieQTL no-overlap]
                #             [bg overlap,    bg no-overlap   ]
                table = [
                    [n_ieqtl_overlap, n_ieqtl_total  - n_ieqtl_overlap],
                    [bg_overlap_count, bg_no_overlap]
                ]
                odds_ratio, p_fisher = fisher_exact(table, alternative="greater")

                print(f"   Contingency table: {table}")
                print(f"   Fisher's exact (one-sided): OR = {odds_ratio:.2f}, p = {p_fisher:.4g}")

                # Save to CSV for reproducibility
                fisher_results = pd.DataFrame([{
                    "n_ieqtl_total":   n_ieqtl_total,
                    "n_ieqtl_overlap": n_ieqtl_overlap,
                    "n_bg_total":      bg_total_tested,
                    "n_bg_overlap":    bg_overlap_count,
                    "odds_ratio":      float(odds_ratio),
                    "p_fisher_one_sided": float(p_fisher),
                }])
                fisher_file = OUT / "ieqtl_tf_overlap_fisher.csv"
                fisher_results.to_csv(fisher_file, index=False)
                print(f"   Saved: {fisher_file}")
            else:
                print(f"   Background file not found: {bg_file}")
                print("   Skipping Fisher test (no background to compare)")

        else:
            print("   No SNPs overlap TF peaks")
else:
    print(f"   TF peak file not found: {tf_bed}")
    print("   Skipping TF overlap check")

# -----------------------------
# Final summary
# -----------------------------
print("\n" + "="*60)
print("FINALIZATION COMPLETE")
print("="*60)
print(f"\nFinal ieQTL set (delta-model):")
print(f"  - Genes: {len(sig_genes)} (q_gene <= 0.10)")
print(f"  - SNPs: {len(final)} (q_SNP <= 0.10)")

print("\nKey files:")
print(f"  1) {main_file}")
print(f"  2) {gene_file}")
print(f"  3) {bed_file}")
if not snps_in_peaks.empty:
    print("  4) ieqtl_snps_overlapping_tf_peaks.csv (for motif delta-PWM)")
    print("  5) ieqtl_tf_overlap_fisher.csv (Fisher's exact test)")
else:
    print("  4) (no TF overlap file written)")

print("\n" + "="*60)
print("READY FOR MOTIF ANALYSIS")
print("="*60)
if snps_in_peaks.empty:
    print("\nNote: No TF-overlapping SNPs found or TF BED missing.")
    print("      Either re-run Week3 to create filtered_tf_sites.bed,")
    print("      or proceed with motif delta-PWM on all final SNPs (less specific).")