#!/usr/bin/env python3
"""
Matched background control generation for SNP enrichment analysis.

Performs vectorized nearest-TSS computation with strict spacing enforcement,
no-replacement sampling, and adjacent bin fallback.
"""

from pathlib import Path
import argparse
import gzip
import json
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\ms\Desktop\gwas")
FOREGROUND = BASE / "output" / "week2_snp_selection" / "top_influential_snps_pragmatic.csv"
PLATINUM   = BASE / "output" / "week1_stability" / "platinum_modulator_set.csv"
PVAR_FILE  = BASE / "output" / "geno" / "cohort_pruned.pvar"
GFF3_FILE  = BASE / "data" / "maize" / "Zea_mays.B73_RefGen_v4.gff3.gz"
OUTDIR     = BASE / "output" / "week2_enrichment"

BG_PER_FG       = 10
MIN_SPACING_KB  = 50
CIS_WINDOW_KB   = 500
BINS = [(0, 20_000, "0-20kb"), (20_000, 100_000, "20-100kb"), (100_000, 500_000, "100-500kb")]

# Bin adjacency for fallback
ADJ = {
    "0-20kb":      ["20-100kb", "100-500kb"],
    "20-100kb":    ["0-20kb", "100-500kb"],
    "100-500kb":   ["20-100kb"]
}


def load_pvar(p):
    """Load PVAR file."""
    rows = []
    with open(p, "r") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            x = ln.rstrip("\n").split("\t")
            if len(x) < 5:
                continue
            rows.append({
                "CHROM": x[0],
                "POS": int(x[1]),
                "ID": x[2],
                "REF": x[3],
                "ALT": x[4]
            })
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Empty PVAR.")
    
    df["CHR_N"] = df["CHROM"].astype(str).str.replace("chr", "", case=False, regex=False)
    return df


def load_gff3_gene_positions(p):
    """Load gene TSS positions from GFF3."""
    op = gzip.open if str(p).endswith(".gz") else open
    rec = []
    
    with op(p, "rt") as fh:
        for ln in fh:
            if not ln or ln.startswith("#"):
                continue
            
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            
            seqid, _, _, start, end, _, strand, _, attrs = parts
            
            gid = None
            for kv in attrs.split(";"):
                if kv.startswith("ID="):
                    gid = kv.split("=", 1)[1].replace("gene:", "")
                    break
            
            if not (gid and gid.startswith("Zm00001d")):
                continue
            
            tss = int(end) if strand == "-" else int(start)
            chr_norm = seqid.replace("chr", "").replace("Chr", "")
            
            rec.append({"gene": gid, "chr": chr_norm, "tss": tss})
    
    df = pd.DataFrame(rec)
    if df.empty:
        raise RuntimeError("No genes parsed from GFF3.")
    
    return df


def nearest_tss_distance_per_chr(pos_arr: np.ndarray, tss_sorted: np.ndarray) -> np.ndarray:
    """Vectorized nearest-TSS distance computation using searchsorted."""
    idx = np.searchsorted(tss_sorted, pos_arr, side="left")
    
    # Left distances
    left = np.full(pos_arr.shape, np.inf)
    mask_left = idx > 0
    left[mask_left] = np.abs(pos_arr[mask_left] - tss_sorted[idx[mask_left] - 1])
    
    # Right distances
    right = np.full(pos_arr.shape, np.inf)
    mask_right = idx < tss_sorted.size
    right[mask_right] = np.abs(pos_arr[mask_right] - tss_sorted[idx[mask_right]])
    
    return np.minimum(left, right)


def build_exclusion_mask_per_chr(pos_arr: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Build exclusion mask for platinum gene windows."""
    if intervals.size == 0:
        return np.zeros(pos_arr.shape[0], dtype=bool)
    
    mask = np.zeros(pos_arr.shape[0], dtype=bool)
    
    for s, e in intervals:
        mask |= ((pos_arr >= s) & (pos_arr <= e))
    
    return mask


def bin_labels(dist_abs: np.ndarray) -> np.ndarray:
    """Assign distance bins to SNPs."""
    out = np.empty(dist_abs.shape[0], dtype=object)
    out[:] = "out_of_500kb"
    
    for lo, hi, name in BINS:
        m = (dist_abs >= lo) & (dist_abs < hi)
        out[m] = name
    
    return out


def greedy_pick_with_spacing(idx_pool: np.ndarray, pos_pool: np.ndarray,
                             k: int, min_spacing_bp: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Pick up to k unique indices with spatial separation (no replacement)."""
    if idx_pool.size == 0 or k <= 0:
        return np.array([], dtype=int)
    
    order = rng.permutation(idx_pool.size)
    chosen_idx = []
    chosen_pos = []
    
    for o in order:
        cand_idx = idx_pool[o]
        cand_pos = int(pos_pool[o])
        
        if not chosen_pos:
            chosen_idx.append(cand_idx)
            chosen_pos.append(cand_pos)
        else:
            if np.min(np.abs(np.asarray(chosen_pos, dtype=np.int64) - cand_pos)) >= min_spacing_bp:
                chosen_idx.append(cand_idx)
                chosen_pos.append(cand_pos)
        
        if len(chosen_idx) >= k:
            break
    
    return np.array(chosen_idx, dtype=int)


def main():
    """Main execution function."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--foreground", default=str(FOREGROUND))
    ap.add_argument("--platinum", default=str(PLATINUM))
    ap.add_argument("--pvar", default=str(PVAR_FILE))
    ap.add_argument("--gff3", default=str(GFF3_FILE))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--bg_per_fg", type=int, default=BG_PER_FG)
    ap.add_argument("--min_spacing_kb", type=int, default=MIN_SPACING_KB)
    ap.add_argument("--cis_kb", type=int, default=CIS_WINDOW_KB)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relax_rounds", type=int, default=2)
    ap.add_argument("--relax_factor", type=float, default=0.5)
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "background_sampling_log.txt"
    rng = np.random.default_rng(args.seed)
    
    print("="*60)
    print("WEEK 2: MATCHED BACKGROUND GENERATION (VECTORIZED)")
    print("="*60)
    
    # Load data
    print("\nStep 1: Loading data...")
    fg = pd.read_csv(args.foreground)
    assert {"gene", "chr", "pos", "dist_to_tss"}.issubset(fg.columns)
    fg["chr"] = fg["chr"].astype(str).str.replace("chr", "", regex=False)
    print(f"  Foreground SNPs: {len(fg)}")
    
    pvar = load_pvar(args.pvar)
    print(f"  PVAR SNPs: {len(pvar):,}")
    
    gpos = load_gff3_gene_positions(args.gff3)
    print(f"  Genes: {len(gpos):,}")
    
    plat = pd.read_csv(args.platinum)["gene"].astype(str).tolist()
    print(f"  Platinum genes: {len(plat)}")
    
    # Build per-chr TSS arrays
    print("\nStep 2: Building TSS arrays per chromosome...")
    tss_by_chr = {}
    for c, g in gpos.groupby("chr"):
        tss_by_chr[c] = np.sort(g["tss"].to_numpy(np.int64))
    
    # Build exclusion windows
    print("Step 3: Building platinum exclusion windows...")
    win = args.cis_kb * 1000
    excl_by_chr = {}
    
    for c in gpos["chr"].unique():
        rows = gpos.loc[gpos["gene"].isin(plat) & (gpos["chr"] == c), ["tss"]]
        
        if rows.empty:
            excl_by_chr[c] = np.empty((0, 2), dtype=np.int64)
        else:
            t = rows["tss"].to_numpy(np.int64)
            excl_by_chr[c] = np.stack([np.maximum(0, t - win), t + win], axis=1)
    
    # Precompute for ALL PVAR SNPs
    print("\nStep 4: Precomputing TSS distances for all SNPs...")
    idx_by_chr = {}
    pos_by_chr = {}
    dist_by_chr = {}  # Store actual distances per chromosome
    bin_by_chr = {}
    allowed_by_chr = {}
    
    for c, vp in pvar.groupby("CHR_N"):
        print(f"  Processing chromosome {c}...")
        
        pos = vp["POS"].to_numpy(np.int64)
        tss = tss_by_chr.get(c, np.array([], dtype=np.int64))
        
        if tss.size == 0:
            dist = np.full(pos.shape[0], np.inf)
            bins = np.array(["out_of_500kb"] * pos.shape[0], dtype=object)
            allowed = np.ones(pos.shape[0], dtype=bool)
        else:
            dist = nearest_tss_distance_per_chr(pos, tss)
            bins = bin_labels(dist)
            excl = build_exclusion_mask_per_chr(pos, excl_by_chr.get(c, np.empty((0, 2), dtype=np.int64)))
            allowed = ~excl
        
        idx_by_chr[c] = vp.index.to_numpy()
        pos_by_chr[c] = pos
        dist_by_chr[c] = dist
        bin_by_chr[c] = bins
        allowed_by_chr[c] = allowed
    
    # Build candidate pools per (chr, bin)
    print("\nStep 5: Building candidate pools...")
    pools = {}
    
    for c, idx in idx_by_chr.items():
        bins = bin_by_chr[c]
        allow = allowed_by_chr[c]
        
        for _, _, name in BINS:
            sel = idx[(bins == name) & allow]
            if sel.size:
                pools[(c, name)] = sel
    
    print(f"  Created {len(pools)} (chr, bin) pools")
    
    # Sampling
    print(f"\nStep 6: Sampling {args.bg_per_fg}:1 matched backgrounds...")
    logs = []
    min_spacing_bp = args.min_spacing_kb * 1000
    bg_rows = []
    
    for i, r in fg.iterrows():
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing foreground SNP {i+1}/{len(fg)}...")
        
        chr_fg = str(r["chr"])
        pos_fg = int(r["pos"])
        absd = int(abs(r["dist_to_tss"]))
        
        # Determine bin
        bname = "out_of_500kb"
        for lo, hi, name in BINS:
            if lo <= absd < hi:
                bname = name
                break
        
        chosen = np.array([], dtype=int)
        chosen_pos = np.array([], dtype=np.int64)  # Track positions across bins
        spacing_bp = min_spacing_bp
        
        # Try primary bin, then adjacent bins with spacing relaxation
        search_bins = [bname] + ADJ.get(bname, [])
        
        for relax_round in range(args.relax_rounds + 1):
            for bn in search_bins:
                pool = pools.get((chr_fg, bn), np.array([], dtype=int))
                
                if pool.size == 0:
                    continue
                
                pool_pos = pvar.loc[pool, "POS"].to_numpy(np.int64)
                
                # Remove already chosen
                if chosen.size > 0:
                    mask = ~np.isin(pool, chosen)
                    pool = pool[mask]
                    pool_pos = pool_pos[mask]
                
                if pool.size == 0:
                    continue
                
                picks = greedy_pick_with_spacing(
                    pool, pool_pos,
                    k=args.bg_per_fg - chosen.size,
                    min_spacing_bp=spacing_bp,
                    rng=rng
                )
                
                # Check spacing against already chosen from other bins
                if picks.size > 0:
                    picked_pos = pvar.loc[picks, "POS"].to_numpy(np.int64)
                    
                    if chosen_pos.size > 0:
                        keep = np.array([
                            np.min(np.abs(chosen_pos - p)) >= spacing_bp 
                            for p in picked_pos
                        ], dtype=bool)
                        picks = picks[keep]
                        picked_pos = picked_pos[keep]
                    
                    if picks.size > 0:
                        chosen = np.concatenate([chosen, picks])
                        chosen_pos = np.concatenate([chosen_pos, picked_pos])
                
                if chosen.size >= args.bg_per_fg:
                    break
            
            if chosen.size >= args.bg_per_fg:
                break
            
            # Relax spacing for next pass
            spacing_bp = int(max(0, spacing_bp * args.relax_factor))
        
        if chosen.size < args.bg_per_fg:
            logs.append(f"Warning: {r['gene']} {chr_fg}:{pos_fg} -> {chosen.size}/{args.bg_per_fg}")
        
        # Record backgrounds with precomputed distance lookup
        for j, cid in enumerate(chosen, 1):
            row = pvar.loc[cid]
            bg_chr_key = str(row["CHR_N"])
            
            # Look up actual distance from precomputed data
            bg_local_idx = np.where(idx_by_chr[bg_chr_key] == cid)[0][0]
            bg_actual_dist = int(dist_by_chr[bg_chr_key][bg_local_idx])
            
            bg_rows.append({
                "fg_gene": r["gene"],
                "fg_snp_id": f"{r['chr']}:{r['pos']}",
                "fg_chr": chr_fg,
                "fg_pos": pos_fg,
                "fg_abs_dist_to_tss": absd,
                "fg_bin": bname,
                "bg_snp_id": str(row["ID"]),
                "bg_chr": bg_chr_key,
                "bg_pos": int(row["POS"]),
                "bg_ref": row["REF"],
                "bg_alt": row["ALT"],
                "bg_abs_dist_to_tss": bg_actual_dist
            })
    
    bg = pd.DataFrame(bg_rows)
    outf = outdir / "background_matched_10x.csv"
    bg.to_csv(outf, index=False)
    
    print("\n" + "="*60)
    print("BACKGROUND GENERATION COMPLETE")
    print("="*60)
    
    # QC statistics
    fg_bins = pd.cut(
        np.abs(fg["dist_to_tss"]),
        bins=[0, 20_000, 100_000, 500_000, np.inf],
        labels=["0-20kb", "20-100kb", "100-500kb", "out_of_500kb"],
        right=False
    )
    
    # Bin actual background distances
    bg_bins = pd.cut(
        bg["bg_abs_dist_to_tss"],
        bins=[0, 20_000, 100_000, 500_000, np.inf],
        labels=["0-20kb", "20-100kb", "100-500kb", "out_of_500kb"],
        right=False
    )
    
    qc = {
        "foreground_count": len(fg),
        "background_count": len(bg),
        "actual_ratio": len(bg) / len(fg),
        "foreground_bin_counts": fg_bins.value_counts(dropna=False).to_dict(),
        "background_bin_counts": bg_bins.value_counts(dropna=False).to_dict(),
        "unique_bg_snps": int(bg[["bg_chr", "bg_pos"]].drop_duplicates().shape[0]),
        "min_spacing_kb": args.min_spacing_kb,
        "settings": {
            "bg_per_fg": args.bg_per_fg,
            "min_spacing_kb": args.min_spacing_kb,
            "cis_exclusion_kb": args.cis_kb,
            "relax_rounds": args.relax_rounds,
            "relax_factor": args.relax_factor
        }
    }
    
    with open(outdir / "background_qc.json", "w") as fh:
        json.dump(qc, fh, indent=2)
    
    if logs:
        Path(log_path).write_text("\n".join(logs))
    
    print(f"\nForeground SNPs: {len(fg)}")
    print(f"Background SNPs: {len(bg)}")
    print(f"Actual ratio: {len(bg)/len(fg):.1f}:1")
    print(f"Unique backgrounds: {qc['unique_bg_snps']}")
    
    print(f"\nForeground bin distribution:")
    for bin_name, count in sorted(qc["foreground_bin_counts"].items()):
        print(f"  {bin_name}: {count}")
    
    print(f"\nBackground bin distribution:")
    for bin_name, count in sorted(qc["background_bin_counts"].items()):
        print(f"  {bin_name}: {count}")
    
    print(f"\nSaved:")
    print(f"  {outf}")
    print(f"  {outdir / 'background_qc.json'}")
    
    if logs:
        print(f"  {log_path} ({len(logs)} warnings)")
    
    print("\nREADY FOR: Consequence enrichment analysis")


if __name__ == "__main__":
    main()