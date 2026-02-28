"""
Motif delta-PWM disruption analysis.

Scores all JASPAR plant motifs against ieQTL SNPs overlapping TF peaks,
computing disruption scores and empirical p-values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import random
import re

try:
    from statsmodels.stats.multitest import fdrcorrection
except ImportError:
    pass

# --- CONSTANTS AND HELPER FUNCTIONS ---

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUT = BASE / "output" / "week6_ieqtl"
WINDOW = 15
BASES = ['A','C','G','T']
BASE2IDX = {'A':0, 'C':1, 'G':2, 'T':3}

def sanitize(seq: str) -> str:
    s = seq.upper()
    return ''.join(ch if ch in 'ACGT' else 'N' for ch in s)

def base_llr(row_vector, base_char):
    b = BASE2IDX.get(str(base_char).upper())
    if b is None: return 0.0
    return float(row_vector[b])

def probabilities_to_log_odds_with_pseudocount(prob_array, bg={'A':0.25,'C':0.25,'G':0.25,'T':0.25}):
    order = ['A', 'C', 'G', 'T']
    pseudocount = 0.001
    probs_with_pc = prob_array.astype(float) + pseudocount
    probs_normalized = probs_with_pc / np.sum(probs_with_pc, axis=1, keepdims=True)
    bg_vec = np.array([bg[b] for b in order], dtype=float)
    lo = np.log2(probs_normalized / bg_vec)
    return lo

def calculate_disruption_at_best_match(ref_seq31, alt_seq31, lo_pwm):
    m = len(lo_pwm)
    best_ref_score = -1e18
    best_s = -1
    best_strand = '+'
    start_min = max(0, WINDOW - (m - 1))
    start_max = min(len(ref_seq31) - m, WINDOW)

    for s in range(start_min, start_max + 1):
        ref_segment = ref_seq31[s : s + m]
        score = sum(base_llr(lo_pwm[j], base) for j, base in enumerate(ref_segment))
        if score > best_ref_score:
            best_ref_score = score
            best_s = s
            best_strand = '+'
            
    rc_ref = str(Seq(ref_seq31).reverse_complement())
    for s in range(start_min, start_max + 1):
        ref_segment = rc_ref[s : s + m]
        score = sum(base_llr(lo_pwm[j], base) for j, base in enumerate(ref_segment))
        if score > best_ref_score:
            best_ref_score = score
            best_s = s
            best_strand = '-'

    if best_strand == '+':
        alt_segment = alt_seq31[best_s : best_s + m]
        best_alt_score = sum(base_llr(lo_pwm[j], base) for j, base in enumerate(alt_segment))
    else:
        rc_alt = str(Seq(alt_seq31).reverse_complement())
        alt_segment = rc_alt[best_s : best_s + m]
        best_alt_score = sum(base_llr(lo_pwm[j], base) for j, base in enumerate(alt_segment))

    return best_alt_score - best_ref_score, best_ref_score

def main():
    print("="*60)
    print("MOTIF DELTA-PWM DISRUPTION ANALYSIS")
    print("="*60)

    # --- Load Data ---
    print("\n1. Loading SNP and Genome data...")
    snps = pd.read_csv(OUT / "ieqtl_snps_overlapping_tf_peaks.csv")
    ieqtl = pd.read_csv(OUT / "ieqtl_final_results.csv")
    snps = snps.merge(ieqtl[['snp', 'ref', 'alt']], on='snp', how='left')
    fasta_file = BASE / "data" / "maize" / "ref" / "agpv4" / "Zm-B73-REFERENCE-GRAMENE-4.0.fa"
    
    # Normalize chromosome names by stripping 'Chr'/'chr' prefixes
    genome = {rec.id.split()[0].replace('Chr', '').replace('chr', ''): rec.seq for rec in SeqIO.parse(str(fasta_file), "fasta")}
    
    print(f"   Loaded {len(snps)} SNPs and {len(genome)} chromosomes.")

    # --- Load Motifs ---
    print("\n2. Loading and processing JASPAR motifs...")
    jaspar_file = BASE / "data" / "motifs" / "jaspar_plants" / "JASPAR2024_CORE_plants_non-redundant.meme"
    pwm_list = []
    current_motif = None
    with open(jaspar_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('MOTIF'):
                if current_motif: pwm_list.append(current_motif)
                parts = line.split(); current_motif = {'id': parts[1], 'name': parts[2], 'probs': []}
            elif current_motif and (line[0].isdigit() or (line.startswith(" ") and line.strip())):
                try:
                    values = [float(x) for x in line.split()]; 
                    if len(values) == 4: current_motif['probs'].append(values)
                except ValueError: continue
    if current_motif: pwm_list.append(current_motif)
    for motif in pwm_list: motif['probs'] = np.array(motif['probs'])
    lo_pwms = {m['id']: (m['name'], probabilities_to_log_odds_with_pseudocount(m['probs'])) for m in pwm_list}
    print(f"   Loaded {len(lo_pwms)} motifs.")
    if not lo_pwms: raise RuntimeError("FATAL: Motif loading failed.")

    # --- Scoring ---
    print("\n3. Scoring all motifs for all SNPs (this may take 5-10 minutes)...")
    results = []
    for idx, row in snps.iterrows():
        snp, chr_id, pos, ref, alt = row['snp'], str(row['chr']), int(row['pos']), row.get('ref', 'N'), row.get('alt', 'N')
        print(f"   Scoring SNP {idx+1}/{len(snps)}: {snp}")
        if chr_id not in genome: continue
        chr_seq = genome[chr_id]
        start, end = pos - 1 - WINDOW, pos - 1 + WINDOW + 1
        if start < 0 or end > len(chr_seq): continue
        ref_seq = sanitize(str(chr_seq[start:end]).upper())
        if not alt or pd.isna(alt) or len(str(alt)) != 1: continue
        alt_seq = ref_seq[:WINDOW] + str(alt).upper() + ref_seq[WINDOW+1:]
        for matrix_id, (motif_name, lo_pwm) in lo_pwms.items():
            delta, _ = calculate_disruption_at_best_match(ref_seq, alt_seq, lo_pwm)
            results.append({'snp': snp, 'abs_delta_llr': abs(delta)})
    
    if not results:
        print("\nCRITICAL ERROR: No SNPs were scored. Check input files for consistency.")
        return

    res = pd.DataFrame(results)
    
    # --- Summarizing ---
    print("\n4. Summarizing disruption scores...")
    snp_summary = res.groupby('snp')['abs_delta_llr'].sum().reset_index()
    snp_summary.rename(columns={'abs_delta_llr': 'disruption_score_sum'}, inplace=True)
    snp_summary = snp_summary.merge(snps[['snp', 'gene', 'chr', 'pos', 'ref', 'alt']].drop_duplicates(), on='snp')
    print("   Summary complete.")

    # --- P-value Calculation ---
    print("\n5. Calculating p-values...")
    p_values = []
    n = 200
    for idx, row in snp_summary.iterrows():
        snp_id, obs_score = row['snp'], row['disruption_score_sum']
        print(f"   Calculating p-value for SNP {idx+1}/{len(snp_summary)}: {snp_id}")
        chr_id, pos = str(row['chr']), int(row['pos'])
        chr_seq = genome[chr_id]
        start, end = pos - 1 - WINDOW, pos - 1 + WINDOW + 1
        ref_seq = sanitize(str(chr_seq[start:end]).upper())
        ref_base = ref_seq[WINDOW]
        alt_bases = [b for b in BASES if b != ref_base]
        
        precalculated_null_scores = {}
        for alt_base in alt_bases:
            alt_seq = ref_seq[:WINDOW] + alt_base + ref_seq[WINDOW+1:]
            total_disruption = sum(abs(calculate_disruption_at_best_match(ref_seq, alt_seq, lo_pwm)[0]) for _, lo_pwm in lo_pwms.values())
            precalculated_null_scores[alt_base] = total_disruption
            
        null_scores = [random.choice(list(precalculated_null_scores.values())) for _ in range(n)]
        ge = sum(1 for s in null_scores if s >= obs_score)
        p_val = (ge + 1) / (n + 1)
        p_values.append(p_val)

    snp_summary['p_value'] = p_values
    
    # --- Final Processing ---
    reject, q_values = fdrcorrection(snp_summary['p_value'], alpha=0.05)
    snp_summary['q_value'] = q_values
    snp_summary['significant'] = reject
    snp_summary.sort_values('p_value', ascending=True, inplace=True)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for _, row in snp_summary.iterrows():
        sig_marker = "*" if row['significant'] else ""
        print(f"  {row['snp']:<20} Score={row['disruption_score_sum']:.1f} (p={row['p_value']:.4f}, q={row['q_value']:.4f}{sig_marker})")

    top_file = OUT / "motif_disruption_summary_per_snp.csv"
    snp_summary.to_csv(top_file, index=False)
    print(f"\nSaved final disruption summary: {top_file}")
    print("\nMOTIF ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()