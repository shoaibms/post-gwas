#!/usr/bin/env python3
"""
Corrected Gene ID Version Check for Filtered Files
==================================================
Validates that the gene IDs in the filtered expression data match the
GFF3 annotation. This script is designed to check the cleaned files
from the 'output/data_filtered/' directory.

Usage:
    python corrected_gene_id_check.py
"""

import pandas as pd
import gzip
import re
from pathlib import Path
import sys

def print_header(title: str):
    """Prints a formatted header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)

def check_expression_gene_ids(fpkm_path: Path):
    """Check gene ID format in FILTERED FPKM files."""
    print_header("GENE ID VERSION VALIDATION (FILTERED FILES)")

    print(f"Checking filtered gene IDs from: {fpkm_path}")

    if not fpkm_path.exists():
        print(f"ERROR: Filtered FPKM file not found: {fpkm_path}")
        return "FILE_NOT_FOUND"

    # Read gene IDs from FPKM header
    opener = gzip.open if str(fpkm_path).endswith(".gz") else open
    try:
        with opener(fpkm_path, "rt", encoding="utf-8") as fh:
            header = fh.readline().strip().split("\t")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return "READ_ERROR"

    genes = header[1:]  # Skip first column (sample names)
    sample = genes[:20]  # First 20 genes for inspection

    print(f"\nTotal genes in FILTERED expression data: {len(genes):,}")
    print(f"Example gene IDs from filtered FPKM file:")
    for i, gene in enumerate(sample):
        print(f"  {i+1:2d}. {gene}")

    # Pattern matching
    agpv4_pattern = r"Zm0+1d\d+"  # AGPv4: Zm00001d######
    namv5_pattern = r"Zm0+1eb\d+" # NAM v5: Zm00001eb######
    ensrna_pattern = r"ENSRNA\d+"  # Ensembl RNA IDs

    agpv4_matches = [g for g in sample if re.match(agpv4_pattern, g)]
    namv5_matches = [g for g in sample if re.match(namv5_pattern, g)]
    ensrna_matches = [g for g in sample if re.match(ensrna_pattern, g)]

    print_header("VERSION DETECTION RESULTS")
    print(f"AGPv4-like genes (Zm00001d######): {len(agpv4_matches)}/20")
    if agpv4_matches:
        print(f"  Examples: {agpv4_matches[:5]}")

    print(f"NAM v5-like genes (Zm00001eb#####): {len(namv5_matches)}/20")
    if namv5_matches:
        print(f"  Examples: {namv5_matches[:5]}")

    print(f"ENSRNA IDs (should be 0 in filtered): {len(ensrna_matches)}/20")
    if ensrna_matches:
        print(f"  Examples: {ensrna_matches[:5]}")

    # Decision logic
    print_header("VALIDATION OUTCOME")

    if len(agpv4_matches) >= 15:  # Most genes are AGPv4
        print("SUCCESS: AGPv4 Gene IDs Detected.")
        print("   - Filtered expression data uses: Zm00001d###### format")
        print("   - Compatible with: Zea_mays.B73_RefGen_v4.gff3.gz")
        print("   - ENSRNA IDs successfully filtered out")
        return "AGPv4"

    elif len(namv5_matches) >= 15:  # Most genes are NAM v5
        print("SUCCESS: NAM v5 Gene IDs Detected.")
        print("   - Expression data uses: Zm00001eb##### format")
        print("   - Compatible with: NAM v5 annotations")
        return "NAMv5"

    elif len(ensrna_matches) > 0:
        print("ERROR: ENSRNA IDs Still Present.")
        print("   - The filtering process may not have worked correctly.")
        print("   - Expected: All ENSRNA IDs to be removed.")
        return "ENSRNA_PRESENT"

    else:
        print("UNKNOWN GENE ID FORMAT")
        print(f"   - AGPv4: {len(agpv4_matches)}, NAM v5: {len(namv5_matches)}, ENSRNA: {len(ensrna_matches)}")
        return "UNKNOWN"

def check_gff3_gene_ids(gff3_path: Path):
    """Check gene IDs in the GFF3 file with improved parsing"""
    print_header("GFF3 GENE ID SAMPLE (IMPROVED PARSING)")

    print(f"Checking GFF3 file: {gff3_path}")

    if not gff3_path.exists():
        print(f"ERROR: GFF3 file not found: {gff3_path}")
        return []

    gene_ids = []
    try:
        with gzip.open(gff3_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 9 and parts[2].lower() == "gene":
                    attributes = parts[8]

                    # Parse ID attribute more carefully
                    for attr in attributes.split(";"):
                        if attr.startswith("ID="):
                            raw_id = attr.split("=", 1)[1]
                            # Remove prefixes like "gene:" or "Gene:"
                            gene_id = raw_id.replace("gene:", "").replace("Gene:", "")
                            gene_ids.append(gene_id)
                            if len(gene_ids) >= 20:
                                break
                if len(gene_ids) >= 20:
                    break
    except Exception as e:
        print(f"Error reading GFF3: {e}")
        return []

    print(f"Example gene IDs from GFF3 (first 20):")
    for i, gene in enumerate(gene_ids):
        print(f"  {i+1:2d}. {gene}")

    return gene_ids

def check_data_prep_outputs(base_dir: Path):
    """Check if data prep outputs exist and their status"""
    print_header("DATA PREPARATION OUTPUTS CHECK")

    expected_outputs = {
        "T_long.parquet": "output/data/T_long.parquet",
        "Gene map": "output/data/gene_map.csv",
        "Phenotypes": "output/data/P.csv",
        "Genotype (PGEN)": "output/geno/cohort_pruned.pgen",
        "PCs": "output/geno/pcs.eigenvec",
        "TRAW": "output/geno/G_traw.traw"
    }

    print("Checking data preparation outputs:")

    existing_count = 0
    for name, path in expected_outputs.items():
        full_path = base_dir / path
        if full_path.exists():
            try:
                size = full_path.stat().st_size
                size_mb = size / (1024*1024)
                print(f"   - Found: {name}: {path} ({size_mb:.1f} MB)")
                existing_count += 1
            except:
                print(f"   - Found: {name}: {path} (exists)")
                existing_count += 1
        else:
            print(f"   - Missing: {name}: {path}")

    print(f"\nData prep completeness: {existing_count}/{len(expected_outputs)} files present")

    return existing_count == len(expected_outputs)

def main():
    base_dir = Path.cwd()

    # Check if we're in the right directory
    required_dirs = ["output/data_filtered", "data/maize"]
    missing_dirs = [d for d in required_dirs if not (base_dir / d).exists()]

    if missing_dirs:
        print("ERROR: Not in correct directory or missing directories.")
        print(f"Current directory: {base_dir}")
        print("Missing directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\nPlease run this script from: C:\\Users\\ms\\Desktop\\gwas\\")
        sys.exit(1)

    # File paths
    fpkm_path = base_dir / "output/data_filtered/WW_209-Uniq_FPKM.agpv4.txt.gz"
    gff3_path = base_dir / "data/maize/Zea_mays.B73_RefGen_v4.gff3.gz"

    print(f"Working directory: {base_dir}")
    print(f"FILTERED FPKM file: {fpkm_path}")
    print(f"GFF3 file: {gff3_path}")

    # Check if filtered files exist
    if not fpkm_path.exists():
        print(f"\nERROR: Filtered FPKM file not found.")
        print(f"Expected: {fpkm_path}")
        print(f"\nThis suggests the filtering process has not been completed.")
        print(f"Please run the filtering script first, or check if files are in the right location.")
        return

    # Run checks
    expr_result = check_expression_gene_ids(fpkm_path)
    gff3_genes = check_gff3_gene_ids(gff3_path)

    # Check data prep outputs
    data_prep_complete = check_data_prep_outputs(base_dir)

    # Final assessment
    print_header("FINAL ASSESSMENT")

    if expr_result == "AGPv4" and gff3_path.name.startswith("Zea_mays.B73_RefGen_v4"):
        print("Assessment: Alignment between expression data and GFF3 annotation successful.")
        print("   - Filtered expression data: AGPv4 gene IDs")
        print("   - GFF3 annotation: AGPv4 (B73_RefGen_v4)")
        print("   - Gene ID filtering: SUCCESSFUL")
        print("   - STATUS: Ready for analysis.")

        if data_prep_complete:
            print("   - Data preparation: COMPLETE")
        else:
            print("   - Data preparation: INCOMPLETE - re-run with filtered files.")

    elif expr_result == "ENSRNA_PRESENT":
        print("Assessment: Filtering incomplete.")
        print("   - ENSRNA IDs are still present in the filtered files.")
        print("   - The filtering process needs to be re-run.")

    elif expr_result == "FILE_NOT_FOUND":
        print("Assessment: Filtered files missing.")
        print("   - Run the filtering process first.")

    else:
        print("Assessment: Unclear result, manual review needed.")
        print(f"   - Expression result: {expr_result}")
        print("   - Please verify gene ID formats manually.")

    print_header("NEXT STEPS")

    if expr_result == "AGPv4":
        print("Ready to proceed with your plan:")
        print("   1. Cohort freeze completed.")
        if data_prep_complete:
            print("   2. Data preparation completed.")
            print("   3. NEXT: Run Matrix eQTL baselines (Step 2 of note.txt)")
            print("   4. THEN: Create starter gene list (Step 3 of note.txt)")
        else:
            print("   2. Re-run data prep with filtered files:")
            print("      python code\\data_reading\\clean_prep_data.py --cohort-csv output/cohort/core_all3_env.csv --ww output/data_filtered/WW_209-Uniq_FPKM.agpv4.txt.gz --ws1 output/data_filtered/WS1_208-uniq_FPKM.agpv4.txt.gz --ws2 output/data_filtered/WS2_210-uniq_FPKM.agpv4.txt.gz --gff3 data/maize/Zea_mays.B73_RefGen_v4.gff3.gz")

    else:
        print("Resolve gene ID issues before proceeding:")
        print("   - Check if filtering completed successfully.")
        print("   - Verify filtered files contain clean Zm00001d IDs.")

if __name__ == "__main__":
    main()