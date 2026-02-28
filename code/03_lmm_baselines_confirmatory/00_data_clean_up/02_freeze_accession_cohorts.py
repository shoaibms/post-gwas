#!/usr/bin/env python3
"""
Cohort Freeze Script for Environment-Conditional eQTL Transformers
Creates frozen accession lists from coverage_matrix.csv for reproducible downstream analyses.

Usage: python freeze_cohorts_from_coverage.py
"""

import pandas as pd
import pathlib
from datetime import datetime

def main():
    # File paths
    coverage_matrix_path = pathlib.Path(r"C:\Users\ms\Desktop\gwas\output\inspect\coverage_matrix.csv")
    output_dir = pathlib.Path(r"C:\Users\ms\Desktop\gwas\output\cohort")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read coverage matrix
    print(f"Reading coverage matrix from: {coverage_matrix_path}")
    df = pd.read_csv(coverage_matrix_path)
    
    print(f"Coverage matrix shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Helper function for Y/N columns
    Y = lambda c: df[c].eq("Y")
    
    # Define cohort filters
    anyT = Y("has_T_WW") | Y("has_T_WS1") | Y("has_T_WS2")
    all3T = Y("has_T_WW") & Y("has_T_WS1") & Y("has_T_WS2")
    
    # Create cohorts
    cohorts = {
        "core_all3_env.csv": {
            "filter": all3T,
            "description": "Accessions with transcriptome data in all 3 environments (WW, WS1, WS2)"
        },
        "G_and_anyT.csv": {
            "filter": Y("has_G") & anyT,
            "description": "Accessions with genotype data and transcriptome data in any environment"
        },
        "labeled_P.csv": {
            "filter": Y("has_G") & anyT & Y("has_P"),
            "description": "Accessions with genotype, transcriptome (any env), and phenotype data"
        }
    }
    
    # Generate cohort files
    results = {}
    for filename, cohort_info in cohorts.items():
        cohort_df = df.loc[cohort_info["filter"], ["accession"]]
        output_path = output_dir / filename
        cohort_df.to_csv(output_path, index=False)
        
        n_accessions = len(cohort_df)
        results[filename] = n_accessions
        
        print(f"\n{filename}:")
        print(f"  Description: {cohort_info['description']}")
        print(f"  Number of accessions: {n_accessions}")
        print(f"  Saved to: {output_path}")
        
        # Show first few accessions as preview
        if n_accessions > 0:
            preview = cohort_df['accession'].head(10).tolist()
            print(f"  Preview: {', '.join(preview)}")
            if n_accessions > 10:
                print(f"    ... and {n_accessions - 10} more")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("COHORT FREEZE SUMMARY")
    print(f"{'='*60}")
    print(f"Generated {len(cohorts)} cohort files:")
    for filename, count in results.items():
        print(f"  - {filename}: {count} accessions")
    
    # Validate expected numbers from plan
    expected = {
        "core_all3_env.csv": 185,  # ≈185 from plan
        "G_and_anyT.csv": 209,     # n ≥ 209 from plan  
        "labeled_P.csv": 143       # 143 from plan
    }
    
    print(f"\nValidation against plan expectations:")
    all_good = True
    for filename, expected_count in expected.items():
        actual_count = results[filename]
        status = "OK" if actual_count >= expected_count * 0.9 else "WARNING"
        print(f"  {status} {filename}: {actual_count} (expected ~{expected_count})")
        if actual_count < expected_count * 0.9:
            all_good = False
    
    if all_good:
        print(f"\n[OK] All cohorts generated successfully and meet expectations.")
    else:
        print(f"\n[WARNING] Some cohorts may need review - check the coverage matrix data.")
    
    # Create a metadata file
    metadata = {
        "created_at": datetime.now().isoformat(),
        "coverage_matrix_path": str(coverage_matrix_path),
        "total_accessions_in_matrix": len(df),
        "cohorts": results
    }
    
    metadata_path = output_dir / "cohort_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"\nReady for next step: Data preparation (Section 2 of plan)")

if __name__ == "__main__":
    main()