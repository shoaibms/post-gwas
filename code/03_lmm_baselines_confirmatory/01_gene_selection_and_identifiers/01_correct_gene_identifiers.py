#!/usr/bin/env python3
"""
Correct Gene Identifiers in Result Files
-----------------------------------------
Replaces placeholder 'GENE_X' identifiers in ECT result CSVs with real
Zm00001d gene IDs recovered from the transformer data bundle (.pt file).
"""
import torch
import pandas as pd
from pathlib import Path

# Path to the .pt file containing the ordered gene ID mapping
PT_FILE_PATH = Path(r"C:\Users\ms\Desktop\gwas\output\transformer_data\transformer_data_100_genes_drought.pt")

# Result files that need gene ID correction
FILES_TO_CORRECT = [
    Path(r"C:\Users\ms\Desktop\gwas\output\ect_v3m_drought_full_100\ect_oof_r2_by_gene.csv"),
    Path(r"C:\Users\ms\Desktop\gwas\output\ect_v3m_drought_full_100\ect_cis_mass_by_env.csv"),
    Path(r"C:\Users\ms\Desktop\gwas\output\ect_v3m_drought_dynamic_21\ect_oof_r2_by_gene.csv"),
]


def recover_gene_ids():
    """
    Loads gene IDs from the .pt file and replaces GENE_X placeholders
    in the specified result files.
    """
    print(f"Loading gene ID mapping from: {PT_FILE_PATH}")

    # Step 1: Load the data and extract the real gene IDs in order
    try:
        data = torch.load(PT_FILE_PATH, map_location='cpu')
        # Robustly find the gene list, checking common keys
        real_gene_ids = data.get('gene_ids') or data.get('gene_names')
        if real_gene_ids is None:
            raise KeyError("Could not find 'gene_ids' or 'gene_names' in the .pt file.")
        print(f"Successfully loaded {len(real_gene_ids)} real gene IDs.")
    except Exception as e:
        print(f"ERROR: Could not load or parse the .pt file. {e}")
        return

    # Step 2: Create the mapping dictionary (the "Rosetta Stone")
    # This will map 'GENE_0' -> 'Zm00001d...', 'GENE_1' -> 'Zm00001d...', etc.
    gene_map = {f"GENE_{i}": real_id for i, real_id in enumerate(real_gene_ids)}
    print("Created mapping for GENE_X -> Real Gene ID.")

    # Step 3: Loop through each file, apply the mapping, and save a new version
    print("\nProcessing files...")
    for file_path in FILES_TO_CORRECT:
        if not file_path.exists():
            print(f"  - WARNING: File not found, skipping: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            # Check if the 'gene_id' column exists
            if 'gene_id' in df.columns:
                # Use the map to replace the placeholders.
                # .fillna(df['gene_id']) ensures that any IDs that are not in the map (e.g., already corrected) are kept as they are.
                df['gene_id'] = df['gene_id'].map(gene_map).fillna(df['gene_id'])

                # Create a new filename for the corrected output
                output_path = file_path.with_name(f"{file_path.stem}_corrected.csv")
                
                # Save the corrected dataframe
                df.to_csv(output_path, index=False)
                print(f"  - SUCCESS: Corrected '{file_path.name}' -> saved as '{output_path.name}'")
            else:
                print(f"  - WARNING: No 'gene_id' column found in {file_path.name}, skipping.")

        except Exception as e:
            print(f"  - ERROR: Could not process {file_path.name}. {e}")

if __name__ == "__main__":
    recover_gene_ids()
    print("\nGene ID recovery process complete.")