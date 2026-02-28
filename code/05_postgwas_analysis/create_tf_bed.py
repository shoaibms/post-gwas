"""
Create TF BED from media-10.gz with correct parsing
Format: Column 1=TF name, Column 2=chr:pos
"""

import pandas as pd
import gzip
from pathlib import Path

BASE = Path(r"C:\Users\ms\Desktop\gwas")
OUT = BASE / "output" / "week3_tf_binding"
OUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("CREATING TF BED FILE")
print("="*60)

tf_file = BASE / "data" / "maize" / "media-10.gz"

print("\n1. Parsing media-10.gz...")
print("   Format: TF name | chr:pos | closest_gene | ...")

data = []
with gzip.open(tf_file, 'rt') as f:
    for line_num, line in enumerate(f, 1):
        # Skip comments
        if line.startswith('#'):
            continue
        
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        tf_name = parts[0]
        chr_pos = parts[1]
        
        # Parse chr:pos
        if ':' not in chr_pos:
            continue
        
        try:
            chr_part, pos_part = chr_pos.split(':', 1)
            chr_clean = chr_part.replace('chr', '')
            pos = int(pos_part)
            data.append({'tf': tf_name, 'chr': chr_clean, 'pos': pos})
        except:
            continue

tf = pd.DataFrame(data)
print(f"   Loaded: {len(tf):,} sites")
print(f"   Chromosomes: {sorted(tf['chr'].unique())}")

# Deduplicate by chr:pos
print("\n2. Deduplicating by position...")
tf = tf.drop_duplicates(subset=['chr', 'pos']).sort_values(['chr', 'pos'])
print(f"   After dedup: {len(tf):,} sites")

# Create BED
print("\n3. Creating BED format...")
bed = pd.DataFrame({
    'chr': tf['chr'],
    'start': (tf['pos'] - 50).clip(lower=1),
    'end': tf['pos'] + 50,
    'tf': tf['tf']
})

# Save
out_file = OUT / "filtered_tf_sites.bed"
bed.to_csv(out_file, sep='\t', header=False, index=False)

print(f"\nSaved: {out_file}")
print(f"Final sites: {len(bed):,}")
print(f"Chromosomes: {sorted(bed['chr'].unique())}")

# Sample output
print(f"\nFirst 3 BED entries:")
print(bed.head(3).to_string(index=False, header=False))

print("\n" + "="*60)
print("TF BED FILE CREATED")
print("="*60)
print("\nNext: Run ieqtl_finalize.py to check overlaps")