#!/usr/bin/env python3
"""
Download AGPv4 GO annotations directly from maize-GAMER
No ID conversion needed - native AGPv4 format
"""

import gzip
import pandas as pd
from pathlib import Path
from urllib.request import Request, urlopen

BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
GO_DIR = BASE_DIR / "data" / "maize" / "process" / "go"
GO_DIR.mkdir(parents=True, exist_ok=True)

GAF_URL = "https://download.maizegdb.org/GeneFunction_and_Expression/Maize-GAMER/B73_RefGen_v4/e.agg_data/maize.B73.AGPv4.aggregate.gaf.gz"
# Fallback mirror: "https://raw.githubusercontent.com/timedreamer/public_dataset/master/maize.B73.AGPv4.aggregate.gaf.gz"
GAF_FILE = GO_DIR / "maize.AGPv4.gaf.gz"
OUTPUT_CSV = GO_DIR / "maize_gene_to_go_v4.csv"

print("="*60)
print("DOWNLOADING AGPv4 GO ANNOTATIONS FROM MAIZE-GAMER")
print("="*60)

print(f"\nDownload URL: {GAF_URL}")
print(f"Saving to: {GAF_FILE}")

req = Request(GAF_URL, headers={"User-Agent":"Mozilla/5.0"})
with urlopen(req, timeout=300) as r, open(GAF_FILE, "wb") as out:
    out.write(r.read())

# Verify downloaded file is valid gzip
with open(GAF_FILE, "rb") as f:
    head = f.read(2)
assert head == b"\x1f\x8b", f"Not gzip (starts with {head!r}). Server likely returned HTML."

print("\n" + "="*60)
print("PARSING GAF FILE")
print("="*60)

rows = []
line_count = 0
skipped = 0

with gzip.open(GAF_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
    for ln in fh:
        line_count += 1
        
        if ln.startswith("!"):
            continue
        
        fields = ln.rstrip("\n").split("\t")
        
        if len(fields) < 15:
            skipped += 1
            continue
        
        db = fields[0]
        gene = fields[1]
        qualifier = fields[3]
        go_id = fields[4]
        evidence = fields[6]
        
        if not gene or not go_id.startswith("GO:"):
            skipped += 1
            continue
        
        rows.append({
            'gene': gene,
            'go_id': go_id,
            'evidence': evidence
        })

print(f"\nProcessed {line_count} lines")
print(f"Skipped {skipped} lines")
print(f"Extracted {len(rows)} GO annotations")

df = pd.DataFrame(rows)

print(f"\nUnique genes: {df['gene'].nunique()}")
print(f"Unique GO terms: {df['go_id'].nunique()}")

print("\nSample annotations:")
print(df.head(10))

print("\n" + "="*60)
print("CREATING GENE-TO-GO MAPPING")
print("="*60)

gene_to_go = df.groupby('gene')['go_id'].apply(
    lambda x: ';'.join(sorted(set(x)))
).reset_index()
gene_to_go.columns = ['gene', 'go_terms']

gene_to_go.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Mapped genes: {len(gene_to_go)}")

print("\nSample gene-to-GO mappings:")
print(gene_to_go.head(10))

print("\n" + "="*60)
print("CHECKING COVERAGE ON YOUR GENE SETS")
print("="*60)

DECOUPLE_500KB = BASE_DIR / "output" / "ect_alt" / "integrated" / "decouple_labels_500kb.csv"
DECOUPLE_2MB = BASE_DIR / "output" / "ect_alt" / "integrated" / "decouple_labels_2Mb.csv"

lab_500kb = pd.read_csv(DECOUPLE_500KB)
lab_2mb = pd.read_csv(DECOUPLE_2MB)

go_genes = set(gene_to_go['gene'].unique())

coverage_500kb = lab_500kb['gene'].isin(go_genes).mean()
coverage_2mb = lab_2mb['gene'].isin(go_genes).mean()

print(f"\nGO coverage on 500kb genes: {coverage_500kb:.1%}")
print(f"GO coverage on 2Mb genes: {coverage_2mb:.1%}")

if coverage_500kb >= 0.90:
    print("\nEXCELLENT: Coverage >90% - Ready for enrichment analysis")
else:
    print(f"\nWARNING: Coverage {coverage_500kb:.1%} < 90%")
    print("Some genes may not have GO annotations")

print("\n" + "="*60)
print("AGPv4 GO ANNOTATIONS READY")
print("="*60)
print(f"\nFinal file: {OUTPUT_CSV}")
print("Ready to proceed with GO enrichment analysis")