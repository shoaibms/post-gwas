#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filters FPKM matrices to include only gene columns present in a maize AGPv4 GFF3 file.

This script processes one or more FPKM matrix files, retaining only the columns
that correspond to gene IDs found in the provided AGPv4 GFF3 reference.
It generates filtered FPKM files and a JSON report detailing the filtering process.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Set, TextIO

LOG = logging.getLogger(__name__)


def get_opener(path: Path) -> Callable[..., TextIO]:
    """Return the appropriate file opener based on file extension."""
    return gzip.open if path.suffix == ".gz" else open


def read_gff3_gene_ids(gff3_path: Path) -> Set[str]:
    """Extracts Zm00001d gene IDs from a GFF3 file."""
    ids = set()
    opener = get_opener(gff3_path)
    with opener(gff3_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            ftype, attrs = parts[2].lower(), parts[8]
            if ftype != "gene":
                continue

            gid = None
            # Prioritize ID=gene: format
            match = re.search(r"ID=gene:([^;]+)", attrs)
            if match:
                gid = match.group(1)
            else:
                # Fallback for ID=Zm... format
                match = re.search(r"ID=([^;]+)", attrs)
                if match and match.group(1).startswith("Zm00001d"):
                    gid = match.group(1)

            if gid:
                # Keep only the root ID, e.g., Zm00001d027232 from Zm00001d027232.1
                ids.add(gid.split(".")[0])
    return ids


def detect_delim(header: str) -> str:
    """Detects the delimiter in a header line."""
    if "\t" in header:
        return "\t"
    if "," in header:
        return ","
    raise ValueError("Could not determine delimiter (tab or comma).")


def normalize_to_zm_root(x: str) -> str | None:
    """
    Extracts the core Zm00001d ID from a string.

    Accepts variants like Zm00001d123456, Zm00001d123456.1,
    Zm00001d123456_T001, or Zm00001d123456-P001.
    """
    match = re.search(r"(Zm00001d\d+)", x)
    return match.group(1) if match else None


def filter_one(input_path: Path, output_path: Path, keep_genes: Set[str]) -> Dict:
    """Filters one FPKM matrix file against a set of gene IDs."""
    opener_in = get_opener(input_path)
    opener_out = get_opener(output_path)

    with opener_in(input_path, "rt", encoding="utf-8", errors="replace") as fi:
        header = fi.readline().rstrip("\n")
        try:
            sep = detect_delim(header)
        except ValueError as e:
            raise RuntimeError(f"Unknown delimiter in {input_path.name}: {e}")

        cols = header.split(sep)

        # First column is assumed to be accession/sample ID
        keep_idx = [0]
        kept_labels, dropped_labels = [], []
        for j, label in enumerate(cols[1:], start=1):
            root = normalize_to_zm_root(label)
            if root and root in keep_genes:
                keep_idx.append(j)
                kept_labels.append(label)
            else:
                dropped_labels.append(label)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with opener_out(output_path, "wt", encoding="utf-8") as fo:
            fo.write(sep.join(cols[i] for i in keep_idx) + "\n")
            for line in fi:
                parts = line.rstrip("\n").split(sep)
                if len(parts) < len(cols):
                    # Handle ragged lines by padding with empty strings
                    parts.extend([""] * (len(cols) - len(parts)))
                fo.write(sep.join(parts[i] for i in keep_idx) + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "n_cols_input": len(cols) - 1,
        "n_cols_kept": len(kept_labels),
        "n_cols_dropped": len(dropped_labels),
        "example_dropped": dropped_labels[:10],
        "example_kept": kept_labels[:10],
    }


def main():
    """Main script execution."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Filter FPKM matrices to AGPv4 gene models.")
    ap.add_argument("--gff3", default=r"C:\Users\ms\Desktop\gwas\data\maize\Zea_mays.B73_RefGen_v4.gff3.gz", type=Path, help="Path to AGPv4 GFF3 file.")
    ap.add_argument("--ww", default=r"C:\Users\ms\Desktop\gwas\data\maize\WW_209-Uniq_FPKM.txt.gz", type=Path, help="Path to WW FPKM matrix.")
    ap.add_argument("--ws1", default=r"C:\Users\ms\Desktop\gwas\data\maize\WS1_208-uniq_FPKM.txt.gz", type=Path, help="Path to WS1 FPKM matrix.")
    ap.add_argument("--ws2", default=r"C:\Users\ms\Desktop\gwas\data\maize\WS2_210-uniq_FPKM.txt.gz", type=Path, help="Path to WS2 FPKM matrix.")
    ap.add_argument("--outdir", default=r"C:\Users\ms\Desktop\gwas\output\data_filtered", type=Path, help="Output directory for filtered files and report.")
    args = ap.parse_args()

    LOG.info("Parsing AGPv4 GFF3 for Zm00001d gene IDs…")
    keep_genes = read_gff3_gene_ids(args.gff3)
    LOG.info(f"Found {len(keep_genes):,} AGPv4 genes in GFF3.")

    report = {"gff3": str(args.gff3), "n_gff_genes": len(keep_genes), "files": {}}

    # Define input-output file mappings
    mapping = [
        (args.ww, args.outdir / "WW_209-Uniq_FPKM.agpv4.txt.gz"),
        (args.ws1, args.outdir / "WS1_208-uniq_FPKM.agpv4.txt.gz"),
        (args.ws2, args.outdir / "WS2_210-uniq_FPKM.agpv4.txt.gz"),
    ]

    for inp, outp in mapping:
        LOG.info(f"Filtering {inp.name} → {outp.name}")
        try:
            rep = filter_one(inp, outp, keep_genes)
            report["files"][inp.name] = rep
            LOG.info(f"  Kept {rep['n_cols_kept']:,} / {rep['n_cols_input']:,} gene columns.")
        except Exception as e:
            LOG.error(f"Failed to process {inp.name}: {e}")

    # Write summary report
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_json = args.outdir / "filter_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOG.info(f"Wrote report to {out_json}")


if __name__ == "__main__":
    main()
