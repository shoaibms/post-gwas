#!/usr/bin/env python3
"""
Preflight Data Interoperability Report
---------------------------------------
Validates gene-level cis-SNP coverage by cross-referencing a gene list against
a GFF3 annotation and a PLINK .pvar genotype file. Produces a Markdown report
and per-gene coverage CSVs.
"""
import argparse, gzip, hashlib, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

def md5(path):
    h = hashlib.md5()
    with (gzip.open(path, 'rb') if str(path).endswith('.gz') else open(path, 'rb')) as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def _norm_chr(x):
    """Normalize chromosome strings to 1-10, dropping others."""
    if x is None:
        return None
    s = str(x).lower()
    if s.startswith("chromosome_"):
        s = s[11:]
    elif s.startswith("chr"):
        s = s[3:]
    if s in {"mt", "pt", "mitochondrion", "chloroplast"}:
        return None
    try:
        i = int(s)
        return str(i) if 1 <= i <= 10 else None
    except ValueError:
        return None

def read_gff_genes(gff_path, max_rows=None):
    cols = ['seqid','src','type','start','end','score','strand','phase','attr']
    gff = pd.read_csv(gzip.open(gff_path,'rt') if str(gff_path).endswith('.gz') else gff_path,
                      sep='\t', comment='#', header=None, names=cols, nrows=max_rows)
    g = gff[gff['type'].isin(['gene','mRNA','transcript'])].copy()
    def gid(a):
        a = str(a)
        for part in a.split(';'):
            if part.startswith('ID='):
                v = part.split('=',1)[1]
                v = v.split(':',1)[-1]  # handle ID=gene:Zm00001d...
                return v
        return None
    g['gene_id'] = g['attr'].map(gid)
    g['chr_norm'] = g['seqid'].map(_norm_chr)
    g = g.dropna(subset=['gene_id','chr_norm']).copy()
    g['tss'] = np.where(g['strand'] == '-', g['end'], g['start'])
    g['tss'] = g['tss'].astype(int)
    return g[['gene_id','chr_norm','tss']].drop_duplicates('gene_id')

def read_pvar(pvar_path):
    pvar = pd.read_csv(pvar_path, sep='\t', comment='#', header=None,
                       names=['CHR','POS','ID','REF','ALT'])
    pvar['chr_norm'] = pvar['CHR'].map(_norm_chr)
    return pvar.dropna(subset=['chr_norm'])[['chr_norm','POS','ID']].rename(columns={'POS':'pos'})

def assert_agpv4(series):
    ok = series.astype(str).str.match(r'^Zm00001d\d{5}$')
    if ok.mean() < 1.0:
        bad = series.loc[~ok].unique()[:10]
        raise SystemExit(f'[AGPv4-IDs-REQUIRED] Found non-AGPv4 IDs (e.g., {list(bad)})')
    return True

def compute_cis_counts(genes_df, pvar, window_bp=1_000_000):
    """For each gene, count SNPs within a window around its TSS."""
    out = []
    for chrom, genes_on_chrom in genes_df.groupby('chr_norm', sort=False):
        snp_pos = pvar.loc[pvar['chr_norm'] == chrom, 'pos'].astype(int).values
        if snp_pos.size == 0:
            out.append(genes_on_chrom.assign(n_cis_snps=0))
            continue
        
        snp_pos.sort()
        
        tss = genes_on_chrom['tss'].values
        low_bound = tss - window_bp
        high_bound = tss + window_bp
        
        left_indices = np.searchsorted(snp_pos, low_bound, side='left')
        right_indices = np.searchsorted(snp_pos, high_bound, side='right')
        
        counts = right_indices - left_indices
        out.append(genes_on_chrom.assign(n_cis_snps=counts))
    
    if not out:
        return pd.DataFrame(columns=['gene_id', 'chr_norm', 'tss', 'n_cis_snps'])
    
    return pd.concat(out, ignore_index=True)[['gene_id', 'chr_norm', 'tss', 'n_cis_snps']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gff', required=True)
    ap.add_argument('--pvar', required=True)
    ap.add_argument('--gene_list', required=True, help='CSV with column gene_id')
    ap.add_argument('--expr', required=False, help='Expression table to sanity-check gene IDs (optional)')
    ap.add_argument('--window_kb', type=int, default=1000, help="Mapping window for features, in kb.")
    ap.add_argument('--local_kb', type=int, default=20, help="Local window for reporting, in kb.")
    ap.add_argument('--outdir', default=r"C:\Users\ms\Desktop\gwas\output\reports")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    manifest = {'gff': args.gff, 'pvar': args.pvar, 'gene_list': args.gene_list,
                'gff_md5': md5(args.gff), 'pvar_md5': md5(args.pvar)}

    # Load
    genes = pd.read_csv(args.gene_list)
    if 'gene_id' not in genes.columns:
        raise SystemExit('[INPUT] gene_list must have column gene_id')
    assert_agpv4(genes['gene_id'])
    gff = read_gff_genes(args.gff)
    pvar = read_pvar(args.pvar)

    # Join coords
    gc = genes[['gene_id']].merge(gff, on='gene_id', how='left')
    missing = gc['chr_norm'].isna().sum()
    if missing:
        raise SystemExit(f'[GFF-MISS] {missing} / {len(gc)} genes not found in GFF (AGPv4).')

    # Coverage at mapping window
    cis = compute_cis_counts(gc, pvar, window_bp=args.window_kb*1000)
    cov_any = (cis['n_cis_snps'] >= 1).mean()
    cov_ge8 = (cis['n_cis_snps'] >= 8).mean()
    cis.to_csv(outdir/'cis_mapping_coverage.csv', index=False)

    # Local(±local_kb) label stats for reporting
    local = compute_cis_counts(gc, pvar, window_bp=args.local_kb*1000)
    local.rename(columns={'n_cis_snps':'n_local_±%dkb'%args.local_kb}, inplace=True)
    local.to_csv(outdir/'local20kb_label_counts.csv', index=False)

    # Optional expression header sniff (sanity on gene-ID format)
    expr_head = None
    if args.expr and os.path.exists(args.expr):
        try:
            df = pd.read_csv(args.expr, nrows=1)
            expr_head = df.columns[:10].tolist()
        except Exception as e:
            expr_head = f'error: {e!s}'

    # Write Markdown report
    md_path = outdir/'preflight_report.md'
    expr_head_line = f"- **Expr header (first 10 cols)**: `{expr_head}`\n" if expr_head else ""
    
    report_md = f"""# Preflight Data Interop Report

- **Assembly/GFF**: `{args.gff}` (md5 `{manifest['gff_md5']}`)
- **Genotype PVAR**: `{args.pvar}` (md5 `{manifest['pvar_md5']}`)
- **Gene list**: `{args.gene_list}` (N={len(genes)})
{expr_head_line}
## Chromosome normalization policy

All contigs normalized to `1..10`; organelles/scaffolds dropped.

## Coverage

- Mapping window (±{args.window_kb} kb):
  - Genes with ≥1 cis SNP: **{cov_any:.1%}**
  - Genes with ≥8 cis SNPs: **{cov_ge8:.1%}**
- Local label window (±{args.local_kb} kb): counts in `local20kb_label_counts.csv`

## Gates

- **FAIL** if ≥1 cis coverage < **80%** at ±{args.window_kb} kb
- **FAIL** if any gene ID not AGPv4 (`Zm00001d#####`)
"""
    md_path.write_text(report_md, encoding='utf-8')
    
    # Exit code gate
    if cov_any < 0.80:
        print(f"[CIS-COVERAGE-FAIL] Only {cov_any:.1%} genes have ≥1 cis SNP at ±{args.window_kb} kb; see {md_path}")
        sys.exit(2)
    print(f"[OK] Coverage looks good. Report: {md_path}")

if __name__ == "__main__":
    main()
