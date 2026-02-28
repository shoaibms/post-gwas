#!/usr/bin/env python3
"""
Figure S5: Individual Locus Deep Dives
Comprehensive ieQTL visualization with distinct panels for genomic context, 
expression trajectories, and genotype frequencies
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats
import gzip
import warnings
warnings.filterwarnings('ignore')

# Import colour configuration
try:
    from colour_config import colors
except ImportError:
    colors = None

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
OUTPUT_DIR = BASE_DIR / "figures" / "output"
SOURCE_DIR = OUTPUT_DIR / "source_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Input files
EXPR_MATRIX = BASE_DIR / "output" / "expression_matrix_drought.csv"
LEAD_SNPS = BASE_DIR / "figures" / "output" / "source_data" / "figS5_candidate_loci.csv"
VCF_FILE = BASE_DIR / "data" / "maize" / "zea_mays_miss0.6_maf0.05.recode.vcf.gz"
TF_BED = BASE_DIR / "output" / "week3_tf_binding" / "filtered_tf_sites.bed"
GFF3 = BASE_DIR / "data" / "maize" / "Zea_mays.B73_RefGen_v4.gff3.gz"

# Metadata paths
SAMPLE_METADATA = BASE_DIR / "data" / "maize" / "process" / "metadata" / "sample_metadata.csv"
IDMAP = BASE_DIR / "data" / "maize" / "process" / "metadata" / "sample_to_vcf_id.csv"
COHORT_FILE = BASE_DIR / "output" / "cohort" / "core_all3_env.csv"

# Expression files
EXPR_FILES = {
    'WW': BASE_DIR / "output" / "data_filtered" / "WW_209-Uniq_FPKM.agpv4.txt.gz",
    'WS1': BASE_DIR / "output" / "data_filtered" / "WS1_208-uniq_FPKM.agpv4.txt.gz",
    'WS2': BASE_DIR / "output" / "data_filtered" / "WS2_210-uniq_FPKM.agpv4.txt.gz"
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10.5,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'medium',
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#008080',
    'axes.labelcolor': '#1b4332',
    'text.color': '#1b4332',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'xtick.color': '#008080',
    'ytick.color': '#008080',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 9.5,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#B2DFDB',
    'legend.fancybox': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

print("="*80)
print("FIGURE S5: INDIVIDUAL LOCUS DEEP DIVES")
print("="*80)

# =============================================================================
# VCF PARSER
# =============================================================================

def parse_vcf_genotypes(vcf_file, chr_num, pos):
    """Parse VCF to extract genotypes"""
    print(f"    Parsing VCF for chr{chr_num}:{pos}...")
    
    try:
        with gzip.open(vcf_file, 'rt') as f:
            samples = None
            
            for line in f:
                if line.startswith('##'):
                    continue
                
                if line.startswith('#CHROM'):
                    parts = line.strip().split('\t')
                    samples = parts[9:]
                    print(f"      VCF samples: {len(samples)}")
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 10:
                    continue
                
                vcf_chr = parts[0].replace('chr', '')
                vcf_pos = int(parts[1])
                
                if str(vcf_chr) == str(chr_num) and vcf_pos == pos:
                    print(f"      [OK] Found SNP")
                    
                    format_field = parts[8]
                    format_keys = format_field.split(':')
                    
                    try:
                        gt_index = format_keys.index('GT')
                    except ValueError:
                        print(f"      [WARN] No GT field")
                        return None, None
                    
                    genotypes = []
                    for sample_data in parts[9:]:
                        sample_fields = sample_data.split(':')
                        
                        if len(sample_fields) > gt_index:
                            gt_string = sample_fields[gt_index]
                        else:
                            genotypes.append(np.nan)
                            continue
                        
                        if '/' in gt_string:
                            alleles = gt_string.split('/')
                        elif '|' in gt_string:
                            alleles = gt_string.split('|')
                        else:
                            genotypes.append(np.nan)
                            continue
                        
                        if '.' in alleles[0] or '.' in alleles[1]:
                            genotypes.append(np.nan)
                        else:
                            try:
                                geno = int(alleles[0]) + int(alleles[1])
                                genotypes.append(geno)
                            except (ValueError, IndexError):
                                genotypes.append(np.nan)
                    
                    geno_df = pd.DataFrame({
                        'vcf_sample_id': samples,
                        'genotype': genotypes
                    })
                    
                    valid_genos = [g for g in genotypes if not np.isnan(g)]
                    if len(valid_genos) > 0:
                        n_alt_alleles = sum(valid_genos)
                        total_alleles = len(valid_genos) * 2
                        maf = n_alt_alleles / total_alleles
                    else:
                        maf = None
                    
                    geno_counts = pd.Series(valid_genos).value_counts().sort_index()
                    print(f"      Valid genotypes: {len(valid_genos)}/{len(genotypes)}")
                    print(f"      MAF = {maf:.3f}" if maf else "      MAF = NA")
                    print(f"      Distribution: {dict(geno_counts)}")
                    
                    return geno_df, maf
        
        print(f"      [WARN] SNP not found")
        return None, None
        
    except Exception as e:
        print(f"      [ERROR] {e}")
        return None, None


def parse_gff_gene(gff_file, gene_id):
    """Extract gene coordinates from GFF3 - checks gene_id attribute"""
    try:
        with gzip.open(gff_file, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                seqid, source, feat, start, end, score, strand, phase, attrs = parts
                
                if feat != 'gene':
                    continue
                
                attr_dict = {}
                for kv in attrs.split(';'):
                    if '=' in kv:
                        key, val = kv.split('=', 1)
                        attr_dict[key] = val
                
                # Check gene_id attribute (plain ID) or ID attribute (with prefix)
                if (attr_dict.get('gene_id') == gene_id or 
                    attr_dict.get('ID') == gene_id or
                    attr_dict.get('ID') == f'gene:{gene_id}'):
                    return {
                        'chr': seqid.replace('chr', '').replace('Chr', ''),
                        'start': int(start),
                        'end': int(end),
                        'strand': strand
                    }
    except Exception as e:
        print(f"      [ERROR] GFF3 error: {e}")
    
    return None


def find_overlapping_tf_peaks(chr_num, pos, peaks_df):
    """Find TF peaks that overlap the SNP position"""
    chr_peaks = peaks_df[peaks_df['chr'] == str(chr_num)]
    overlaps = chr_peaks[
        (chr_peaks['start'] <= pos) & (chr_peaks['end'] >= pos)
    ]
    return overlaps

# =============================================================================
# LOAD SAMPLE METADATA
# =============================================================================
print("\n1. Loading sample metadata...")

if SAMPLE_METADATA.exists():
    print(f"  [OK] Using: {SAMPLE_METADATA}")
    sample_metadata = pd.read_csv(SAMPLE_METADATA)
    print(f"  Loaded: {len(sample_metadata)} samples")
else:
    print(f"  [INFO] Creating metadata from source files...")
    
    metadata_records = []
    for condition, fpath in EXPR_FILES.items():
        if not fpath.exists():
            continue
        
        df = pd.read_csv(fpath, sep='\t', compression='gzip', usecols=[0])
        sample_ids = df.iloc[:, 0].astype(str).tolist()
        
        for sample_id in sample_ids:
            accession = sample_id.split('_')[0] if '_' in sample_id else sample_id
            metadata_records.append({
                'sample_id': sample_id,
                'accession': accession,
                'condition': condition
            })
    
    sample_metadata = pd.DataFrame(metadata_records)
    
    SAMPLE_METADATA.parent.mkdir(parents=True, exist_ok=True)
    sample_metadata.to_csv(SAMPLE_METADATA, index=False)
    print(f"  [OK] Created: {SAMPLE_METADATA}")

sample_metadata['condition'] = pd.Categorical(
    sample_metadata['condition'], 
    categories=['WW', 'WS1', 'WS2'], 
    ordered=True
)

print(f"  Conditions: {dict(sample_metadata['condition'].value_counts())}")
print(f"  Unique accessions: {sample_metadata['accession'].nunique()}")

# --- Filter to 198-accession analysis cohort ---
print("\n  Filtering to analysis cohort (n=198)...")
if COHORT_FILE.exists():
    cohort_df = pd.read_csv(COHORT_FILE)
    analysis_cohort = cohort_df['accession'].tolist()
    print(f"  [OK] Loaded analysis cohort: {len(analysis_cohort)} accessions")
    sample_metadata = sample_metadata[sample_metadata['accession'].isin(analysis_cohort)]
    print(f"  After cohort filter: {len(sample_metadata)} samples, {sample_metadata['accession'].nunique()} accessions")
else:
    print(f"  [WARN] Cohort file not found: {COHORT_FILE}")

# =============================================================================
# LOAD EXPRESSION DATA
# =============================================================================
print("\n2. Loading expression data...")

if not EXPR_MATRIX.exists():
    raise FileNotFoundError(f"Expression file not found: {EXPR_MATRIX}")

expr_full = pd.read_csv(EXPR_MATRIX)
print(f"[OK] Expression matrix: {expr_full.shape[0]} samples x {expr_full.shape[1]-1} genes")

if 'ID' not in expr_full.columns:
    raise ValueError("Expression matrix missing 'ID' column")

expr_full = expr_full.merge(
    sample_metadata[['sample_id', 'condition', 'accession']], 
    left_on='ID', 
    right_on='sample_id', 
    how='inner'
)

print(f"  Merged: {len(expr_full)} samples")
print(f"  Conditions: {dict(expr_full['condition'].value_counts())}")

# =============================================================================
# SAMPLE ID MAPPING
# =============================================================================
print("\n3. Sample ID mapping...")

if IDMAP.exists():
    print(f"  [OK] Using ID map: {IDMAP}")
    id_map = pd.read_csv(IDMAP)
    expr_full = expr_full.merge(id_map, on='ID', how='left')
else:
    print(f"  No ID map found - assuming expression ID = VCF sample ID")
    expr_full['vcf_sample_id'] = expr_full['ID']

vcf_id_coverage = expr_full['vcf_sample_id'].notna().mean()
print(f"  VCF ID coverage: {vcf_id_coverage:.1%}")
assert vcf_id_coverage > 0.95, f"Many samples missing VCF IDs ({vcf_id_coverage:.1%})"

# =============================================================================
# LOAD LEAD SNPS
# =============================================================================
print("\n4. Loading lead SNPs...")

if not LEAD_SNPS.exists():
    raise FileNotFoundError(f"Lead SNPs file not found: {LEAD_SNPS}")

loci = pd.read_csv(LEAD_SNPS)
print(f"[OK] Lead SNPs: {len(loci)}")

# =============================================================================
# LOAD TF PEAKS
# =============================================================================
print("\n5. Loading TF peaks...")

if TF_BED.exists():
    tf_peaks = pd.read_csv(TF_BED, sep='\t', 
                           names=['chr', 'start', 'end', 'name', 'score', 'strand'])
    print(f"[OK] TF peaks: {len(tf_peaks)}")
else:
    print(f"[WARN] TF BED not found")
    tf_peaks = pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score', 'strand'])

# =============================================================================
# COLOR PALETTE
# =============================================================================
if colors:
    print("  Using color palette from colour_config.py")
    # Genotype colours: mediumturquoise, palegreen, seagreen
    GENO_COLORS = ['#48D1CC', '#98FB98', '#2E8B57']
    GENE_FACECOLOR = colors.palegreen
    GENE_EDGECOLOR = colors.darkslategray
    SNP_COLOR = colors.mediumturquoise
    TF_FACECOLOR = colors.yellowgreen
    TF_EDGECOLOR = colors.seagreen
    STATS_EDGECOLOR = colors.grid_color
    ERROR_COLOR = colors.darkslategray
else:
    print("  Using fallback color palette")
    GENO_COLORS = ['#48D1CC', '#98FB98', '#2E8B57']
    GENE_FACECOLOR = '#98FB98'
    GENE_EDGECOLOR = '#2F4F4F'
    SNP_COLOR = '#48D1CC'
    TF_FACECOLOR = '#9ACD32'
    TF_EDGECOLOR = '#2E8B57'
    STATS_EDGECOLOR = '#B2DFDB'
    ERROR_COLOR = '#2F4F4F'

# =============================================================================
# CREATE FIGURE
# =============================================================================
print("\n6. Creating figure...")

fig = plt.figure(figsize=(15, 5 * len(loci)))
gs = GridSpec(len(loci), 9, figure=fig, hspace=0.4, wspace=0.4)

# =============================================================================
# PROCESS EACH LOCUS
# =============================================================================

for locus_idx, (_, locus) in enumerate(loci.iterrows()):
    gene = locus['gene']
    snp = locus['snp']
    chr_num = str(locus['chr']).replace('chr', '')
    pos = int(locus['pos'])
    beta = float(locus.get('beta_GxE', 0))
    pval = float(locus.get('p_GxE', 1))
    
    print(f"\n{'='*60}")
    print(f"LOCUS {locus_idx+1}: {gene} x chr{chr_num}:{pos}")
    print(f"{'='*60}")
    
    row = locus_idx
    
    # ========================================================================
    # PANEL A: GENOMIC CONTEXT
    # ========================================================================
    print(f"  Panel A: Genomic context...")
    ax_a = fig.add_subplot(gs[row, 0:3])
    
    gene_coords = parse_gff_gene(GFF3, gene)
    
    if gene_coords:
        gene_start = gene_coords['start']
        gene_end = gene_coords['end']
        gene_mid = (gene_start + gene_end) / 2
        
        window = 50000
        plot_start = max(1, int(gene_mid - window))
        plot_end = int(gene_mid + window)
        
        gene_rect = Rectangle(
            (gene_start, 0.4), gene_end - gene_start, 0.2,
            facecolor=GENE_FACECOLOR, edgecolor=GENE_EDGECOLOR, linewidth=2
        )
        ax_a.add_patch(gene_rect)
        
        ax_a.text(gene_mid, 0.5, gene, ha='center', va='center',
                 fontsize=9, fontweight='bold')
        
        ax_a.axvline(pos, color=SNP_COLOR, linestyle='--', linewidth=2, 
                    label=f'SNP: {snp}', alpha=0.8)
        
        if len(tf_peaks) > 0:
            overlapping = find_overlapping_tf_peaks(chr_num, pos, tf_peaks)
            if len(overlapping) > 0:
                for _, tf in overlapping.iterrows():
                    tf_rect = Rectangle(
                        (tf['start'], 0.7), tf['end'] - tf['start'], 0.15,
                        facecolor=TF_FACECOLOR, edgecolor=TF_EDGECOLOR, 
                        linewidth=1.5, alpha=0.7
                    )
                    ax_a.add_patch(tf_rect)
                print(f"    TF overlaps: {len(overlapping)}")
        
        ax_a.set_xlim(plot_start, plot_end)
        ax_a.set_ylim(0, 1)
        ax_a.set_xlabel(f'Position on chr{chr_num} (bp)', fontsize=10, fontweight='bold')
        ax_a.set_yticks([])
        ax_a.spines['left'].set_visible(False)
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)
        ax_a.legend(loc='upper right', fontsize=8)
    else:
        ax_a.text(0.5, 0.5, 'Gene Coordinates Not Found', 
                 ha='center', va='center', transform=ax_a.transAxes,
                 fontsize=10, color=ERROR_COLOR)
        ax_a.axis('off')
    
    ax_a.text(-0.08, 1.05, chr(65 + locus_idx*3), 
             transform=ax_a.transAxes,
             fontsize=14, fontweight='bold', va='top')
    
    # ========================================================================
    # PANEL B: EXPRESSION TRAJECTORIES
    # ========================================================================
    print(f"  Panel B: Expression trajectories...")
    ax_b = fig.add_subplot(gs[row, 3:6])
    
    geno_df, real_maf = parse_vcf_genotypes(VCF_FILE, chr_num, pos)
    
    if geno_df is not None and real_maf is not None and gene in expr_full.columns:
        gene_data = expr_full[['ID', 'vcf_sample_id', 'condition', 'accession', gene]].copy()
        gene_data = gene_data.rename(columns={gene: 'expression'})
        
        panel_data = gene_data.merge(
            geno_df, 
            on='vcf_sample_id', 
            how='inner'
        )
        panel_data = panel_data.dropna(subset=['genotype'])
        panel_data['genotype'] = panel_data['genotype'].astype(int)
        
        print(f"    Matched samples: {len(panel_data)}")
        for cond in ['WW', 'WS1', 'WS2']:
            cond_data = panel_data[panel_data['condition'] == cond]
            geno_counts = cond_data['genotype'].value_counts().sort_index()
            print(f"      {cond}: n={len(cond_data)}, genotypes={dict(geno_counts)}")
        
        if len(panel_data) > 0:
            trajectories = []
            for gt in [0, 1, 2]:
                gt_data = panel_data[panel_data['genotype'] == gt]
                if len(gt_data) >= 1:
                    trajectory = {'genotype': gt, 'n': len(gt_data)}
                    
                    for cond in ['WW', 'WS1', 'WS2']:
                        cond_data = gt_data[gt_data['condition'] == cond]['expression']
                        trajectory[f'{cond}_mean'] = cond_data.mean()
                        trajectory[f'{cond}_sem'] = cond_data.sem()
                        trajectory[f'{cond}_n'] = len(cond_data)
                    
                    trajectories.append(trajectory)
            
            x_pos = [0, 1, 2]
            x_labels = ['WW', 'WS1', 'WS2']
            
            for traj in trajectories:
                gt = traj['genotype']
                means = [traj['WW_mean'], traj['WS1_mean'], traj['WS2_mean']]
                sems = [traj['WW_sem'], traj['WS1_sem'], traj['WS2_sem']]
                
                gt_data = panel_data[panel_data['genotype'] == gt]
                n_individuals = gt_data['accession'].nunique()
                gt_label = ['Ref/Ref', 'Ref/Alt', 'Alt/Alt'][gt]
                label = f'{gt_label} (N={n_individuals})'
                
                ax_b.plot(x_pos, means, 'o-', linewidth=2.5, markersize=10,
                         color=GENO_COLORS[gt], label=label,
                         markeredgecolor='white', markeredgewidth=2)
                
                ax_b.errorbar(x_pos, means, yerr=sems, fmt='none',
                             ecolor=GENO_COLORS[gt], alpha=0.5, linewidth=2)
            
            ax_b.set_xticks(x_pos)
            ax_b.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
            ax_b.set_xlabel('Treatment', fontsize=10, fontweight='bold')
            ax_b.set_ylabel('Expression (FPKM)', fontsize=10, fontweight='bold')
            ax_b.grid(alpha=0.3, linewidth=0.5)
            ax_b.legend(loc='best', fontsize=8, framealpha=0.9)
            
            stats_text = f"beta_GxE = {beta:.3f}\np_GxE = {pval:.2e}\nMAF = {real_maf:.3f}"
            ax_b.text(0.98, 0.02, stats_text, transform=ax_b.transAxes,
                     fontsize=7, va='bottom', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.9, edgecolor=STATS_EDGECOLOR))
            
            panel_source = panel_data[['ID', 'condition', 'genotype', 'expression']].copy()
            panel_source.to_csv(
                SOURCE_DIR / f"figS5_locus{locus_idx+1}_{gene}_trajectories.csv", 
                index=False
            )
            
            summary_lines = [
                f"Locus {locus_idx+1}: {gene} x chr{chr_num}:{pos}",
                f"Total samples: {len(panel_data)}",
                f"MAF: {real_maf:.3f}",
                f"Effect size (beta_GxE): {beta:.3f}",
                f"P-value (p_GxE): {pval:.2e}",
                "",
                "Samples by genotype x condition:"
            ]
            for gt in sorted(panel_data['genotype'].unique()):
                gt_label = ['Ref/Ref', 'Ref/Alt', 'Alt/Alt'][int(gt)]
                summary_lines.append(f"  {gt_label} (genotype {int(gt)}):")
                gt_data = panel_data[panel_data['genotype'] == gt]
                for cond in ['WW', 'WS1', 'WS2']:
                    n = len(gt_data[gt_data['condition'] == cond])
                    summary_lines.append(f"    {cond}: n={n}")
            
            with open(SOURCE_DIR / f"figS5_locus{locus_idx+1}_summary.txt", 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            print(f"    [OK] Trajectories plotted: {len(trajectories)} genotype groups")
        else:
            ax_b.text(0.5, 0.5, 'No Overlapping Data', 
                     ha='center', va='center', transform=ax_b.transAxes,
                     fontsize=10, color=ERROR_COLOR)
    else:
        ax_b.text(0.5, 0.5, 'Data Not Available', 
                 ha='center', va='center', transform=ax_b.transAxes,
                 fontsize=10, color=ERROR_COLOR)
    
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    
    ax_b.text(-0.12, 1.05, chr(65 + locus_idx*3 + 1), 
             transform=ax_b.transAxes,
             fontsize=14, fontweight='bold', va='top')
    
    # ========================================================================
    # PANEL C: GENOTYPE FREQUENCY
    # ========================================================================
    print(f"  Panel C: Genotype frequency...")
    ax_c = fig.add_subplot(gs[row, 6:9])
    
    if geno_df is not None and 'panel_data' in locals() and len(panel_data) > 0:
        geno_counts = panel_data['genotype'].value_counts().sort_index()
        geno_labels = ['Ref/Ref', 'Ref/Alt', 'Alt/Alt']
        
        counts = [geno_counts.get(i, 0) for i in range(3)]
        y_pos = np.arange(len(geno_labels))
        
        bars = ax_c.barh(y_pos, counts, 
                         color=[GENO_COLORS[i] for i in range(3)],
                         edgecolor='white', linewidth=2)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                ax_c.text(count + max(counts)*0.02, i, f'n={count}',
                         va='center', ha='left', fontsize=9, fontweight='bold')
        
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(geno_labels, fontsize=10, fontweight='bold')
        ax_c.set_xlabel('Sample Count', fontsize=10, fontweight='bold')
        ax_c.set_xlim(0, max(counts) * 1.2)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.grid(axis='x', alpha=0.3, linewidth=0.5)
        
        if counts[1] == 0:
            ax_c.text(0.5, -0.15, 'Note: No heterozygotes in cohort',
                     transform=ax_c.transAxes, ha='center', va='top',
                     fontsize=8, style='italic', color='#B8B8B8')
    else:
        ax_c.text(0.5, 0.5, 'No Genotype Data', ha='center', va='center',
                 transform=ax_c.transAxes, fontsize=10, color=ERROR_COLOR)
        ax_c.axis('off')
    
    ax_c.text(-0.15, 1.05, chr(65 + locus_idx*3 + 2), 
             transform=ax_c.transAxes,
             fontsize=14, fontweight='bold', va='top')

# =============================================================================
# SAVE FIGURE
# =============================================================================
print("\n7. Saving figure...")

fig.suptitle('', fontsize=14, fontweight='bold', y=0.97)

output_png = OUTPUT_DIR / "figure_s5_locus_examples_FINAL.png"
output_pdf = OUTPUT_DIR / "figure_s5_locus_examples_FINAL.pdf"

plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')

print(f"[OK] PNG: {output_png}")
print(f"[OK] PDF: {output_pdf}")

print("\n" + "="*80)
print("FIGURE S5 COMPLETE")
print("="*80)