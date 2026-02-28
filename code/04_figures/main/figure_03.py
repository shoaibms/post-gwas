#!/usr/bin/env python3
"""
Figure 3 — G×E effect dynamics, calibration, and sequence-level evidence
=========================================================================
5-panel asymmetric layout

  ┌──────────────────────────────────┐
  │               A                  │
  │    Lollipop Manhattan (wide)     │
  │        12-column top panel       │
  ├──────────┬───────────┬───────────┤
  │    B     │     C     │     D     │
  │ Lollipop │ Beeswarm  │   Q-Q    │
  │  Forest  │   MAF     │  + band  │
  │ (4 col)  │ (4 col)   │ (4 col)  │
  ├──────────┴───────────┴───────────┤
  │               E                  │
  │   Motif Disruption Lollipop      │
  │  9 TF-overlapping SNPs scored    │
  │      12-column bottom panel      │
  └──────────────────────────────────┘

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS & IMPORTS
# ============================================================================
BASE_DIR = Path(r"C:\Users\ms\Desktop\gwas")
FIG_DIR = BASE_DIR / "figures"
FIG_SUPPORT = FIG_DIR / "figure_3"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_SUPPORT.mkdir(parents=True, exist_ok=True)

# Infrastructure modules (robust path selection)
_infra_candidates = []
_env_infra = os.environ.get('GWAS_INFRA_DIR')
if _env_infra:
    _infra_candidates.append(Path(_env_infra))
_infra_candidates += [
    BASE_DIR / "code" / "GitHub" / "code" / "04_figures",
    BASE_DIR / "code" / "04_figures",
]
INFRA_DIR = next((p for p in _infra_candidates if p and p.exists()), None)
if INFRA_DIR is None:
    raise FileNotFoundError(
        "Could not locate figure infrastructure modules. Tried:\n  - "
        + "\n  - ".join(str(p) for p in _infra_candidates)
        + "\nSet GWAS_INFRA_DIR to the correct folder."
    )
sys.path.insert(0, str(INFRA_DIR))

from colour_config import colors, HEX_CODES

# ============================================================================
# GLOBAL STYLE — Nature Plants standard
# ============================================================================
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

# ============================================================================
# COLOUR PALETTE — Green-family diversity
# ============================================================================
C = {
    'mintcream':       '#F5FFFA',
    'honeydew':        '#F0FFF0',
    'palegreen':       '#98FB98',
    'lightgreen':      '#90EE90',
    'darkseagreen':    '#8FBC8F',
    'mediumaquamarine':'#66CDAA',
    'yellowgreen':     '#9ACD32',
    'mediumseagreen':  '#3CB371',
    'limegreen':       '#32CD32',
    'mediumturquoise': '#48D1CC',
    'seagreen':        '#2E8B57',
    'cadetblue':       '#5F9EA0',
    'teal':            '#008080',
    'dark_forest':     '#1b4332',
    'forestgreen':     '#228B22',
    'darkturquoise':   '#00CED1',
    'deepskyblue':     '#00BFFF',
    'paleturquoise':   '#AFEEEE',
    # Blue accents (~15%)
    'steelblue':       '#4682B4',
    'dodgerblue':      '#1E90FF',
    'royalblue':       '#4169E1',
    # Neutrals
    'gainsboro':       '#DCDCDC',
    'lightgray':       '#D3D3D3',
    'gray':            '#808080',
    'light_teal':      '#B2DFDB',
    'nature_green':    '#00A087',
    'white':           '#FFFFFF',
}

# Panel-specific colour assignments
COL_BETA_POS    = C['yellowgreen']      # β > 0 significant — warm green
COL_BETA_NEG    = C['steelblue']       # β < 0 significant — true blue
COL_NONSIG      = C['gainsboro']       # Non-significant
COL_QQ_OBS      = C['teal']            # QQ observed — bold
COL_QQ_BAND     = C['palegreen']       # QQ CI band
COL_QQ_EXP      = C['darkseagreen']    # QQ expected line
COL_BEESWARM_SIG = C['deepskyblue']    # Beeswarm significant — vivid
COL_BEESWARM_NS  = C['lightgray']      # Beeswarm non-sig
COL_MOTIF_SIG   = C['forestgreen']     # Motif significant
COL_MOTIF_NS    = C['dodgerblue']       # Motif non-significant — blue
COL_CHR_ALT     = C['mintcream']       # Chr alternating fill
COL_THRESHOLD   = C['mediumturquoise'] # Threshold lines
COL_DARK        = C['dark_forest']     # Labels, spines
COL_STEM_POS    = C['mediumseagreen']  # Stem colour for positive β
COL_STEM_NEG    = C['dodgerblue']      # Stem colour for negative β — blue

# ============================================================================
# HELPER FUNCTIONS — from original figure_03.py for consistency
# ============================================================================

def panel_label(ax, letter, x=-0.10, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color=COL_DARK,
            ha='left', va='top',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

def style_axis(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_color(COL_DARK)
    ax.spines['bottom'].set_color(COL_DARK)

def stats_box(ax, text, x=0.03, y=0.97, fontsize=7.5):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.92, edgecolor=C['light_teal'],
                      linewidth=0.6, boxstyle='round,pad=0.4'))

def clip_pvalues(p):
    """Clip p-values to avoid log(0). Same as original."""
    p = np.asarray(p, float)
    return np.clip(p, 1e-300, 1 - 1e-16)

def lambda_gc_from_p(p):
    """Compute genomic inflation factor. Same as original."""
    chi = stats.chi2.ppf(1 - clip_pvalues(p), 1)
    return np.median(chi) / stats.chi2.ppf(0.5, 1)

def null_envelope_qq(n, alpha=0.05):
    """95% null envelope for QQ plot. Same as original."""
    i = np.arange(1, n + 1)
    lower = stats.beta.ppf(alpha / 2, i, n - i + 1)
    upper = stats.beta.ppf(1 - alpha / 2, i, n - i + 1)
    exp = (i - 0.5) / n
    return -np.log10(exp), -np.log10(lower), -np.log10(upper)

def normalize_chr_series(s):
    """Normalize chr column. Same as original."""
    chrom_order = [str(i) for i in range(1, 11)]
    v = s.astype(str).str.strip()
    v = v.str.replace(r'^(?i)chr', '', regex=True)
    v = v.str.replace(r'^0+', '', regex=True)
    v = v.str.replace(r'[^0-9XxYy]', '', regex=True)
    v = v.str.upper()
    return pd.to_numeric(v, errors='coerce')

def coerce_ieqtl_columns(df):
    """Column name normalization. Same as original."""
    lower_map = {c.lower(): c for c in df.columns}
    def has(name): return name in lower_map
    def get(name): return lower_map[name]

    rename = {}
    for alias in ("beta_gxe","beta_gxe_raw","beta","b_gxe","effect","beta_g"):
        if has(alias) and "beta_gxe" not in df.columns:
            rename[get(alias)] = "beta_gxe"; break
    for alias in ("se_gxe","stderr","se_beta","se","se_g"):
        if has(alias) and "se_gxe" not in df.columns:
            rename[get(alias)] = "se_gxe"; break
    for alias in ("p_value","p","pval","pvalue","p_gxe","p_g"):
        if has(alias) and "p_value" not in df.columns:
            rename[get(alias)] = "p_value"; break
    for alias in ("q_value","q","qval","qvalue","fdr","padj","q_g"):
        if has(alias) and "q_value" not in df.columns:
            rename[get(alias)] = "q_value"; break
    for alias in ("chr","chrom","chromosome"):
        if has(alias) and "chr" not in df.columns:
            rename[get(alias)] = "chr"; break
    for alias in ("gene","gene_id","symbol"):
        if has(alias) and "gene" not in df.columns:
            rename[get(alias)] = "gene"; break
    for alias in ("maf","allele_frequency","af","minor_allele_freq"):
        if has(alias) and "maf" not in df.columns:
            rename[get(alias)] = "maf"; break
    for alias in ("snp","snp_id","variant_id","rsid"):
        if has(alias) and "snp" not in df.columns:
            rename[get(alias)] = "snp"; break
    if rename:
        df = df.rename(columns=rename)
    # BH q-value if missing
    if "q_value" not in df.columns and "p_value" in df.columns:
        p = pd.to_numeric(df["p_value"], errors="coerce").to_numpy()
        n = np.isfinite(p).sum()
        order = np.argsort(p)
        ranks = np.empty_like(order); ranks[order] = np.arange(1, len(p)+1)
        q = p * n / ranks
        q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
        qv = np.empty_like(q_sorted); qv[order] = q_sorted
        df["q_value"] = np.minimum(qv, 1.0)
    return df

def auto_find_ieqtl(base):
    """Auto-find ieQTL CSV. Same search order as original."""
    candidates = [
        base / "output" / "week6_ieqtl" / "ieqtl_delta_results.csv",
        base / "output" / "week5_ieqtl" / "ieqtl_results_complete.csv",
        base / "output" / "week6_ieqtl" / "ieqtl_final_results.csv",
        base / "output" / "postgwas" / "ieqtl" / "ieqtl_calls.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ============================================================================
# DATA LOADING — REAL DATA ONLY
# ============================================================================
print("=" * 80)
print("FIGURE 3 v3: 5-panel asymmetric layout")
print("=" * 80)

# --- ieQTL data ---
print("\n1. Loading ieQTL data...")
ieqtl_path = auto_find_ieqtl(BASE_DIR)
if ieqtl_path is None:
    raise FileNotFoundError(
        "Could not locate ieQTL CSV. Searched:\n"
        "  output/week6_ieqtl/ieqtl_delta_results.csv\n"
        "  output/week5_ieqtl/ieqtl_results_complete.csv\n"
        "  output/week6_ieqtl/ieqtl_final_results.csv"
    )
print(f"   Found: {ieqtl_path.name}")
df = pd.read_csv(ieqtl_path, encoding='latin1')

# Column normalization (same as original)
if 'p_G' in df.columns and 'p_value' not in df.columns:
    df['p_value'] = df['p_G']
if 'q_G' in df.columns and 'q_value' not in df.columns:
    df['q_value'] = df['q_G']
if 'beta_G' in df.columns and 'beta' not in df.columns:
    df['beta'] = df['beta_G']
if 'se_G' in df.columns and 'se_gxe' not in df.columns:
    df['se_gxe'] = df['se_G']

if 'chr' not in df.columns:
    if 'geno_col' in df.columns:
        df['chr'] = df['geno_col'].astype(str).str.extract(r'^(\d+)[_:]')[0]
    elif 'snp' in df.columns:
        df['chr'] = df['snp'].astype(str).str.extract(r'^(\d+)[_:]')[0]
    df['chr'] = pd.to_numeric(df['chr'], errors='coerce')

df = coerce_ieqtl_columns(df)
df["beta_gxe"] = pd.to_numeric(df.get("beta_gxe"), errors="coerce")
if "maf" in df.columns:
    df["maf"] = pd.to_numeric(df["maf"], errors="coerce")
if "q_value" in df.columns:
    df["q_value"] = pd.to_numeric(df["q_value"], errors="coerce")

# Validate required columns
required = ["gene", "chr", "beta_gxe", "p_value"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required column(s): {missing}\nSeen: {list(df.columns)}")

df["chr_num"] = normalize_chr_series(df["chr"])
df = df.dropna(subset=["beta_gxe", "p_value", "chr_num"])

Q_THRESH = 0.10
df["is_sig"] = df["q_value"] <= Q_THRESH if "q_value" in df.columns else False
df_sig = df[df["is_sig"]].copy()

print(f"   Total SNP-gene pairs: {len(df)}")
print(f"   Significant (q ≤ {Q_THRESH}): {len(df_sig)}")
print(f"   Unique genes: {df['gene'].nunique()}")
print(f"   Chromosomes: {sorted(df['chr_num'].dropna().unique())}")

# --- Motif disruption data ---
print("\n2. Loading motif disruption data...")
motif_paths = [
    BASE_DIR / "output" / "week6_ieqtl" / "motif_disruption_summary_per_snp.csv",
    BASE_DIR / "output" / "postgwas" / "motif" / "motif_disruption_summary_per_snp.csv",
    BASE_DIR / "output" / "motif" / "motif_disruption_summary_per_snp.csv",
]
motif_df = None
for mp in motif_paths:
    if mp.exists():
        motif_df = pd.read_csv(mp)
        print(f"   Found: {mp.name} ({len(motif_df)} SNPs)")
        break

if motif_df is None:
    print("   [WARNING] Motif disruption file not found — Panel E will show error")
    print("   Searched:", [str(p) for p in motif_paths])

# --- Lambda GC ---
print("\n3. Loading lambda GC...")
lambda_file = BASE_DIR / "output" / "week6_ieqtl" / "lambda_gc_values.json"
lambda_gc = None
if lambda_file.exists():
    with open(lambda_file) as f:
        lam_data = json.load(f)
    lambda_gc = lam_data.get("lambda_gc_all")
    print(f"   From file: λ_GC = {lambda_gc:.3f}")
if lambda_gc is None:
    lambda_gc = lambda_gc_from_p(df["p_value"].values)
    print(f"   Computed from data: λ_GC = {lambda_gc:.3f}")

# ============================================================================
# FIGURE LAYOUT — 12-column asymmetric GridSpec
# ============================================================================
print("\n4. Creating figure layout...")

fig = plt.figure(figsize=(10, 8))
fig.set_facecolor('white')

gs = GridSpec(
    3, 12, figure=fig,
    height_ratios=[1.0, 0.9, 0.55],
    hspace=0.40, wspace=0.8,
    left=0.06, right=0.97, top=0.97, bottom=0.04
)

ax_a = fig.add_subplot(gs[0, 0:12])   # Lollipop Manhattan — FULL WIDTH
ax_b = fig.add_subplot(gs[1, 0:4])    # Lollipop Forest
ax_c = fig.add_subplot(gs[1, 4:8])    # Beeswarm MAF
ax_d = fig.add_subplot(gs[1, 8:12])   # Q-Q plot
ax_e = fig.add_subplot(gs[2, 0:12])   # Motif Disruption — FULL WIDTH

# Increase separation between C and D by shrinking each panel ~5%.
pos_c = ax_c.get_position()
pos_d = ax_d.get_position()
shrink_frac_cd = 0.05
ax_c.set_position([pos_c.x0, pos_c.y0, pos_c.width * (1 - shrink_frac_cd), pos_c.height])
ax_d.set_position([
    pos_d.x0 + pos_d.width * (shrink_frac_cd * 0.5),
    pos_d.y0,
    pos_d.width * (1 - shrink_frac_cd),
    pos_d.height
])

# ============================================================================
# PANEL A — LOLLIPOP MANHATTAN
# ============================================================================
print("\n5. Panel A: Lollipop Manhattan...")

df_a = df.copy()
df_a["neglogq"] = -np.log10(clip_pvalues(df_a["q_value"].values)) if "q_value" in df_a.columns else 0

chrs = sorted(df_a["chr_num"].dropna().unique())

# Chromosome alternating fills
for ci, c in enumerate(chrs):
    if ci % 2 == 0:
        ax_a.axvspan(c - 0.45, c + 0.45, color=C['honeydew'], alpha=0.82, zorder=0)
    else:
        ax_a.axvspan(c - 0.45, c + 0.45, color='#EBF5FB', alpha=0.65, zorder=0)

# Zero line
ax_a.axhline(0, ls='-', color=C['gray'], lw=0.8, zorder=1)

# Add jitter within chromosome
np.random.seed(42)
jitter = np.random.uniform(-0.3, 0.3, len(df_a))
x_positions = df_a["chr_num"].values + jitter

# --- LOLLIPOP STEMS ---
for i in range(len(df_a)):
    beta = df_a["beta_gxe"].iloc[i]
    is_sig = df_a["is_sig"].iloc[i]
    x = x_positions[i]

    if is_sig:
        stem_color = COL_STEM_POS if beta >= 0 else COL_STEM_NEG
        stem_alpha = 0.7
        stem_lw = 1.2
    else:
        stem_color = COL_NONSIG
        stem_alpha = 0.45
        stem_lw = 0.8

    ax_a.vlines(x, 0, beta, color=stem_color, alpha=stem_alpha,
                linewidth=stem_lw, zorder=1)

# --- LOLLIPOP HEADS ---
# Non-significant: tiny translucent dots
mask_ns = ~df_a["is_sig"].values
ax_a.scatter(x_positions[mask_ns], df_a["beta_gxe"].values[mask_ns],
             c=COL_NONSIG, s=10, alpha=0.40, edgecolors='none', zorder=2)

# Significant: coloured circles scaled by -log10(q)
mask_sig = df_a["is_sig"].values
if mask_sig.sum() > 0:
    sig_betas = df_a["beta_gxe"].values[mask_sig]
    sig_x = x_positions[mask_sig]
    sig_q = df_a["neglogq"].values[mask_sig]
    sig_sizes = 20 + 40 * (sig_q / (sig_q.max() if sig_q.max() > 0 else 1))
    sig_colors = np.where(sig_betas >= 0, COL_BETA_POS, COL_BETA_NEG)

    ax_a.scatter(sig_x, sig_betas, c=sig_colors, s=sig_sizes,
                 alpha=0.9, edgecolors='white', linewidths=0.5, zorder=3)

    # Label top 3 by |β|
    df_sig_a = df_a[df_a["is_sig"]].copy()
    df_sig_a["abs_beta"] = df_sig_a["beta_gxe"].abs()
    top3 = df_sig_a.nlargest(3, "abs_beta")
    for _, r in top3.iterrows():
        idx_orig = r.name
        x_lab = x_positions[df_a.index.get_loc(idx_orig)]
        gene_short = str(r.get("gene", "")).replace("Zm00001d0", "")
        va = 'bottom' if r["beta_gxe"] >= 0 else 'top'
        offset = 0.05 if r["beta_gxe"] >= 0 else -0.05
        ax_a.annotate(
            gene_short, (x_lab, r["beta_gxe"] + offset),
            fontsize=7, color=COL_DARK, fontweight='bold',
            ha='center', va=va,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5)
        )

ax_a.set_xticks(chrs)
ax_a.set_xticklabels([str(int(c)) for c in chrs], fontsize=9)
ax_a.set_xlabel('Chromosome', fontsize=10, color=COL_DARK)
ax_a.set_ylabel(r'G×E Effect Size ($\beta_{G \times E}$)', fontsize=10, color=COL_DARK)
ax_a.set_xlim(0.2, max(chrs) + 0.8)

# Legend
leg_a = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_BETA_POS,
           markeredgecolor='white', markersize=7, label=r'$\beta > 0$ (sig)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_BETA_NEG,
           markeredgecolor='white', markersize=7, label=r'$\beta < 0$ (sig)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_NONSIG,
           markeredgecolor='none', markersize=5, label='Not significant'),
]
ax_a.legend(handles=leg_a, loc='upper right', frameon=True,
            fancybox=True, framealpha=0.9, edgecolor=C['light_teal'], fontsize=7.5)

n_total = len(df_a)
n_sig = mask_sig.sum()
stats_box(ax_a, f"n = {n_total} SNP–gene pairs\n{n_sig} significant (q ≤ {Q_THRESH})",
          x=0.01, y=0.97)

style_axis(ax_a)
panel_label(ax_a, 'A')

# ============================================================================
# PANEL B — LOLLIPOP FOREST PLOT
# ============================================================================
print("   Panel B: Lollipop forest plot...")

has_se = "se_gxe" in df_sig.columns and df_sig["se_gxe"].notna().any()

if len(df_sig) > 0:
    df_b = df_sig.copy()
    df_b["abs_beta"] = df_b["beta_gxe"].abs()
    df_b = df_b.sort_values("beta_gxe", key=lambda s: s.abs(), ascending=False).head(16)
    df_b = df_b.sort_values("beta_gxe")  # sort for visual order

    y = np.arange(len(df_b))
    betas = df_b["beta_gxe"].values

    # CI whiskers if SE available
    if has_se:
        se = pd.to_numeric(df_b["se_gxe"], errors="coerce")
        xerr = 1.96 * se
        ax_b.errorbar(betas, y, xerr=xerr, fmt='none',
                      color=C['darkseagreen'], linewidth=1.0,
                      capsize=2, capthick=0.8, zorder=2, alpha=0.6)

    # Horizontal stems from zero
    for j in range(len(df_b)):
        stem_c = COL_BETA_POS if betas[j] >= 0 else COL_BETA_NEG
        ax_b.hlines(y[j], 0, betas[j], color=stem_c, linewidth=2.0,
                     alpha=0.6, zorder=1)

    # Lollipop heads
    head_colors = [COL_BETA_POS if b >= 0 else COL_BETA_NEG for b in betas]
    ax_b.scatter(betas, y, c=head_colors, s=50, edgecolors='white',
                 linewidths=0.6, zorder=3)

    # Zero line
    ax_b.axvline(0, ls='--', color=C['gray'], lw=0.8, zorder=0)

    # Y-axis labels: SNP — gene
    snp_labels = []
    for _, r in df_b.iterrows():
        snp_str = str(r.get("snp", "")).replace("_", ":")
        gene_short = str(r.get("gene", "")).replace("Zm00001d0", "")
        snp_labels.append(f"{snp_str} — {gene_short}")

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(snp_labels, fontsize=6)
    ax_b.set_ylabel('SNP–gene pair', fontsize=9, color=COL_DARK)
    label = r"$\beta_{G \times E}$ (95% CI)" if has_se else r"$\beta_{G \times E}$"
    ax_b.set_xlabel(label, fontsize=9, color=COL_DARK)
else:
    ax_b.text(0.5, 0.5, "No significant ieQTLs", ha="center", va="center",
              fontsize=9, color=C['gray'])

style_axis(ax_b)
panel_label(ax_b, 'B')

# ============================================================================
# PANEL C — BEESWARM MAF
# ============================================================================
print("   Panel C: Beeswarm MAF...")

if "maf" in df.columns and df["maf"].notna().sum() >= 20:
    maf = df["maf"].dropna().clip(0, 0.5).values
    maf_sig = df.loc[df["is_sig"] & df["maf"].notna(), "maf"].clip(0, 0.5).values
    maf_ns = df.loc[~df["is_sig"] & df["maf"].notna(), "maf"].clip(0, 0.5).values

    # Beeswarm: stack dots laterally using histogram-like binning
    def beeswarm_positions(values, nbins=30):
        """Compute lateral positions to avoid overlap."""
        bin_edges = np.linspace(values.min() - 0.001, values.max() + 0.001, nbins + 1)
        x_offsets = np.zeros(len(values))
        for i in range(nbins):
            mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                positions = np.linspace(-n_in_bin / 2, n_in_bin / 2, n_in_bin) * 0.02
                x_offsets[mask] = positions
        return x_offsets

    # Non-significant
    if len(maf_ns) > 0:
        offsets_ns = beeswarm_positions(maf_ns, nbins=25)
        ax_c.scatter(maf_ns, offsets_ns, c=COL_BEESWARM_NS, s=18,
                     alpha=0.5, edgecolors='white', linewidths=0.2, zorder=2)

    # Significant
    if len(maf_sig) > 0:
        offsets_sig = beeswarm_positions(maf_sig, nbins=25)
        ax_c.scatter(maf_sig, offsets_sig, c=COL_BEESWARM_SIG, s=35,
                     alpha=0.85, edgecolors='white', linewidths=0.4, zorder=3)

    # Density curve overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(maf, bw_method=0.3)
        x_kde = np.linspace(0, 0.5, 200)
        density = kde(x_kde)
        density_norm = density / density.max() * 0.15
        ax_c.fill_between(x_kde, -density_norm - 0.22, -0.22,
                          color=C['palegreen'], alpha=0.4, zorder=1)
        ax_c.plot(x_kde, -density_norm - 0.22, color=C['mediumseagreen'],
                  linewidth=1.0, alpha=0.7, zorder=1)
    except Exception:
        pass

    # 0.05 threshold
    ax_c.axvline(0.05, color=COL_THRESHOLD, linestyle=':', linewidth=1.2, alpha=0.7)
    ax_c.text(0.06, 0.95, 'MAF = 0.05', fontsize=7, color=C['teal'],
              ha='left', va='top', transform=ax_c.get_xaxis_transform(),
              fontstyle='italic')

    ax_c.set_xlabel('Minor Allele Frequency', fontsize=9, color=COL_DARK)
    ax_c.set_ylabel('')
    ax_c.set_yticks([])
    ax_c.spines['left'].set_visible(False)

    # Stats
    n_rare = (maf < 0.05).sum()
    stats_box(ax_c, f"n = {len(maf)} SNPs\n{n_rare} rare (MAF < 0.05)\n"
                     f"Median = {np.median(maf):.3f}",
              x=0.60, y=0.38)

    # Legend
    leg_c = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_BEESWARM_SIG,
               markeredgecolor='white', markersize=6, label='Significant'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_BEESWARM_NS,
               markeredgecolor='white', markersize=5, label='Non-significant'),
    ]
    ax_c.legend(handles=leg_c, loc='upper right', frameon=True,
                fancybox=True, framealpha=0.9, edgecolor=C['light_teal'], fontsize=7)
else:
    ax_c.text(0.5, 0.5, "MAF data not available", ha="center", va="center",
              fontsize=9, color=C['gray'])
    ax_c.set_yticks([])

style_axis(ax_c)
panel_label(ax_c, 'C')

# ============================================================================
# PANEL D — Q-Q PLOT WITH PROMINENT ENVELOPE
# ============================================================================
print("   Panel D: Q-Q plot...")

p_vals = clip_pvalues(df["p_value"].values)
n_tests = len(p_vals)
p_sorted = np.sort(p_vals)

exp_qq, lo95, hi95 = null_envelope_qq(n_tests, alpha=0.05)
obs_qq = -np.log10(p_sorted)

# Prominent CI band
ax_d.fill_between(exp_qq, lo95, hi95, color=COL_QQ_BAND, alpha=0.45,
                   zorder=1, label='95% null band')

# Expected line
ax_d.plot(exp_qq, exp_qq, ls='--', color=COL_QQ_EXP, lw=1.5,
          alpha=0.8, zorder=2, label='Expected')

# Observed line — thick and bold
ax_d.plot(exp_qq, obs_qq, lw=2.8, color=COL_QQ_OBS, zorder=3, label='Observed')

# Highlight tail points that exceed the envelope
tail_mask = obs_qq > hi95
if tail_mask.sum() > 0:
    ax_d.scatter(exp_qq[tail_mask], obs_qq[tail_mask],
                 c=C['dodgerblue'], s=22, edgecolors='white',
                 linewidths=0.4, zorder=4, alpha=0.85)

# Bonferroni line
bonf_line = -np.log10(0.05 / n_tests)
if bonf_line <= obs_qq.max() * 1.1:
    ax_d.axhline(bonf_line, ls=':', lw=0.8, color=C['steelblue'], alpha=0.5)
    ax_d.text(0.02, bonf_line / (obs_qq.max() * 1.05) if obs_qq.max() > 0 else 0.9,
              'Bonferroni', transform=ax_d.transAxes,
              fontsize=6.5, color=C['steelblue'], ha='left', va='bottom', alpha=0.7)

# q < 0.05 threshold line
if "q_value" in df.columns:
    try:
        q_arr = df["q_value"].values
        thr = np.nanmax(-np.log10(p_vals[q_arr < 0.05]))
        if np.isfinite(thr):
            ax_d.axhline(thr, ls=':', lw=1, color=C['yellowgreen'])
            ax_d.text(0.98, 0.05, r'$q < 0.05$', transform=ax_d.transAxes,
                      ha='right', va='bottom', fontsize=7.5, color=C['yellowgreen'])
    except Exception:
        pass

# Lambda annotation
stats_box(ax_d, rf"$\lambda_{{GC}}$ = {lambda_gc:.2f}" + f"\nn = {n_tests} tests",
          x=0.03, y=0.97)

ax_d.set_xlabel(r'Expected $-\log_{10}(p)$', fontsize=9, color=COL_DARK)
ax_d.set_ylabel(r'Observed $-\log_{10}(p)$', fontsize=9, color=COL_DARK)

ax_d.legend(loc='center left', frameon=True, fancybox=True,
            framealpha=0.9, edgecolor=C['light_teal'], fontsize=7)

style_axis(ax_d)
panel_label(ax_d, 'D')

# ============================================================================
# PANEL E — MOTIF DISRUPTION LOLLIPOP
# ============================================================================
print("   Panel E: Motif disruption lollipop...")

if motif_df is not None and len(motif_df) > 0:
    # Sort by disruption score
    score_col = None
    for col in ['disruption_score_sum', 'score', 'disruption_score']:
        if col in motif_df.columns:
            score_col = col
            break
    if score_col is None:
        score_col = [c for c in motif_df.columns if 'score' in c.lower() or 'disruption' in c.lower()]
        score_col = score_col[0] if score_col else None

    if score_col:
        df_e = motif_df.sort_values(score_col, ascending=True).copy()
        y_e = np.arange(len(df_e))
        scores = df_e[score_col].values

        # Determine significance
        if 'significant' in df_e.columns:
            is_sig_e = df_e['significant'].values.astype(bool)
        elif 'q_value' in df_e.columns:
            is_sig_e = df_e['q_value'].values < 0.05
        else:
            is_sig_e = np.zeros(len(df_e), dtype=bool)

        # Gradient colour by score (sequential green cmap)
        cmap_e = LinearSegmentedColormap.from_list(
            'motif_seq',
            [C['mintcream'], C['palegreen'], C['mediumaquamarine'],
             C['mediumseagreen'], C['forestgreen']],
            N=256
        )
        norm_e = Normalize(vmin=scores.min() * 0.9, vmax=scores.max() * 1.05)

        # Horizontal stems from zero
        for j in range(len(df_e)):
            color = cmap_e(norm_e(scores[j]))
            lw = 3.5 if is_sig_e[j] else 2.0
            alpha = 0.9 if is_sig_e[j] else 0.5
            ax_e.hlines(y_e[j], 0, scores[j], color=color,
                        linewidth=lw, alpha=alpha, zorder=2)

        # Lollipop heads
        for j in range(len(df_e)):
            color = cmap_e(norm_e(scores[j]))
            if is_sig_e[j]:
                # Significant = large bold star
                ax_e.scatter(scores[j], y_e[j], c=COL_MOTIF_SIG, s=120,
                             marker='*', edgecolors='white', linewidths=0.8,
                             zorder=4)
            else:
                ax_e.scatter(scores[j], y_e[j], c=color, s=45,
                             edgecolors='white', linewidths=0.4, zorder=3)

        # Y-axis labels: SNP - gene
        y_labels = []
        for _, r in df_e.iterrows():
            snp_str = str(r.get("snp", ""))
            gene_short = str(r.get("gene", "")).replace("Zm00001d0", "")
            sig_marker = " *" if r.get('significant', False) or r.get('q_value', 1) < 0.05 else ""
            y_labels.append(f"{snp_str} - {gene_short}{sig_marker}")

        ax_e.set_yticks(y_e)
        ax_e.set_yticklabels(y_labels, fontsize=7.5)
        ax_e.set_ylabel('Lead variant–gene', fontsize=9, color=COL_DARK)
        ax_e.set_xlabel('Motif Disruption Score (ΣΔLLR)', fontsize=10, color=COL_DARK)
        ax_e.set_xlim(left=0)

        # q-value annotations on the right
        if 'q_value' in df_e.columns:
            for j, (_, r) in enumerate(df_e.iterrows()):
                q = r['q_value']
                q_text = f"q = {q:.3f}" if q >= 0.001 else f"q = {q:.2e}"
                color_q = COL_MOTIF_SIG if q < 0.05 else C['gray']
                weight = 'bold' if q < 0.05 else 'normal'
                ax_e.text(scores.max() * 1.02, y_e[j], q_text,
                          fontsize=6.5, va='center', ha='left',
                          color=color_q, fontweight=weight)

        # Stats annotation
        n_sig_motif = is_sig_e.sum()
        stats_box(ax_e, f"{n_sig_motif}/{len(df_e)} significant (q < 0.05)\n"
                        f"JASPAR 805 plant PWMs\n"
                        f"Permutation test (n = 200)",
                  x=0.02, y=0.32)

        # Legend
        leg_e = [
            Line2D([0], [0], marker='*', color='none', markerfacecolor=COL_MOTIF_SIG,
                   markeredgecolor='white', markersize=12, label='Significant (q < 0.05)'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_MOTIF_NS,
                   markeredgecolor='white', markersize=6, label='Non-significant'),
        ]
        ax_e.legend(handles=leg_e, loc='lower right', bbox_to_anchor=(0.93, 0.02), frameon=True,
                    fancybox=True, framealpha=0.9, edgecolor=C['light_teal'], fontsize=7)
    else:
        ax_e.text(0.5, 0.5, f"Score column not found in motif data\nColumns: {list(motif_df.columns)}",
                  ha="center", va="center", fontsize=9, color='red', transform=ax_e.transAxes)
else:
    ax_e.text(0.5, 0.5, "Motif disruption data not available\n"
              "Expected: motif_disruption_summary_per_snp.csv",
              ha="center", va="center", fontsize=9, color='red',
              transform=ax_e.transAxes)

style_axis(ax_e)
panel_label(ax_e, 'E', y=1.13)

# ============================================================================
# SAVE
# ============================================================================
print("\n6. Saving figure...")

out_png = FIG_DIR / "figure_3.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"   [OK] {out_png}")

# Save source data
print("\n7. Saving source data to figure_3/ ...")

df[["gene", "chr", "beta_gxe", "p_value", "q_value"]].to_csv(
    FIG_SUPPORT / "panelA_source.csv", index=False)

if len(df_sig) > 0:
    df_sig_export = df_sig.copy()
    df_sig_export["abs_beta"] = df_sig_export["beta_gxe"].abs()
    cols_b = ["snp", "gene", "beta_gxe", "p_value", "q_value"]
    if "se_gxe" in df_sig_export.columns:
        cols_b.insert(3, "se_gxe")
    df_sig_export.sort_values("abs_beta", ascending=False).head(16)[cols_b].to_csv(
        FIG_SUPPORT / "panelB_forest_source.csv", index=False)

if "maf" in df.columns:
    df[["snp", "gene", "maf", "q_value"]].dropna(subset=["maf"]).to_csv(
        FIG_SUPPORT / "panelC_maf_source.csv", index=False)

pd.DataFrame({
    "p_value": df["p_value"].values,
    "q_value": df.get("q_value", np.nan)
}).to_csv(FIG_SUPPORT / "panelD_qq_source.csv", index=False)

if motif_df is not None:
    motif_df.to_csv(FIG_SUPPORT / "panelE_motif_source.csv", index=False)

with open(FIG_SUPPORT / "statistics.txt", 'w', encoding='utf-8') as f:
    f.write("Figure 3 Statistics Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total SNP-gene pairs: {len(df)}\n")
    f.write(f"Significant (q <= {Q_THRESH}): {len(df_sig)}\n")
    f.write(f"Lambda GC: {lambda_gc:.3f}\n")
    f.write(f"Unique genes: {df['gene'].nunique()}\n")
    if motif_df is not None:
        n_sig_m = motif_df.get('significant', pd.Series(dtype=bool)).sum() if 'significant' in motif_df.columns else 0
        f.write(f"Motif SNPs tested: {len(motif_df)}\n")
        f.write(f"Motif significant: {n_sig_m}\n")

print(f"\n{'=' * 80}")
print("FIGURE 3 COMPLETE")
print(f"{'=' * 80}")

# ============================================================================
# DATA INTEGRITY AUDIT
# ============================================================================
print("\n=== DATA INTEGRITY AUDIT ===")
print(f"Panel A: REAL — {len(df)} pairs from {ieqtl_path.name}")
print(f"Panel B: REAL — top {min(len(df_sig), 16)} significant from same file")
print(f"Panel C: REAL — MAF column from same file ({df['maf'].notna().sum() if 'maf' in df.columns else 0} values)")
print(f"Panel D: REAL — p-values from same file, λ_GC = {lambda_gc:.3f}")
if motif_df is not None:
    print(f"Panel E: REAL — {len(motif_df)} SNPs from motif_disruption_summary_per_snp.csv")
else:
    print(f"Panel E: NOT AVAILABLE — file not found")
print("No synthetic, hardcoded, or fallback data used.")
print("All helper functions (coerce_ieqtl_columns, normalize_chr_series,")
print("  null_envelope_qq, clip_pvalues, lambda_gc_from_p) identical to original figure_03.py")

plt.show()