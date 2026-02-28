"""
Color Configuration for GWAS Manuscript Figures
===============================================
Centralised colour definitions and semantic mappings used across all
main figures (1-4) and supplementary figures (S1-S6).

Palette hierarchy:
  ~50-60%  Mid-tone greens (mediumseagreen, limegreen, cadetblue, mediumturquoise)
  ~20%     Light yellow-green accents (springgreen, palegreen, yellowgreen)
  ~10-20%  Light blue-green accents (darkturquoise, deepskyblue)
  ~10%     Dark accents (seagreen, teal, dark_forest, forestgreen)

Design system (axes, text, grid) uses green-family tones throughout.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# DESIGN SYSTEM — UI chrome colours (text, spines, grids, borders)
# ============================================================================
TEXT_PRIMARY   = "#1b4332"   # dark forest green — titles, labels, panel letters
TEXT_SECONDARY = "#2E8B57"   # sea green — secondary annotations
TEXT_TERTIARY  = "#5F9EA0"   # cadet blue — muted annotations
SPINE_COLOR    = "#008080"   # teal — axis spines and tick marks
GRID_COLOR     = "#B2DFDB"   # light teal — gridlines, legend/box edges
GREY_NEUTRAL   = "#B8B8B8"   # warm neutral grey — non-significant elements
GREY_LIGHT     = "#D3D3D3"   # light grey — backgrounds
GREY_PALE      = "#F0F0F0"   # pale grey — subtle fills


class ColorConfig:
    def __init__(self):
        # ================================================================
        # Design system references (accessible via colors.xxx)
        # ================================================================
        self.text_primary   = TEXT_PRIMARY
        self.text_secondary = TEXT_SECONDARY
        self.text_tertiary  = TEXT_TERTIARY
        self.spine_color    = SPINE_COLOR
        self.grid_color     = GRID_COLOR
        self.grey_neutral   = GREY_NEUTRAL

        # ================================================================
        # Core named colours
        # ================================================================
        self.forestgreen = '#228B22'
        self.limegreen = '#32CD32'
        self.green = '#008000'
        self.lime = '#00FF00'
        self.gainsboro = '#DCDCDC'
        self.mediumseagreen = '#3CB371'
        self.springgreen = '#00FF7F'
        self.mintcream = '#F5FFFA'
        self.mediumspringgreen = '#00FA9A'
        self.mediumaquamarine = '#66CDAA'
        self.aquamarine = '#7FFFD4'
        self.turquoise = '#40E0D0'
        self.mediumturquoise = '#48D1CC'
        self.azure = '#F0FFFF'
        self.lightcyan = '#E0FFFF'
        self.paleturquoise = '#AFEEEE'
        self.darkslategray = TEXT_PRIMARY
        self.teal = '#008080'
        self.darkcyan = '#008B8B'
        self.yellowgreen = '#9ACD32'
        self.cyan = '#00FFFF'
        self.darkturquoise = '#00CED1'
        self.cadetblue = '#5F9EA0'
        self.powderblue = '#B0E0E6'
        self.lightblue = '#ADD8E6'
        self.deepskyblue = '#00BFFF'
        self.skyblue = '#87CEEB'
        self.lightskyblue = '#87CEFA'
        self.steelblue = '#4682B4'
        self.aliceblue = '#F0F8FF'
        self.dodgerblue = '#1E90FF'
        self.lightslategray = '#778899'
        self.gray = GREY_NEUTRAL
        self.darkgray = '#A9A9A9'
        self.lightgray = GREY_LIGHT
        self.slategray = '#708090'
        self.darkseagreen = '#8FBC8F'
        self.seagreen = '#2E8B57'
        self.olivedrab = '#6B8E23'
        self.palegreen = '#98FB98'
        self.nature_green = '#00A087'
        self.royalblue = '#4169E1'

        # ================================================================
        # Semantic colour assignments for MAIN FIGURES
        # ================================================================
        # Figure 1 + 4: gene classes & stability states
        self.classifications = {
            "platinum": self.limegreen,          # bright mid-tone green
            "modulator": self.mediumseagreen,    # mid-tone green
            "driver": self.darkturquoise,        # blue-green accent
            "stable": self.darkseagreen,         # muted green
            "unstable": self.deepskyblue         # light blue-green
        }

        # Figure 3: observed/expected + envelopes
        self.statistics = {
            "significant": self.limegreen,
            "borderline": self.mediumturquoise,
            "nonsignificant": GREY_NEUTRAL,
            "observed": self.seagreen,
            "expected": self.darkseagreen,
            "ci_band": self.palegreen
        }

        # Figure 2: consequence colours — maximise contrast between types
        self.consequences = {
            "5_prime_UTR": self.darkturquoise,       # vivid blue-green
            "3_prime_UTR": self.yellowgreen,          # yellow-green (distinct)
            "upstream": self.mediumseagreen,           # mid-green
            "downstream": self.cadetblue,              # muted teal
            "exon": self.forestgreen,                  # deep green
            "intron": self.mediumaquamarine,            # light green
            "intergenic": self.steelblue               # blue accent
        }

        # S5: genotype palette (green-teal family)
        self.genotypes = {
            0: self.mediumturquoise,  # hom-ref
            1: self.palegreen,        # het
            2: self.seagreen          # hom-alt
        }

        # ================================================================
        # Colormaps for heatmaps/tiles
        # ================================================================
        # Sequential: mint → teal → deep green
        self.sequential_cmap = LinearSegmentedColormap.from_list(
            "gwas_seq",
            [
                HEX_CODES['mintcream'],
                HEX_CODES['turquoise'],
                HEX_CODES['mediumseagreen'],
                HEX_CODES['seagreen'],
            ]
        )
        # Diverging: steelblue → light teal → aquamarine → mediumseagreen → forestgreen
        self.diverging_cmap = LinearSegmentedColormap.from_list(
            "gwas_div",
            [
                HEX_CODES['steelblue'],
                HEX_CODES['light_teal'],
                HEX_CODES['aquamarine'],
                HEX_CODES['mediumseagreen'],
                HEX_CODES['forestgreen'],
            ]
        )
        # Green-forward sequential colormap
        self.sequential_cmap_green = LinearSegmentedColormap.from_list(
            "gwas_seq_green",
            [
                HEX_CODES['mintcream'],
                HEX_CODES['palegreen'],
                HEX_CODES['mediumseagreen'],
                HEX_CODES['seagreen'],
            ]
        )

    # ----------------------- helper utilities -----------------------
    def add_panel_label(self, ax, label, x=-0.12, y=1.08, **kwargs):
        """Add panel letter (A, B, C, etc.)."""
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left",
                color=TEXT_PRIMARY, **kwargs)

    def add_stats_box(self, ax, text, x=0.03, y=0.97):
        """Add a compact stats box (top-left by default)."""
        ax.text(x, y, text, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left",
                color=TEXT_PRIMARY,
                bbox=dict(facecolor="white", alpha=0.92,
                          edgecolor=GRID_COLOR, lw=0.8,
                          boxstyle="round,pad=0.35"))

    def style_axis(self, ax):
        """Standard minimalist axis styling with green-toned spines."""
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            if ax.spines[spine].get_visible():
                ax.spines[spine].set_color(SPINE_COLOR)

    def configure_matplotlib(self):
        """Set global matplotlib parameters matching the design system."""
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
            'axes.edgecolor': SPINE_COLOR,
            'axes.labelcolor': TEXT_PRIMARY,
            'text.color': TEXT_PRIMARY,
            'xtick.labelsize': 9.5,
            'ytick.labelsize': 9.5,
            'xtick.color': SPINE_COLOR,
            'ytick.color': SPINE_COLOR,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'legend.fontsize': 9.5,
            'legend.frameon': True,
            'legend.framealpha': 0.92,
            'legend.edgecolor': GRID_COLOR,
            'legend.fancybox': True,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.15,
            'savefig.facecolor': 'white',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })


# Hex lookup for all named colours used across the project
HEX_CODES = {
    # Dark accents (~10%)
    'dark_forest': '#1b4332',
    'forestgreen': '#228B22',
    'seagreen': '#2E8B57',
    'teal': '#008080',
    'darkcyan': '#008B8B',
    # Mid-tone greens (~50-60%)
    'mediumseagreen': '#3CB371',
    'limegreen': '#32CD32',
    'mediumturquoise': '#48D1CC',
    'cadetblue': '#5F9EA0',
    'mediumaquamarine': '#66CDAA',
    'darkseagreen': '#8FBC8F',
    'turquoise': '#40E0D0',
    'nature_green': '#00A087',
    # Light yellow-green (~20%)
    'yellowgreen': '#9ACD32',
    'springgreen': '#00FF7F',
    'palegreen': '#98FB98',
    'mediumspringgreen': '#00FA9A',
    # Light blue-green (~10-20%)
    'darkturquoise': '#00CED1',
    'deepskyblue': '#00BFFF',
    'paleturquoise': '#AFEEEE',
    # Ultra-light (backgrounds)
    'mintcream': '#F5FFFA',
    'lightcyan': '#E0FFFF',
    'light_teal': '#B2DFDB',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    # Neutrals
    'grey_neutral': '#B8B8B8',
    'grey_light': '#D3D3D3',
    'grey_pale': '#F0F0F0',
    'gainsboro': '#DCDCDC',
    # Blue accents (~10-20%)
    'steelblue': '#4682B4',
    'royalblue': '#4169E1',
    'dodgerblue': '#1E90FF',
    'skyblue': '#87CEEB',
    'lightskyblue': '#87CEFA',
    'powderblue': '#B0E0E6',
    # Legacy (kept for compatibility)
    'green': '#008000',
    'lime': '#00FF00',
    'cyan': '#00FFFF',
    'olivedrab': '#6B8E23',
    'gray': '#B8B8B8',
    'lightgray': '#D3D3D3',
    'slategray': '#708090',
}

# Singleton used by plotting code
colors = ColorConfig()

# Configure MPL defaults on import
colors.configure_matplotlib()

if __name__ == "__main__":
    print("=" * 70)
    print("Design system:")
    print(f"  TEXT_PRIMARY:  {TEXT_PRIMARY}")
    print(f"  SPINE_COLOR:   {SPINE_COLOR}")
    print(f"  GRID_COLOR:    {GRID_COLOR}")
    print("Semantic colour assignments:")
    print("  classifications (Fig 1, 4):", colors.classifications)
    print("  statistics (Fig 3)        :", colors.statistics)
    print("  consequences (Fig 2)      :", colors.consequences)
    print("Colormaps: sequential, diverging, sequential_green")
    print("=" * 70)
