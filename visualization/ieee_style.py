"""
IEEE Publication Formatting Guidelines & Matplotlib Theme.
Standardizes fonts (Times New Roman / Serif), sizing (3.5 in single-column, 7.16 in double-column),
high-resolution rendering (300 DPI), and clean mathematical styling.
"""

import matplotlib.pyplot as plt

def apply_ieee_style():
    """
    Applies IEEE style guidelines for academic journals and conference proceedings:
    - Font: Times New Roman / Serif with Computer Modern math typesetting
    - Sizes: 8-10pt for readability in printed two-column or single-column layout
    - DPI: 300 DPI minimum for clear line art
    - Grid: Subtle, non-intrusive dashed grid
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "axes.titleweight": "normal",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.5,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#CCCCCC",
        "figure.titlesize": 10.5,
        "figure.titleweight": "normal",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": "#D0D0D0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

# Standard IEEE Column Widths in inches
IEEE_SINGLE_COL_WIDTH = 3.5
IEEE_DOUBLE_COL_WIDTH = 7.16
