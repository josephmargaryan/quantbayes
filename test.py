# Publication-ready ELBO plots with LOG-SCALE on y-axis.
# Uses col0 as step (x) and col1 as ELBO (y), drops NaNs and nonpositive y for log scale.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator

# ----- Matplotlib config -----
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,  # Embed TrueType fonts
        "ps.fonttype": 42,
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

BASE_DIR = Path("/Users/josephmargaryan/Desktop/AI_STATS")


def load_xy_two_col_csv(path: Path):
    """
    Reads a 2-column CSV: col0=step, col1=ELBO.
    - Auto-detect delimiter
    - Coerce to numeric
    - Drop rows with NaNs in either column
    - Sort by step (x)
    Returns: x (np.ndarray), y (np.ndarray)
    """
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    return x[order], y[order]


def plot_group(curves, outfile_stub: str):
    """
    curves: list of (label, filename) tuples
    outfile_stub: base filename for saved figures (without extension)
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Track positive y-range across series for tidy limits
    global_min_pos, global_max = np.inf, 0.0

    for label, fname in curves:
        x, y = load_xy_two_col_csv(BASE_DIR / fname)

        # For log-scale: drop nonpositive values (if any)
        mpos = y > 0
        x, y = x[mpos], y[mpos]

        if y.size == 0:
            raise ValueError(
                f"All nonpositive values in {fname}; cannot plot on log scale."
            )

        global_min_pos = min(global_min_pos, np.min(y))
        global_max = max(global_max, np.max(y))

        ax.plot(x, y, label=label, linewidth=2.2)

    ax.set_xlabel("Training step")
    ax.set_ylabel("ELBO (log scale)")
    ax.set_yscale("log")

    # Nice limits (slight padding)
    ax.set_ylim(bottom=global_min_pos * 0.9, top=global_max * 1.05)

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    ax.margins(x=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()

    fig.savefig(BASE_DIR / f"{outfile_stub}.pdf", bbox_inches="tight")
    fig.savefig(BASE_DIR / f"{outfile_stub}.png", bbox_inches="tight")
    plt.show()


# ----- Figure 1: BCCB vs SpectralBCCB vs Conv2d -----
group1 = [
    ("BCCB", "Circulant2D_loss.csv"),
    ("SpectralBCCB", "RFFT2D_loss.csv"),
    ("Conv2d", "conv_loss.csv"),
]
plot_group(group1, "elbo_bccb_spectralbccb_conv2d_logy")

# ----- Figure 2: Dense vs Circulant vs SpectralCirculant -----
group2 = [
    ("Dense", "dense_loss.csv"),
    ("Circulant", "Circulant1D_loss.csv"),
    ("SpectralCirculant", "RFFT1D_loss.csv"),
]
plot_group(group2, "elbo_dense_circulant_spectralcirculant_logy")
