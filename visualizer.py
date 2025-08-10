# visualizer.py
import io
import base64
import os
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

sns.set_theme(style="whitegrid")
sns.set_context("notebook")


def _encode_png_bytes_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _try_palette_quantize(png_bytes: bytes) -> bytes:
    """Try to convert PNG to paletted (P) mode to save size."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        # Convert to adaptive palette - this often reduces PNG size significantly
        pal = img.convert("P", palette=Image.ADAPTIVE)
        out = io.BytesIO()
        pal.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes


def _render_png_bytes(fig, dpi: int = 80) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", optimize=True)
    return buf.getvalue()


def plot_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Regression",
    max_bytes: int = 100_000,
    try_reductions: bool = True,
) -> str:
    """
    Create a scatter + dotted red regression line and return a PNG data URI.

    - Ensures axes labelled.
    - If resulting PNG > max_bytes, tries smaller figure/dpi and palette quantize.
    - Returns "data:image/png;base64,...."
    """
    if df is None or x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Required columns missing for plotting")

    # Keep only numeric values for plotting
    tmp = df[[x_col, y_col]].dropna()
    try:
        tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
        tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    except Exception:
        pass
    tmp = tmp.dropna()
    if tmp.empty:
        raise ValueError("No numeric data available for plotting")

    # Try a sequence of render options to stay under size limit
    fig_sizes = [(6, 4), (5, 3.2), (4.5, 3), (4, 2.8)]
    dpis = [80, 72, 60]

    last_png = None
    for figsize in fig_sizes:
        for dpi in dpis:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            sns.regplot(
                x=x_col,
                y=y_col,
                data=tmp,
                scatter_kws={"alpha": 0.6, "s": 20},
                line_kws={"color": "red", "linestyle": "--", "linewidth": 1.2},
                ax=ax,
            )
            ax.set_xlabel(str(x_col).capitalize())
            ax.set_ylabel(str(y_col).capitalize())
            ax.set_title(title)
            # Draw grid lightly
            ax.grid(True, linewidth=0.5, alpha=0.4)

            png_bytes = _render_png_bytes(fig, dpi=dpi)
            plt.close(fig)

            # quick size check
            if len(png_bytes) <= max_bytes:
                return _encode_png_bytes_to_data_uri(png_bytes)

            last_png = png_bytes

    # If still too large and try_reductions, try palette quantize
    if try_reductions and last_png:
        small = _try_palette_quantize(last_png)
        if len(small) <= max_bytes:
            return _encode_png_bytes_to_data_uri(small)

    # Final fallback: if still too large, return a compact plot by re-rendering smaller
    # We'll render a minimal tiny figure (very small)
    try:
        fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=60)
        sns.regplot(
            x=x_col,
            y=y_col,
            data=tmp,
            scatter_kws={"alpha": 0.5, "s": 10},
            line_kws={"color": "red", "linestyle": "--", "linewidth": 1.0},
            ax=ax,
        )
        ax.set_xlabel(str(x_col).capitalize(), fontsize=8)
        ax.set_ylabel(str(y_col).capitalize(), fontsize=8)
        ax.set_title(title, fontsize=9)
        png_bytes = _render_png_bytes(fig, dpi=60)
        plt.close(fig)
        if len(png_bytes) <= max_bytes:
            return _encode_png_bytes_to_data_uri(png_bytes)

        # Attempt palette quantize on this tiny version
        pq = _try_palette_quantize(png_bytes)
        if len(pq) <= max_bytes:
            return _encode_png_bytes_to_data_uri(pq)
    except Exception:
        pass

    # At this point: cannot make it <= max_bytes, return best-effort (but still a data URI)
    if last_png:
        return _encode_png_bytes_to_data_uri(last_png)

    raise RuntimeError("Failed to generate plot image")
