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


import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_regression(df: pd.DataFrame, x_col: str, y_col: str, title: str = "") -> str:
    """
    Plot a scatterplot with a dotted red regression line and return as base64 PNG.
    """
    # Validate columns
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Missing required columns: {x_col}, {y_col}")

    # Ensure numeric
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        raise ValueError("No valid numeric data to plot.")

    # Create plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    sns.regplot(
        x=x_col, y=y_col, data=df,
        scatter=False,
        line_kws={"color": "red", "linestyle": ":"}  # dotted red line
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"{y_col} vs {x_col}")

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()

    # Convert to base64
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"

    # Ensure < 100 KB
    if len(base64.b64decode(img_base64)) >= 100_000:
        raise ValueError("Generated image exceeds 100 KB limit.")

    return data_uri
