import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

sns.set_theme(style="whitegrid")
sns.set_context("notebook")


def _try_palette_quantize(png_bytes: bytes) -> bytes:
    """Try to convert PNG to paletted (P) mode to save size."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        pal = img.convert("P", palette=Image.ADAPTIVE)
        out = io.BytesIO()
        pal.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes


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

    # Quantize image to shrink size
    png_bytes = _try_palette_quantize(buf.getvalue())

    # Convert to base64
    img_base64 = base64.b64encode(png_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"

    # Ensure < 100 KB
    if len(png_bytes) >= 100_000:
        raise ValueError("Generated image exceeds 100 KB limit.")

    return data_uri
