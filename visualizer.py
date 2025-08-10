# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
import io
import traceback
from PIL import Image

MAX_FILE_SIZE = 100_000  # 100 KB

def plot_regression(df, x_col, y_col, title=""):
    """
    Create scatterplot with dotted red regression line and compress PNG to < 100 KB.
    Returns: data:image/png;base64,<...>
    """
    try:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Missing required columns: {x_col}, {y_col}")

        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.dropna(subset=[x_col, y_col])
        if df.empty:
            raise ValueError("No valid data to plot after cleaning.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False,
                    color="red", line_kws={'linestyle': 'dotted'})

        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.title(title or f"{x_col} vs {y_col}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        image = Image.open(buf).convert("RGBA")

        # Try optimization loop (PNG compression / resize fallback)
        compressed = io.BytesIO()
        image.save(compressed, format="PNG", optimize=True)
        size = compressed.tell()

        # If still too large, progressively reduce dimensions
        width, height = image.size
        while size > MAX_FILE_SIZE and (width > 200 and height > 200):
            width = int(width * 0.9)
            height = int(height * 0.9)
            resized = image.resize((width, height), Image.LANCZOS)
            compressed = io.BytesIO()
            resized.save(compressed, format="PNG", optimize=True)
            size = compressed.tell()
            image = resized

        compressed.seek(0)
        encoded = base64.b64encode(compressed.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        traceback.print_exc()
        return f"Error in plot_regression: {e}"
