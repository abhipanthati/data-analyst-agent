import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
import io
import traceback

def plot_regression(df, x_col, y_col, title=""):
    """
    Generate a scatter plot with regression line for x_col vs y_col.
    Returns a base64-encoded PNG image URI. Handles missing columns and empty data.
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
        sns.regplot(
            data=df, x=x_col, y=y_col,
            scatter=False,
            color="red",
            line_kws={'linestyle': 'dotted'}
        )

        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.title(title or f"{x_col} vs {y_col}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error in plot_regression: {e}")
        traceback.print_exc()
        return "Error in plot generation"
