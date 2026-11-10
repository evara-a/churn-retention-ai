"""Exploratory data analysis: prints a summary and saves plots to outputs/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .preprocess import TARGET


def run_eda(df: pd.DataFrame, out_dir: str = "outputs") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("\n=== EDA ===")
    print(f"Shape: {df.shape}")
    print("\nMissing values per column (after coercing blanks):")
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce") if "TotalCharges" in df else None
    miss = df.isna().sum()
    if tc is not None:
        miss["TotalCharges"] = tc.isna().sum()
    print(miss[miss > 0].to_string() or "none")
    print(f"\nChurn rate: {(df[TARGET] == 'Yes').mean():.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    sns.countplot(data=df, x=TARGET, ax=axes[0, 0], palette="Set2", hue=TARGET, legend=False)
    axes[0, 0].set_title("Churn distribution")

    sns.histplot(data=df, x="tenure", hue=TARGET, multiple="stack", bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Tenure vs churn")

    order = df["Contract"].dropna().unique()
    rate = df.dropna(subset=["Contract"]).groupby("Contract")[TARGET].apply(lambda s: (s == "Yes").mean())
    rate.loc[order].plot(kind="bar", ax=axes[1, 0], color="#e07a5f", rot=0)
    axes[1, 0].set_title("Churn rate by contract type")
    axes[1, 0].set_ylabel("Churn rate")

    df2 = df.copy()
    df2["MonthlyCharges"] = pd.to_numeric(df2["MonthlyCharges"], errors="coerce")
    sns.boxplot(data=df2, x=TARGET, y="MonthlyCharges", ax=axes[1, 1], palette="Set2", hue=TARGET, legend=False)
    axes[1, 1].set_title("Monthly charges vs churn")

    plt.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=130)
    plt.close(fig)
    print(f"Saved {out / 'eda_overview.png'}")
