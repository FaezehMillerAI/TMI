import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, default="templates/results.csv")
    p.add_argument("--out_dir", type=str, default="plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if df.empty:
        print("results.csv is empty")
        return

    sns.set_theme(style="whitegrid")

    metric_cols = ["bleu1", "bleu4", "rougeL", "cider", "fidelity", "usefulness", "consistency"]
    for m in metric_cols:
        if m not in df.columns:
            continue
        plt.figure(figsize=(8, 4))
        order = (
            df.groupby("method_variant", dropna=False)[m]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        ax = sns.barplot(data=df, x="method_variant", y=m, order=order, estimator="mean", errorbar="sd")
        ax.set_title(f"{m} by method variant")
        ax.set_xlabel("method_variant")
        ax.set_ylabel(m)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out_path = out_dir / f"{m}_by_method.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

    # Seed stability plot for full model
    full_df = df[df["method_variant"] == "full_rjp"]
    if not full_df.empty and "seed" in full_df.columns and "bleu1" in full_df.columns:
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=full_df.sort_values("seed"), x="seed", y="bleu1", marker="o")
        plt.title("Seed stability (BLEU-1)")
        plt.tight_layout()
        plt.savefig(out_dir / "seed_stability_bleu1.png", dpi=200)
        plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
