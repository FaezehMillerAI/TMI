import argparse
from pathlib import Path

import pandas as pd


def fmt_mean_std(series):
    s = series.dropna().astype(float)
    if len(s) == 0:
        return "-"
    std = s.std(ddof=1) if len(s) > 1 else 0.0
    return f"{s.mean():.4f} $\\pm$ {std:.4f}"


def summary_table(df, variants, metrics):
    rows = []
    for v in variants:
        d = df[df["method_variant"] == v]
        row = {"Method": v}
        for m in metrics:
            row[m] = fmt_mean_std(d[m]) if m in d.columns else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df, caption, label):
    cols = df.columns.tolist()
    align = "l" + "c" * (len(cols) - 1)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]

    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, default="templates/results.csv")
    p.add_argument("--out_dir", type=str, default="paper/tables")
    p.add_argument("--main_variants", nargs="+", default=["full_rjp", "no_ram"])
    p.add_argument(
        "--ablation_variants",
        nargs="+",
        default=["full_rjp", "no_ram", "no_counterfactual", "no_anatomy", "fixed_fusion"],
    )
    p.add_argument(
        "--main_metrics",
        nargs="+",
        default=["bleu1", "bleu4", "rougeL", "cider", "consistency", "fidelity", "usefulness"],
    )
    p.add_argument(
        "--ablation_metrics",
        nargs="+",
        default=["bleu1", "bleu4", "rougeL", "cider", "fidelity", "usefulness", "consistency"],
    )
    args = p.parse_args()

    df = pd.read_csv(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = summary_table(df, args.main_variants, args.main_metrics)
    abl_df = summary_table(df, args.ablation_variants, args.ablation_metrics)

    main_tex = to_latex(main_df, "Main performance comparison on IU X-Ray.", "tab:main_results")
    abl_tex = to_latex(abl_df, "Ablation study of RJP-RAM components.", "tab:ablation_results")

    (out_dir / "main_results.tex").write_text(main_tex)
    (out_dir / "ablation_results.tex").write_text(abl_tex)
    main_df.to_csv(out_dir / "main_results_summary.csv", index=False)
    abl_df.to_csv(out_dir / "ablation_results_summary.csv", index=False)

    print(f"Saved: {out_dir / 'main_results.tex'}")
    print(f"Saved: {out_dir / 'ablation_results.tex'}")


if __name__ == "__main__":
    main()
