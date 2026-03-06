import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def paired_by_seed(df, a, b, metric):
    da = df[df["method_variant"] == a][["seed", metric]].dropna()
    db = df[df["method_variant"] == b][["seed", metric]].dropna()
    m = da.merge(db, on="seed", suffixes=("_a", "_b"))
    return m


def paired_bootstrap_ci(diffs, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return None, None
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(diffs[idx])))
    lo = np.percentile(samples, (100 - ci) / 2)
    hi = np.percentile(samples, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, default="templates/results.csv")
    p.add_argument("--baseline", type=str, default="full_rjp")
    p.add_argument("--compare", type=str, nargs="*", default=[])
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "cider", "fidelity", "usefulness", "consistency"],
    )
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out_json", type=str, default="analysis/significance.json")
    p.add_argument("--out_csv", type=str, default="analysis/significance.csv")
    args = p.parse_args()

    df = pd.read_csv(args.results_csv)
    variants = sorted(df["method_variant"].dropna().unique().tolist())
    compares = args.compare if args.compare else [v for v in variants if v != args.baseline]

    rows = []
    for cmp_variant in compares:
        for metric in args.metrics:
            if metric not in df.columns:
                continue
            m = paired_by_seed(df, args.baseline, cmp_variant, metric)
            if m.empty:
                rows.append(
                    {
                        "baseline": args.baseline,
                        "compare": cmp_variant,
                        "metric": metric,
                        "n_pairs": 0,
                        "mean_diff_compare_minus_baseline": None,
                        "paired_t_pvalue": None,
                        "bootstrap_ci_low": None,
                        "bootstrap_ci_high": None,
                        "significant_0_05": None,
                    }
                )
                continue

            diffs = (m[f"{metric}_b"] - m[f"{metric}_a"]).to_numpy(dtype=float)
            mean_diff = float(np.mean(diffs))
            t_res = ttest_rel(m[f"{metric}_b"], m[f"{metric}_a"], nan_policy="omit")
            lo, hi = paired_bootstrap_ci(diffs, n_boot=args.n_boot)
            rows.append(
                {
                    "baseline": args.baseline,
                    "compare": cmp_variant,
                    "metric": metric,
                    "n_pairs": int(len(diffs)),
                    "mean_diff_compare_minus_baseline": mean_diff,
                    "paired_t_pvalue": float(t_res.pvalue) if t_res.pvalue == t_res.pvalue else None,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "significant_0_05": bool(t_res.pvalue < 0.05) if t_res.pvalue == t_res.pvalue else None,
                }
            )

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved significance results to: {out_json}")
    print(f"Saved significance table to: {out_csv}")


if __name__ == "__main__":
    main()
