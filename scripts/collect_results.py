import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def infer_variant(cfg):
    if not cfg.get("use_ram", True):
        return "no_ram"
    if not cfg.get("use_counterfactual", True):
        return "no_counterfactual"
    if not cfg.get("use_anatomy", True):
        return "no_anatomy"
    if not cfg.get("use_adaptive_fusion", True):
        return "fixed_fusion"
    return "full_rjp"


def collect_run(run_dir: Path):
    artifacts = run_dir / "artifacts"
    cfg = _read_json(artifacts / "config.json", {})
    metrics = _read_json(artifacts / "test_metrics.json", {})
    summary = _read_json(artifacts / "run_summary.json", {})

    return {
        "run_name": run_dir.name,
        "method_variant": infer_variant(cfg),
        "seed": cfg.get("seed"),
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "top_k_retrieval": cfg.get("top_k_retrieval"),
        "use_ram": int(bool(cfg.get("use_ram", True))),
        "use_counterfactual": int(bool(cfg.get("use_counterfactual", True))),
        "use_anatomy": int(bool(cfg.get("use_anatomy", True))),
        "use_adaptive_fusion": int(bool(cfg.get("use_adaptive_fusion", True))),
        "fixed_fusion_alpha": cfg.get("fixed_fusion_alpha"),
        "ram_alpha": cfg.get("ram_alpha"),
        "data_root": cfg.get("data_root"),
        "n_train": summary.get("num_train_records"),
        "n_val": summary.get("num_val_records"),
        "n_test": summary.get("num_test_records"),
        "best_val_loss": summary.get("best_val_loss"),
        "bleu1": metrics.get("bleu1"),
        "bleu2": metrics.get("bleu2"),
        "bleu3": metrics.get("bleu3"),
        "bleu4": metrics.get("bleu4"),
        "rougeL": metrics.get("rougeL"),
        "cider": metrics.get("cider"),
        "consistency": metrics.get("consistency"),
        "plausibility": metrics.get("plausibility"),
        "fidelity": metrics.get("fidelity"),
        "usefulness": metrics.get("usefulness"),
        "notes": "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_root", type=str, default="outputs")
    p.add_argument("--results_csv", type=str, default="templates/results.csv")
    p.add_argument("--append", action="store_true")
    args = p.parse_args()

    outputs_root = Path(args.outputs_root)
    runs = sorted([d for d in outputs_root.iterdir() if d.is_dir()]) if outputs_root.exists() else []
    rows = [collect_run(r) for r in runs if (r / "artifacts" / "test_metrics.json").exists()]

    if not rows:
        print("No completed runs found.")
        return

    fieldnames = list(rows[0].keys())
    mode = "a" if args.append and Path(args.results_csv).exists() else "w"
    with open(args.results_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
