# Paper 1 Codebase (IEEE-TMI Style): RJP-RAM + Beyond-Heatmap Explainability

Standalone modular project for your first paper.

## Methodology Modules

- `rjp_method/data.py`: IU-Xray loading + reproducible splits
- `rjp_method/retrieval.py`: factual / counterfactual / anatomy-aware retrieval
- `rjp_method/model.py`: adaptive fusion decoder + RAM
- `rjp_method/train.py`: training, evaluation, qualitative generation
- `rjp_method/evaluate.py`: BLEU-1/2/3/4, ROUGE-L, CIDEr (official if available, fallback proxy), consistency/fidelity/usefulness
- `rjp_method/explainability.py`: beyond-heatmap explainability

## Novel Explainability (Beyond Heatmaps)

This project generates claim-level evidence reasoning artifacts:
- Claim decomposition from generated report
- Claim-to-evidence support scoring (factual, counterfactual, anatomy)
- Differential reasoning margin per claim (support minus counter-support)
- Interactive HTML visualizations:
  - claim-evidence radar plots
  - differential reasoning bar charts

Outputs are saved in:
- `outputs/<run_name>/artifacts/qualitative/explainability_graphs/*.html`
- `outputs/<run_name>/artifacts/qualitative/qualitative_bundle.json`
- `outputs/<run_name>/artifacts/qualitative/qualitative_report.md`

## Install

```bash
cd /Users/fs525/Desktop/ACL/paper1_rjp_ram
pip install -r requirements.txt
```

## Train (single run)

```bash
python run_paper1.py \
  --data_root /root/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --run_name full_seed42 \
  --epochs 8 \
  --batch_size 16 \
  --top_k_retrieval 5 \
  --ram_alpha 0.15 \
  --seed 42
```

## Recompute test metrics from trained checkpoint (no retraining)

```bash
python evaluate_checkpoint.py \
  --checkpoint outputs/full_seed42/checkpoints/best_model.pt \
  --data_root /root/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --run_name full_seed42_recomputed
```

This computes BLEU-1/2/3/4, ROUGE-L, CIDEr, explainability proxies, and regenerated qualitative bundles.

## Ablation Runs (module-level)

- No RAM: `--disable_ram`
- No counterfactual retrieval: `--disable_counterfactual`
- No anatomy-aware retrieval: `--disable_anatomy`
- Fixed fusion (no adaptive gate): `--disable_adaptive_fusion --fixed_fusion_alpha 0.5`

## Batch experiments for paper tables

```bash
python scripts/launch_experiments_paper1.py \
  --data_root /root/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --seeds 42 123 999 \
  --epochs 8
```

## Comprehensive Evaluation Pipeline

1. Aggregate all runs:
```bash
python scripts/collect_results.py --outputs_root outputs --results_csv templates/results.csv
```

2. Significance tests:
```bash
python scripts/significance_tests.py --results_csv templates/results.csv --baseline full_rjp
```

3. Plot quantitative metrics:
```bash
python scripts/plot_results.py --results_csv templates/results.csv --out_dir plots
```

4. Generate LaTeX tables:
```bash
python scripts/generate_latex_tables.py --results_csv templates/results.csv --out_dir paper/tables
```

## Qualitative Results

After training/recompute, inspect:
- `outputs/<run_name>/artifacts/samples*.json`
- `outputs/<run_name>/artifacts/qualitative*/qualitative_report.md`
- `outputs/<run_name>/artifacts/qualitative*/explainability_graphs/*.html`

## Paper Skeleton

- `paper/paper1_skeleton.tex`
