import argparse

import torch

from rjp_method.config import Config
from rjp_method.data import load_iuxray_records, make_dataloaders
from rjp_method.explainability import generate_qualitative_bundle
from rjp_method.model import RJPModel
from rjp_method.retrieval import RetrievalEngine
from rjp_method.text import Vocab
from rjp_method.train import evaluate_generation
from rjp_method.utils import ensure_dirs, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default="re_eval")
    p.add_argument("--max_samples", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    cfg = Config(
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        max_len=int(ckpt_cfg.get("max_len", 64)),
        image_size=int(ckpt_cfg.get("image_size", 224)),
        top_k_retrieval=int(ckpt_cfg.get("top_k_retrieval", 5)),
        use_ram=bool(ckpt_cfg.get("use_ram", True)),
        use_counterfactual=bool(ckpt_cfg.get("use_counterfactual", True)),
        use_anatomy=bool(ckpt_cfg.get("use_anatomy", True)),
        use_adaptive_fusion=bool(ckpt_cfg.get("use_adaptive_fusion", True)),
        fixed_fusion_alpha=float(ckpt_cfg.get("fixed_fusion_alpha", 0.5)),
        seed=int(ckpt_cfg.get("seed", 42)),
        train_split=float(ckpt_cfg.get("train_split", 0.8)),
        val_split=float(ckpt_cfg.get("val_split", 0.1)),
    )
    ensure_dirs(cfg)

    splits = load_iuxray_records(
        cfg.data_root,
        seed=cfg.seed,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
    )

    vocab = Vocab(min_freq=int(ckpt_cfg.get("min_word_freq", cfg.min_word_freq)))
    vocab.build([r["report"] for r in splits["train"]])

    _, _, test_loader = make_dataloaders(splits, vocab, cfg)

    model = RJPModel(
        vocab_size=len(vocab),
        embed_dim=int(ckpt_cfg.get("embed_dim", cfg.embed_dim)),
        hidden_dim=int(ckpt_cfg.get("hidden_dim", cfg.hidden_dim)),
        pad_idx=vocab.stoi["<pad>"],
        use_ram=cfg.use_ram,
        use_adaptive_fusion=cfg.use_adaptive_fusion,
        fixed_fusion_alpha=cfg.fixed_fusion_alpha,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    retrieval_engine = RetrievalEngine(
        train_records=splits["train"],
        top_k=cfg.top_k_retrieval,
        abnormality_keywords=cfg.abnormality_keywords,
        anatomy_keywords=cfg.anatomy_keywords,
        use_counterfactual=cfg.use_counterfactual,
        use_anatomy=cfg.use_anatomy,
    )

    test_metrics, samples = evaluate_generation(
        model,
        test_loader,
        retrieval_engine,
        vocab,
        cfg,
        device,
        max_samples=args.max_samples,
    )

    write_json(cfg.artifact_dir / "test_metrics_recomputed.json", test_metrics)
    write_json(cfg.artifact_dir / "samples_recomputed.json", samples)
    qual = generate_qualitative_bundle(samples, out_dir=cfg.artifact_dir / "qualitative_recomputed", top_k=cfg.qualitative_top_k)
    write_json(cfg.artifact_dir / "qualitative_recomputed" / "qualitative_summary.json", {"good": len(qual["good_cases"]), "bad": len(qual["bad_cases"])})

    print("Checkpoint evaluation complete.")
    print(f"Saved metrics: {cfg.artifact_dir / 'test_metrics_recomputed.json'}")
    print(f"Saved samples: {cfg.artifact_dir / 'samples_recomputed.json'}")
    print(f"Saved qualitative: {cfg.artifact_dir / 'qualitative_recomputed'}")
    print("Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
