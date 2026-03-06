import argparse

import torch

from rjp_method.config import Config
from rjp_method.data import load_iuxray_records, make_dataloaders
from rjp_method.model import RJPModel
from rjp_method.text import Vocab
from rjp_method.train import run_training
from rjp_method.utils import ensure_dirs, now_tag, set_seed, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/content/iu_xray")
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--download_kagglehub", action="store_true")
    p.add_argument("--kaggle_dataset", type=str, default="raddar/chest-xrays-indiana-university")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--top_k_retrieval", type=int, default=5)
    p.add_argument("--ram_alpha", type=float, default=0.15)
    p.add_argument("--disable_ram", action="store_true")
    p.add_argument("--disable_counterfactual", action="store_true")
    p.add_argument("--disable_anatomy", action="store_true")
    p.add_argument("--disable_adaptive_fusion", action="store_true")
    p.add_argument("--fixed_fusion_alpha", type=float, default=0.5)
    p.add_argument("--disable_explainability_viz", action="store_true")
    p.add_argument("--qualitative_top_k", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name.strip() or f"paper1_rjp_ram_{now_tag()}"

    cfg = Config(
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        image_size=args.image_size,
        top_k_retrieval=args.top_k_retrieval,
        ram_alpha=args.ram_alpha,
        use_ram=not args.disable_ram,
        use_counterfactual=not args.disable_counterfactual,
        use_anatomy=not args.disable_anatomy,
        use_adaptive_fusion=not args.disable_adaptive_fusion,
        fixed_fusion_alpha=args.fixed_fusion_alpha,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        generate_explainability_viz=not args.disable_explainability_viz,
        qualitative_top_k=args.qualitative_top_k,
    )

    set_seed(cfg.seed)
    ensure_dirs(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Run directory: {cfg.run_dir}")

    data_root = cfg.data_root
    if args.download_kagglehub:
        import kagglehub

        data_root = kagglehub.dataset_download(args.kaggle_dataset)
        print(f"Downloaded dataset with kagglehub to: {data_root}")

    splits = load_iuxray_records(
        data_root,
        seed=cfg.seed,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
    )
    print(
        f"Loaded splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    vocab = Vocab(min_freq=cfg.min_word_freq)
    vocab.build([r["report"] for r in splits["train"]])
    print(f"Vocab size: {len(vocab)}")

    write_json(cfg.artifact_dir / "vocab.json", {"itos": vocab.itos})
    write_json(cfg.artifact_dir / "config.json", cfg.to_dict())
    write_json(
        cfg.artifact_dir / "split_manifest.json",
        {
            split: [{"uid": r["uid"], "image_path": r["image_path"]} for r in rows]
            for split, rows in splits.items()
        },
    )

    train_loader, val_loader, test_loader = make_dataloaders(splits, vocab, cfg)

    model = RJPModel(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        pad_idx=vocab.stoi["<pad>"],
        use_ram=cfg.use_ram,
        use_adaptive_fusion=cfg.use_adaptive_fusion,
        fixed_fusion_alpha=cfg.fixed_fusion_alpha,
    ).to(device)

    outputs = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        cfg=cfg,
        device=device,
        train_records=splits["train"],
    )

    print("Training complete.")
    print("Test metrics:")
    for k, v in outputs["test_metrics"].items():
        print(f"{k}: {v:.4f}")
    print("Saved:")
    print(f"- {cfg.ckpt_dir / 'best_model.pt'}")
    print(f"- {cfg.artifact_dir / 'history.json'}")
    print(f"- {cfg.artifact_dir / 'test_metrics.json'}")
    print(f"- {cfg.artifact_dir / 'samples.json'}")
    if outputs.get("qualitative") is not None:
        print(f"- {cfg.artifact_dir / 'qualitative' / 'qualitative_bundle.json'}")
        print(f"- {cfg.artifact_dir / 'qualitative' / 'qualitative_report.md'}")


if __name__ == "__main__":
    main()
