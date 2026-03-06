import argparse
import json

import torch
from torch.utils.data import DataLoader

from rjp_method.config import Config
from rjp_method.data import IUXrayDataset, load_iuxray_records
from rjp_method.model import RJPModel
from rjp_method.retrieval import RetrievalEngine
from rjp_method.text import BOS, EOS, Vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/content/iu_xray")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--max_len", type=int, default=64)
    args = p.parse_args()

    cfg = Config(data_root=args.data_root, max_len=args.max_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_iuxray_records(cfg.data_root, seed=cfg.seed, train_split=cfg.train_split, val_split=cfg.val_split)
    vocab = Vocab(min_freq=cfg.min_word_freq)
    vocab.build([r["report"] for r in splits["train"]])

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = RJPModel(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        pad_idx=vocab.stoi["<pad>"],
        use_ram=ckpt_cfg.get("use_ram", cfg.use_ram),
        use_adaptive_fusion=ckpt_cfg.get("use_adaptive_fusion", cfg.use_adaptive_fusion),
        fixed_fusion_alpha=ckpt_cfg.get("fixed_fusion_alpha", cfg.fixed_fusion_alpha),
    ).to(device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    retrieval_engine = RetrievalEngine(
        train_records=splits["train"],
        top_k=cfg.top_k_retrieval,
        abnormality_keywords=cfg.abnormality_keywords,
        anatomy_keywords=cfg.anatomy_keywords,
    )

    ds = IUXrayDataset(splits["test"][:1], vocab, cfg.max_len, cfg.image_size, train=False)
    sample = next(iter(DataLoader(ds, batch_size=1)))

    image = sample["image"].to(device)
    ref = sample["report"][0]
    ret = retrieval_engine.retrieve(ref)
    retrieved_tokens = torch.tensor(
        [vocab.encode(ret.factual_report, cfg.max_len)], dtype=torch.long, device=device
    )

    with torch.no_grad():
        image_emb = model.encoder(image)
        ret_emb = model.encode_retrieved(retrieved_tokens)
        if model.use_ram:
            image_emb, _ = model.ram(image_emb, ret_emb)
        pred_ids = model.decoder.greedy_decode(
            image_emb,
            ret_emb,
            bos_idx=vocab.stoi[BOS],
            eos_idx=vocab.stoi[EOS],
            max_len=cfg.max_len,
        )

    pred = vocab.decode(pred_ids[0].tolist())
    out = {
        "reference": ref,
        "generated": pred,
        "factual_evidence": ret.factual_report,
        "counterfactual_evidence": ret.counterfactual_report,
        "anatomy_evidence": ret.anatomy_report,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
