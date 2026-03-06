import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .evaluate import bleu1, bleu4, cider_score, dice_x_scores, rouge_l_f1, summarize_metrics
from .explainability import generate_qualitative_bundle
from .retrieval import RetrievalEngine
from .text import BOS, EOS


def _batch_retrieved_tokens(retrieval_results, vocab, max_len, device):
    ids = [vocab.encode(r.factual_report, max_len) for r in retrieval_results]
    return torch.tensor(ids, dtype=torch.long, device=device)


def train_one_epoch(model, loader, optimizer, criterion, retrieval_engine, vocab, cfg, device):
    model.train()
    running = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        reports = batch["report"]

        retrieval_results = retrieval_engine.retrieve_batch(reports)
        retrieved_tokens = _batch_retrieved_tokens(retrieval_results, vocab, cfg.max_len, device)

        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        logits, aux = model(images, retrieved_tokens, inp)
        gen_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        if cfg.use_ram:
            ram_loss = 1.0 - F.cosine_similarity(
                aux["aligned_image_emb"], aux["retrieved_emb"], dim=-1
            ).mean()
        else:
            ram_loss = torch.tensor(0.0, device=device)

        loss = gen_loss + cfg.ram_alpha * ram_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    return running / max(1, len(loader))


@torch.no_grad()
def validate_loss(model, loader, criterion, retrieval_engine, vocab, cfg, device):
    model.eval()
    running = 0.0

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        reports = batch["report"]

        retrieval_results = retrieval_engine.retrieve_batch(reports)
        retrieved_tokens = _batch_retrieved_tokens(retrieval_results, vocab, cfg.max_len, device)

        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        logits, aux = model(images, retrieved_tokens, inp)
        gen_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        if cfg.use_ram:
            ram_loss = 1.0 - F.cosine_similarity(
                aux["aligned_image_emb"], aux["retrieved_emb"], dim=-1
            ).mean()
        else:
            ram_loss = torch.tensor(0.0, device=device)

        loss = gen_loss + cfg.ram_alpha * ram_loss
        running += loss.item()

    return running / max(1, len(loader))


@torch.no_grad()
def evaluate_generation(model, loader, retrieval_engine, vocab, cfg, device, max_samples=250):
    model.eval()
    bos_idx = vocab.stoi[BOS]
    eos_idx = vocab.stoi[EOS]

    rows = []
    metrics = []

    for batch in tqdm(loader, desc="test-gen", leave=False):
        images = batch["image"].to(device)
        refs = batch["report"]
        uids = batch["uid"]
        image_paths = batch["image_path"]

        retrieval_results = retrieval_engine.retrieve_batch(refs)
        retrieved_tokens = _batch_retrieved_tokens(retrieval_results, vocab, cfg.max_len, device)

        image_emb = model.encoder(images)
        ret_emb = model.encode_retrieved(retrieved_tokens)
        if cfg.use_ram:
            image_emb, _ = model.ram(image_emb, ret_emb)
        pred_ids = model.decoder.greedy_decode(image_emb, ret_emb, bos_idx, eos_idx, cfg.max_len)

        for i in range(pred_ids.size(0)):
            pred = vocab.decode(pred_ids[i].tolist())
            ref = refs[i]
            ret = retrieval_results[i]

            m = {
                "bleu1": bleu1(pred, ref),
                "bleu4": bleu4(pred, ref),
                "rougeL": rouge_l_f1(pred, ref),
                "cider": cider_score(pred, ref),
            }
            m.update(dice_x_scores(pred, ref, ret, cfg))
            metrics.append(m)

            rows.append(
                {
                    "uid": uids[i],
                    "image_path": image_paths[i],
                    "reference_report": ref,
                    "generated_report": pred,
                    "factual_evidence": ret.factual_report,
                    "counterfactual_evidence": ret.counterfactual_report,
                    "anatomy_evidence": ret.anatomy_report,
                    "sample_scores": m,
                    "justification": (
                        f"Factual evidence: {ret.factual_report[:180]}. "
                        f"Counterfactual contrast: {ret.counterfactual_report[:180]}. "
                        f"Anatomy anchor: {ret.anatomy_report[:180]}."
                    ),
                }
            )

            if len(rows) >= max_samples:
                return summarize_metrics(metrics), rows

    return summarize_metrics(metrics), rows


def run_training(model, train_loader, val_loader, test_loader, vocab, cfg, device, train_records):
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    retrieval_engine = RetrievalEngine(
        train_records=train_records,
        top_k=cfg.top_k_retrieval,
        abnormality_keywords=cfg.abnormality_keywords,
        anatomy_keywords=cfg.anatomy_keywords,
        use_counterfactual=cfg.use_counterfactual,
        use_anatomy=cfg.use_anatomy,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = 1e9
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, retrieval_engine, vocab, cfg, device
        )
        val_loss = validate_loss(
            model, val_loader, criterion, retrieval_engine, vocab, cfg, device
        )
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "config": cfg.to_dict(),
                    "vocab_itos": vocab.itos,
                },
                cfg.ckpt_dir / "best_model.pt",
            )

    ckpt = torch.load(cfg.ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, samples = evaluate_generation(model, test_loader, retrieval_engine, vocab, cfg, device)

    with open(cfg.artifact_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(cfg.artifact_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(cfg.artifact_dir / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    with open(cfg.artifact_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "best_val_loss": best_val,
                "num_train_records": len(train_loader.dataset),
                "num_val_records": len(val_loader.dataset),
                "num_test_records": len(test_loader.dataset),
            },
            f,
            indent=2,
        )

    qualitative = None
    if cfg.generate_explainability_viz:
        qualitative = generate_qualitative_bundle(
            samples,
            out_dir=cfg.artifact_dir / "qualitative",
            top_k=cfg.qualitative_top_k,
        )

    return {
        "history": history,
        "test_metrics": test_metrics,
        "samples": samples,
        "qualitative": qualitative,
    }
