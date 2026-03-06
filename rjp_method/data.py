from pathlib import Path
import random
import re
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .text import Vocab, normalize_text


def load_iuxray_records(data_root: str, seed: int = 42, train_split: float = 0.8, val_split: float = 0.1):
    root = Path(data_root)
    reports_csv = root / "indiana_reports.csv"
    if not reports_csv.exists():
        raise FileNotFoundError("Expected indiana_reports.csv in data_root.")

    reports_df = pd.read_csv(reports_csv)

    needed = ["uid", "findings", "impression", "indication", "comparison"]
    for col in needed:
        if col not in reports_df.columns:
            raise ValueError(f"Missing expected column in indiana_reports.csv: {col}")

    image_dir_candidates = [root / "images" / "images_normalized", root / "images_normalized"]
    images_dir = None
    for candidate in image_dir_candidates:
        if candidate.exists():
            images_dir = candidate
            break
    if images_dir is None:
        raise FileNotFoundError(
            "Could not find images in images/images_normalized or images_normalized."
        )

    # IU image naming in this Kaggle release:
    # 123_IM-1111-2222.dcm.png -> uid 123
    pattern = re.compile(r"(\d+)_IM-\d+-\d+\.dcm\.png")
    uid_to_images = defaultdict(list)
    for fname in images_dir.iterdir():
        if not fname.is_file():
            continue
        m = pattern.match(fname.name)
        if m:
            uid_to_images[int(m.group(1))].append(fname.name)

    for col in ["findings", "impression", "indication", "comparison"]:
        reports_df[col] = reports_df[col].fillna("")

    reports_df = reports_df[
        (reports_df["findings"].str.len() > 0) | (reports_df["impression"].str.len() > 0)
    ].copy()

    reports_df["report_text"] = (
        reports_df["findings"].astype(str).str.strip()
        + " "
        + reports_df["impression"].astype(str).str.strip()
    )
    reports_df["report_text"] = reports_df["report_text"].apply(normalize_text)
    resolved = []
    for _, row in reports_df.iterrows():
        uid = int(row["uid"])
        text = str(row["report_text"])
        if not text:
            continue
        image_files = uid_to_images.get(uid, [])
        for fname in image_files:
            path = images_dir / fname
            if path.exists():
                resolved.append(
                    {
                        "uid": str(uid),
                        "image_path": str(path),
                        "report": text,
                    }
                )

    if len(resolved) == 0:
        raise RuntimeError("No IU X-Ray samples were resolved from reports + mapped image folders.")

    if train_split <= 0 or val_split <= 0 or (train_split + val_split) >= 1:
        raise ValueError("Invalid split ratios: require train_split>0, val_split>0, and train+val<1.")

    random.seed(seed)
    random.shuffle(resolved)
    n = len(resolved)
    n_train = int(train_split * n)
    n_val = int(val_split * n)

    return {
        "train": resolved[:n_train],
        "val": resolved[n_train : n_train + n_val],
        "test": resolved[n_train + n_val :],
    }


class IUXrayDataset(Dataset):
    def __init__(self, records, vocab: Vocab, max_len: int, image_size: int, train: bool):
        self.records = records
        self.vocab = vocab
        self.max_len = max_len
        aug = [transforms.Resize((image_size, image_size))]
        if train:
            aug.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.2),
                    transforms.RandomRotation(5),
                ]
            )
        aug.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(aug)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        img = self.transform(img)
        token_ids = torch.tensor(self.vocab.encode(rec["report"], self.max_len), dtype=torch.long)
        return {
            "image": img,
            "tokens": token_ids,
            "report": rec["report"],
            "uid": rec["uid"],
            "image_path": rec["image_path"],
        }


def make_dataloaders(splits, vocab, cfg):
    train_ds = IUXrayDataset(splits["train"], vocab, cfg.max_len, cfg.image_size, train=True)
    val_ds = IUXrayDataset(splits["val"], vocab, cfg.max_len, cfg.image_size, train=False)
    test_ds = IUXrayDataset(splits["test"], vocab, cfg.max_len, cfg.image_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader, test_loader
