import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg):
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def now_tag():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
