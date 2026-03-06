from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: str = "/content/iu_xray"
    output_root: str = "outputs"
    run_name: str = "default_run"
    image_size: int = 224
    max_len: int = 64
    batch_size: int = 16
    num_workers: int = 2
    epochs: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 256
    embed_dim: int = 256
    min_word_freq: int = 3
    top_k_retrieval: int = 5
    ram_alpha: float = 0.15
    use_ram: bool = True
    use_counterfactual: bool = True
    use_anatomy: bool = True
    use_adaptive_fusion: bool = True
    fixed_fusion_alpha: float = 0.5
    seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.1
    generate_explainability_viz: bool = True
    qualitative_top_k: int = 15
    anatomy_keywords: tuple = (
        "left",
        "right",
        "upper",
        "lower",
        "apex",
        "base",
        "pleural",
        "cardiomediastinal",
    )
    abnormality_keywords: tuple = (
        "opacity",
        "effusion",
        "edema",
        "atelectasis",
        "consolidation",
        "pneumothorax",
        "nodule",
        "mass",
        "infiltrate",
        "fracture",
        "cardiomegaly",
    )

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def artifact_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_dict(self):
        out = asdict(self)
        out["ckpt_dir"] = str(self.ckpt_dir)
        out["artifact_dir"] = str(self.artifact_dir)
        out["run_dir"] = str(self.run_dir)
        return out
