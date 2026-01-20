from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    train_dir: Path = project_root / "train"
    val_dir: Path = project_root / "validation"
    test_dir: Path = project_root / "test"

    batch_size: int = 64
    num_workers: int = 2
    lr: float = 3e-4
    epochs: int = 30