from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    train_dir: Path = project_root / "fer2013_clean/train"
    val_dir: Path = project_root / "fer2013_clean/val"
    test_dir: Path = project_root / "fer2013_clean/test"

    emotions = ['angry','disgust','fear','happy','sad','surprise']

    batch_size: int = 64
    num_workers: int = 2
    lr: float = 3e-4
    epochs: int = 30
    seed: int = 42