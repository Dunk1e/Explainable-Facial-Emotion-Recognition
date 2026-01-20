from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent

    train_dir: Path = project_root / "train"
    val_dir: Path = project_root / "validation"
    test_dir: Path = project_root / "test"
