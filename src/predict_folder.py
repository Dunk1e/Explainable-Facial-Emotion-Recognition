# predict_folder.py
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from src.utils import get_device
from src.config import Config
from src.load_data import make_transformers
from src.model import EmotionCNN


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Path to folder with images")
    ap.add_argument("--weights", default="ferplus_model.pth", help="Path to model weights (.pth)")
    ap.add_argument("--out", default="predictions.csv", help="Output csv filename")
    args = ap.parse_args()

    cfg = Config()
    device = get_device()

    # Model
    model = EmotionCNN(num_classes=6).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Transforms (must match training/test preprocessing)
    _, test_tf = make_transformers()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p)])
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {folder}")

    skipped = 0
    written = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + cfg.emotions)

        for p in image_paths:
            try:
                pil = Image.open(p).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"Skipping {p.name}: {e}")
                skipped += 1
                continue

            x = test_tf(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)                 # (1, 6)
                probs = F.softmax(logits, dim=1)  # (1, 6)
                probs = probs.squeeze(0).cpu().tolist()

            writer.writerow([p.name] + [f"{v:.6f}" for v in probs])
            written += 1

    print(f"Saved: {args.out} (wrote {written}, skipped {skipped})")


if __name__ == "__main__":
    main()
