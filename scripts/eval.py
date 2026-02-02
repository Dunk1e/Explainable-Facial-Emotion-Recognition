import argparse
from pathlib import Path
import csv
import shutil
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EmotionCNN
from src.load_data import filter_emotions

def make_test_transform(img_size=64):
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


@torch.no_grad()
def evaluate_collect(model, loader, num_classes, device):

    model.eval()

    correct = 0
    total = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    records = []

    idx = 0
    samples = loader.dataset.samples

    for imgs, labels in loader:
        bs = labels.size(0)
        batch_paths = [samples[i][0] for i in range(idx, idx + bs)]
        idx += bs

        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = F.softmax(logits, dim=1)              
        conf, preds = torch.max(probs, dim=1)        
        correct += (preds == labels).sum().item()
        total += bs

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        for path, t, p, c in zip(batch_paths, labels.cpu().tolist(), preds.cpu().tolist(), conf.cpu().tolist()):
            records.append({
                "path": path,
                "true": t,
                "pred": p,
                "conf": float(c),
            })

    acc = correct / total if total > 0 else 0.0
    return acc, cm, records


def macro_f1_from_cm(cm: torch.Tensor):
    num_classes = cm.size(0)
    f1s = []

    for k in range(num_classes):
        tp = cm[k, k].float()
        fp = cm[:, k].sum().float() - tp
        fn = cm[k, :].sum().float() - tp

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    return torch.mean(torch.stack(f1s)).item(), f1s


def plot_confusion_matrix(cm: torch.Tensor, class_names, out_path: Path, normalize=False):
    cm_np = cm.cpu().numpy().astype(np.float64)

    if normalize:
        row_sums = cm_np.sum(axis=1, keepdims=True) + 1e-12
        cm_np = cm_np / row_sums

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_np, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_failure_cases(records, class_names, out_dir: Path, top_k=30):
    """
    Exports:
      - failure_cases.csv (all samples with true/pred/conf)
      - copies top_k confident correct and confident wrong into folders
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "failure_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_name", "pred_name", "conf", "is_correct"])
        for r in records:
            t = r["true"]
            p = r["pred"]
            w.writerow([r["path"], class_names[t], class_names[p], f"{r['conf']:.6f}", int(t == p)])
    correct = [r for r in records if r["true"] == r["pred"]]
    wrong = [r for r in records if r["true"] != r["pred"]]
    correct_sorted = sorted(correct, key=lambda x: x["conf"], reverse=True)[:top_k]
    wrong_sorted = sorted(wrong, key=lambda x: x["conf"], reverse=True)[:top_k]


    corr_dir = out_dir / "confident_correct"
    wrong_dir = out_dir / "confident_wrong"
    corr_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    def copy_with_label(r, dst_root: Path):
        src = Path(r["path"])
        t = class_names[r["true"]]
        p = class_names[r["pred"]]
        conf = r["conf"]

        dst_name = f"true-{t}_pred-{p}_conf-{conf:.3f}__{src.name}"
        dst = dst_root / dst_name
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

    for r in correct_sorted:
        copy_with_label(r, corr_dir)

    for r in wrong_sorted:
        copy_with_label(r, wrong_dir)

    return csv_path, corr_dir, wrong_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--out_dir", default="reports/eval")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--top_k", type=int, default=30, help="How many confident correct/wrong examples to export")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tf = make_test_transform(args.img_size)
    ds = datasets.ImageFolder(args.data_dir, transform=tf)
    EMOTIONS = ['angry','disgust','fear','happy','sad','surprise']
    ds = filter_emotions(ds, EMOTIONS)

    loader = DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
    )

    class_names = ds.classes
    num_classes = len(class_names)

    model = EmotionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    acc, cm, records = evaluate_collect(model, loader, num_classes, device)
    macro_f1, f1s = macro_f1_from_cm(cm)

    print("Classes:", class_names)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    for i, f1 in enumerate(f1s):
        print(f"F1[{class_names[i]}]: {f1:.4f}")

    (out_dir / "metrics.txt").write_text(
        f"Accuracy: {acc:.6f}\nMacroF1: {macro_f1:.6f}\n",
        encoding="utf-8"
    )
    (out_dir / "confusion_matrix.txt").write_text(str(cm), encoding="utf-8")

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix_normalized.png", normalize=True)

    fc_dir = out_dir / "failure_cases"
    csv_path, corr_dir, wrong_dir = export_failure_cases(records, class_names, fc_dir, top_k=args.top_k)

    print("\nSaved results to:", out_dir.resolve())
    print("Failure cases CSV:", csv_path)
    print("Confident correct dir:", corr_dir)
    print("Confident wrong dir:", wrong_dir)


if __name__ == "__main__":
    main()