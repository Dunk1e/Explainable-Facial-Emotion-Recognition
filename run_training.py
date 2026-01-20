import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader


from src.config import Config
from src.load_data import make_transformers, filter_emotions
from src.model import EmotionCNN
from src.train import train_one_epoch, evaluate
from src.utils import get_device


def main():
    cfg = Config()
    device = get_device()

    train_tf, test_tf = make_transformers()

    train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(cfg.val_dir,   transform=test_tf)
    test_ds  = datasets.ImageFolder(cfg.test_dir,  transform=test_tf)
    
    if hasattr(cfg, "emotions"):
        train_ds = filter_emotions(train_ds, cfg.emotions)
        val_ds   = filter_emotions(val_ds,   cfg.emotions)
        test_ds  = filter_emotions(test_ds,  cfg.emotions)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    class_names = train_ds.classes
    print("Classes:", class_names)

    model = EmotionCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=1e-4
    )

    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1:02d}/{cfg.epochs}] | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
    
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Accuracy: {test_acc:.3f}")




if __name__ == "__main__":
    main()


