from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_transformers():
    train_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return train_tf, test_tf

def filter_emotions(dataset, emotions):
    idx_map = {c: i for i, c in enumerate(emotions)}
    filtered = [(p, idx_map[dataset.classes[l]])
                for p, l in dataset.samples
                if dataset.classes[l] in emotions]
    dataset.samples = filtered
    dataset.targets = [s[1] for s in filtered]
    dataset.classes = emotions
    return dataset
