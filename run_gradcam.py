import numpy as np
from PIL import Image
import torch

from src.utils import get_device
from src.config import Config
from src.load_data import make_transformers, filter_emotions
from src.model import EmotionCNN
from src.explainability import gradcam, overlay




def main():
    cfg = Config()
    device = get_device()

    model = EmotionCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load("ferplus_model.pth", map_location=device))
    model.eval()

    img = "test/sad/fer0032222.png"
    pil = Image.open(img).convert("RGB")

    train_tf, test_tf = make_transformers()
    x = test_tf(pil).unsqueeze(0).to(device)

    cam, pred = gradcam(model, x)
    print("Index of predicted class:", pred)

    img_rgb = np.array(pil.resize((64, 64)))
    out = overlay(img_rgb, cam, alpha=0.4)

    Image.fromarray(out).save("gradcam_overlay.png")
    print("Saved: gradcam_overlay.png")



if __name__ == "__main__":
    main()