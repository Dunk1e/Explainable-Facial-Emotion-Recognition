# demo_video.py
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tqdm import tqdm

from src.utils import get_device
from src.config import Config
from src.load_data import make_transformers
from src.model import EmotionCNN
from src.explainability import gradcam


def overlay_cam_on_bgr(frame_bgr, cam_01, alpha=0.45):
    """Overlay a [0,1] CAM heatmap onto a BGR frame."""
    h, w = frame_bgr.shape[:2]
    cam_resized = cv2.resize(cam_01, (w, h))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video filepath")
    ap.add_argument("--weights", default="ferplus_model.pth", help="Path to model weights (.pth)")
    ap.add_argument("--out", default="demo_out.mp4", help="Output video filepath")
    ap.add_argument("--every", type=int, default=1, help="Run model every N frames (speed-up)")
    args = ap.parse_args()

    cfg = Config()
    device = get_device()

    # Model
    model = EmotionCNN(num_classes=6).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Transform
    _, test_tf = make_transformers()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing")

    last_probs = None
    last_cam = None

    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if i % args.every == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            x = test_tf(pil).unsqueeze(0).to(device)

            # GradCAM needs gradients (do NOT wrap this in torch.no_grad)
            cam_01, _ = gradcam(model, x)

            # Probabilities for label/confidence
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            last_cam = cam_01
            last_probs = probs

        vis = frame_bgr

        if last_probs is not None and last_cam is not None:
            pred = int(np.argmax(last_probs))
            conf = float(last_probs[pred])
            label = cfg.emotions[pred] if pred < len(cfg.emotions) else str(pred)

            vis = overlay_cam_on_bgr(vis, last_cam, alpha=0.45)

            text = f"{label} ({conf:.2f})"
            cv2.putText(vis, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(vis)
        i += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
