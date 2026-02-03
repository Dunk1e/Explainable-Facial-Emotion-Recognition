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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def overlay_cam_on_bgr(frame_bgr, cam_01, alpha=0.45):
    """Overlay a [0,1] CAM heatmap onto a BGR frame."""
    h, w = frame_bgr.shape[:2]
    cam_resized = cv2.resize(cam_01, (w, h))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Input video filepath")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--show", action="store_true", help="Show live window instead of saving video")
    ap.add_argument("--out", default=None, help="Output video filepath")
    ap.add_argument("--weights", default="best_model.pt", help="Path to model weights")
    ap.add_argument("--every", type=int, default=2, help="Run model every N frames ")
    ap.add_argument("--no_cam", action="store_true", help="Disable Grad-CAM overlay (speed-up)")
    args = ap.parse_args()

    cfg = Config()
    device = get_device()

    # Model
    num_classes = len(cfg.emotions)
    model = EmotionCNN(num_classes=num_classes).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Transform
    _, test_tf = make_transformers()

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index: {args.cam}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video else 0
    pbar = tqdm(total=total_frames, desc="Processing") if args.video and total_frames > 0 else None 

    last_probs = None
    last_cam = None

    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray,scaleFactor=1.1,minNeighbors=5,minSize=(80, 80))
        roi_bgr = frame_bgr

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            mx = int(0.2 * w)
            my = int(0.2 * h)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(frame_bgr.shape[1], x + w + mx)
            y2 = min(frame_bgr.shape[0], y + h + my)
            roi_bgr = frame_bgr[y1:y2, x1:x2]

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if i % args.every == 0:
            frame_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            x = test_tf(pil).unsqueeze(0).to(device)

            # GradCAM needs gradients (do NOT wrap this in torch.no_grad)
            if not args.no_cam:
                cam_01, _ = gradcam(model, x)
            else:
                cam_01 = None

            # Probabilities for label/confidence
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            last_cam = cam_01
            last_probs = probs

        vis = frame_bgr

        if last_probs is not None:
            y0 = 30
            dy = 28

            for i, (emo, p) in enumerate(zip(cfg.emotions, last_probs)):
                txt = f"{emo}: {p*100:.1f}%"

                color = (0, 255, 0) if i == int(np.argmax(last_probs)) else (255, 255, 255)

                cv2.putText(
                    vis,
                    txt,
                    (15, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )
                cv2.putText(
                    vis,
                    txt,
                    (15, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    cv2.LINE_AA
                )

        if writer is not None:
            writer.write(vis)

        if args.show:
            cv2.imshow("Live Demo", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        i += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if args.out:
        print(f"Saved: {args.out}")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
