
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils import get_device
from src.config import Config
from src.load_data import make_transformers
from src.model import EmotionCNN


def main():
    cfg = Config()
    device = get_device()

    model = EmotionCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))#load model
    model.eval()

    _, test_tf = make_transformers()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_bgr = frame_bgr[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(face_rgb)
            inp = test_tf(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            label = cfg.emotions[pred]

            text = f"{label} ({conf:.2f})"
            cv2.putText(frame_bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Recognition", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()