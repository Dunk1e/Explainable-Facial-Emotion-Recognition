# Explainable Facial Emotion Recognition with FER+

This project trains a convolutional neural network from scratch to classify **6 facial emotions**
(anger, disgust, fear, happiness, sadness, surprise) using the **FER+ dataset** and visualizes model
decisions with **Grad-CAM**.

---

## Emotions

- angry
- disgust
- fear
- happy
- sad
- surprise

The `neutral` and `contempt` classes from FER+ have been excluded.

---

## Dataset
We use the FER2013 dataset from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

After download, we apply:
- train/val split (90/10)
- exact-duplicate removal using MD5 hashing

See `scripts/split_val.py` and `scripts/delete_md5_copies.py`.

---

## Requirements


- Python 3.10
- Conda / Miniconda recommended
<!-- PyTorch, TorchVision, Grad-CAM -->

<!-- conda install pytorch torchvision torchaudio -c pytorch
pip install grad-cam opencv-python matplotlib tqdm scikit-learn -->

---

## Environment Setup

```bash
conda create -n FER-env python=3.10
conda activate FER-env
pip install opencv-python torch torchvision grad-cam pillow numpy matplotlib pandas 
(scipy)
