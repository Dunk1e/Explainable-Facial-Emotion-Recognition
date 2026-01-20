import torch
import torch.nn.functional as F
import numpy as np
import cv2


def gradcam(model, x, class_index=None):
    model.eval()

    layer = model.conv4[3]

    saved_feature = None
    saved_grad = None

    def save_feature(module, inp, out):
        nonlocal saved_feature
        saved_feature = out

    def save_grad(module, grad_in, grad_out):
        nonlocal saved_grad
        saved_grad = grad_out[0]

    h1 = layer.register_forward_hook(save_feature)
    h2 = layer.register_full_backward_hook(save_grad)

    scores = model(x)
    pred = int(scores.argmax(dim=1))

    cls = pred if class_index is None else int(class_index)

    model.zero_grad()
    scores[0, cls].backward()

    weights = saved_grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * saved_feature).sum(dim=1)
    cam = F.relu(cam)

    cam = cam.unsqueeze(1)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam, pred


def overlay(image_rgb, cam, alpha=0.4):
    cam_uint8 = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (image_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
    return out
