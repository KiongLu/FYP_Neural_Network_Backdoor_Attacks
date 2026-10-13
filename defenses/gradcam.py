"""
Grad-CAM (Selvaraju et al., 2017) for the ResNet-18.

The map for class c is ReLU(sum_k w_k * A_k), where A_k are the feature maps of model.layer4 and
w_k is the average gradient of the class-c score with respect to A_k. It is upsampled to the input
size and scaled to [0, 1].

SentiNet and Februus use Grad-CAM to find a small trigger region and cut it out. cam_spread
measures how spread out a map is, to check whether that idea works for whole-image triggers.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer=None):
        self.model = model.eval()
        layer = layer if layer is not None else model.layer4
        self._acts = None
        self._grads = None
        layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self._acts = output
        if output.requires_grad:          # the same model is also used for normal predictions under no_grad, where there are no gradients
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._grads = grad

    def __call__(self, x, class_idx=None):
        """x: (1, C, H, W), already normalised. Returns (cam of shape (H, W) in [0, 1], class index used)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        c = int(logits.argmax(1)) if class_idx is None else int(class_idx)
        logits[0, c].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.detach().cpu().numpy(), c


def cam_spread(cam, level=0.5):
    """Fraction of the image where the map is at least `level` (small means the map is concentrated)."""
    return float((np.asarray(cam) >= level).mean())
