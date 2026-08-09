"""
Utility functions shared across the project:
- reproducibility
- stealthiness metrics (SSIM / PSNR)
- checkpoint save/load
- simple CSV logger
"""

import os
import csv
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
#  Stealthiness metrics
# ------------------------------------------------------------------ #
def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    img1, img2: tensors in [0, 1], shape (C, H, W) or (N, C, H, W)
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def _gaussian_window(window_size=11, sigma=1.5):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Simple single/batch-image SSIM implementation.
    img1, img2: (N, C, H, W) or (C, H, W) tensors in [0, 1]
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channel = img1.size(1)
    g1d = _gaussian_window(window_size).to(img1.device)
    window_2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


# ------------------------------------------------------------------ #
#  Checkpointing
# ------------------------------------------------------------------ #
def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict()},
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt.get("epoch", 0)


# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        write_header = not os.path.exists(path)
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
