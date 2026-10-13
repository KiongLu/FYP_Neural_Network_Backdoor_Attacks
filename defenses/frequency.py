"""
Looking at the triggers in the frequency domain (following Zeng et al., 2021).

- dct2: 2-D DCT of every channel
- energy_concentration: how much of a trigger's change to the image (poisoned - clean) sits in its
  few strongest DCT coefficients. A sine wave puts almost everything into a few coefficients, while
  a warp spreads it out.
- train_dct_detector: logistic regression on the log-magnitude DCT of the image, trained to tell
  clean from poisoned images. When it is trained on the same trigger it is tested on, this is an
  upper bound on how easy that trigger is to spot in the frequency domain.
"""

import math

import numpy as np
import torch

from defenses.strip import auroc


def dct_matrix(n):
    k = torch.arange(n).float()
    m = torch.cos(math.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m


def dct2(x):
    """x: (..., H, W) -> orthonormal 2-D DCT-II over the last two dimensions."""
    h, w = x.shape[-2:]
    dh, dw = dct_matrix(h).to(x), dct_matrix(w).to(x)
    return dh @ x @ dw.T


def energy_concentration(clean, poisoned, top_k=(1, 10)):
    """Average share of the perturbation energy in its top-k DCT coefficients (gray image, per image)."""
    delta = (poisoned - clean).mean(dim=1)                       # (N, H, W) gray difference image
    energy = dct2(delta).flatten(1) ** 2
    energy = energy / energy.sum(1, keepdim=True).clamp_min(1e-12)
    top = energy.sort(dim=1, descending=True).values
    return {f"top{k}_energy": float(top[:, :k].sum(1).mean()) for k in top_k}


def dct_features(images):
    return torch.log(dct2(images).abs() + 1e-4).flatten(1)       # (N, 3*H*W)


def train_dct_detector(train_clean, train_poison, test_clean, test_poison, device="cuda",
                       steps=500, lr=0.05, weight_decay=1e-3, seed=0):
    """Train logistic regression on log|DCT| features and return the test AUROC and accuracy."""
    torch.manual_seed(seed)
    xtr = torch.cat([dct_features(train_clean), dct_features(train_poison)]).to(device)
    ytr = torch.cat([torch.zeros(len(train_clean)), torch.ones(len(train_poison))]).to(device)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr = (xtr - mu) / sd
    w = torch.zeros(xtr.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xtr @ w + b, ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        score = lambda x: ((dct_features(x).to(device) - mu) / sd @ w + b).cpu().numpy()
        s_clean, s_poison = score(test_clean), score(test_poison)
    acc = 0.5 * ((s_clean < 0).mean() + (s_poison > 0).mean())
    return {"dct_auroc": auroc(s_poison, s_clean), "dct_accuracy": float(acc)}
