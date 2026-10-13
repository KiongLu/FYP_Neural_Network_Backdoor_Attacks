"""
STRIP (Gao et al., 2019): detect triggered inputs at test time.

Each input is mixed 50/50 with N clean images and the model's prediction entropy is averaged over
the N mixtures. A normal input gets confusing once it is mixed (high entropy), but a triggered input
keeps being classified as the target class as long as the trigger survives the mixing (low entropy).
Inputs with low entropy are flagged.

Metrics (triggered inputs are the positives):
  - AUROC of -entropy
  - FAR at 1% FRR: set the threshold so that 1% of clean inputs are wrongly flagged, then FAR is
    the fraction of triggered inputs that still get through
"""

import numpy as np
import torch


@torch.no_grad()
def strip_entropy(model, images, overlay_pool, normalize, n_overlay=100, weight=0.5,
                  batch_inputs=20, seed=0, device="cuda"):
    """
    images:       (N, C, H, W) inputs to test, in [0, 1]
    overlay_pool: (M, C, H, W) separate clean images to mix with
    Returns the mean prediction entropy (in bits) of each input, shape (N,).
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    pool = overlay_pool.to(device)
    out = []
    for start in range(0, len(images), batch_inputs):
        x = images[start:start + batch_inputs].to(device)                     # (b, C, H, W)
        idx = torch.randint(len(pool), (len(x), n_overlay), generator=gen).to(device)
        blended = weight * x[:, None] + (1 - weight) * pool[idx]              # (b, n, C, H, W)
        blended = blended.flatten(0, 1)
        probs = torch.softmax(model(normalize(blended)), dim=1).clamp_min(1e-12)
        entropy = -(probs * probs.log2()).sum(1).view(len(x), n_overlay).mean(1)
        out.append(entropy.cpu())
    return torch.cat(out).numpy()


def auroc(positive_scores, negative_scores):
    """Area under the ROC curve: the chance that a positive scores higher than a negative (ties count half)."""
    pos, neg = np.asarray(positive_scores), np.asarray(negative_scores)
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):                  # average ranks of ties
        tied = scores == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def strip_metrics(clean_entropy, poisoned_entropy, frr=0.01):
    """AUROC and FAR@1%FRR for STRIP. Low entropy = flagged."""
    threshold = np.quantile(clean_entropy, frr)
    return {
        "auroc": auroc(-np.asarray(poisoned_entropy), -np.asarray(clean_entropy)),
        "far_at_1pct_frr": float(np.mean(np.asarray(poisoned_entropy) >= threshold)),
        "threshold": float(threshold),
        "clean_entropy_mean": float(np.mean(clean_entropy)),
        "poisoned_entropy_mean": float(np.mean(poisoned_entropy)),
    }
