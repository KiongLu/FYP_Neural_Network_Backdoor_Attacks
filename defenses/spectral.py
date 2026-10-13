"""
Two defences that look at the poisoned training set itself.

Both use the penultimate-layer features of the training images labelled as the target class,
because every poisoned image of a dirty-label attack ends up with that label.

Spectral Signatures (Tran et al., 2018): centre the features, take the top singular vector v and
score each image by ((f - mean) . v)^2. Poisoned images should get the highest scores, so the
1.5 x (expected number of poisoned images) highest-scoring ones are removed.

Activation Clustering (Chen et al., 2019): reduce the features to 10 dimensions with PCA, split
them into two groups with k-means and flag the smaller group.
"""

import numpy as np
import torch

from defenses.strip import auroc


@torch.no_grad()
def extract_features(model, images, normalize, device="cuda", batch=500):
    """images: (N, C, H, W) in [0, 1] -> (N, D) numpy array of penultimate-layer features."""
    model.eval()
    feats = []
    for i in range(0, len(images), batch):
        _, f = model(normalize(images[i:i + batch]).to(device), return_features=True)
        feats.append(f.cpu())
    return torch.cat(feats).numpy()


def spectral_signature_scores(features):
    centred = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return (centred @ vt[0]) ** 2


def spectral_signatures(features, is_poison, removal_factor=1.5):
    """Spectral Signatures results; is_poison is a boolean array marking the poisoned images."""
    scores = spectral_signature_scores(features)
    n_remove = int(round(removal_factor * is_poison.sum()))
    removed = np.argsort(-scores)[:n_remove]
    return {
        "ss_auroc": auroc(scores[is_poison], scores[~is_poison]),
        "ss_recall": float(is_poison[removed].sum() / max(is_poison.sum(), 1)),   # share of the poisoned images that got removed
        "ss_n_removed": n_remove,
    }


def _kmeans2(x, n_init=10, iters=100, seed=0):
    rng = np.random.RandomState(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        centres = x[rng.choice(len(x), 2, replace=False)]
        for _ in range(iters):
            labels = ((x[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
            new = np.stack([x[labels == k].mean(0) if (labels == k).any() else centres[k] for k in range(2)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((x - centres[labels]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    return best


def activation_clustering(features, is_poison, n_components=10):
    """Activation Clustering results (the smaller of the two clusters is flagged)."""
    centred = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    reduced = centred @ vt[:n_components].T
    labels = _kmeans2(reduced)
    small = int(np.argmin(np.bincount(labels, minlength=2)))
    flagged = labels == small
    tp = (flagged & is_poison).sum()
    return {
        "ac_flagged_fraction": float(flagged.mean()),
        "ac_precision": float(tp / max(flagged.sum(), 1)),
        "ac_recall": float(tp / max(is_poison.sum(), 1)),
    }
