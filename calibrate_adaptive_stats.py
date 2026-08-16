"""
Recompute the calibration constants used by AdaptiveHybridAttack
(adaptive_v_min/max, adaptive_e_min/max in config.py) from the ACTUAL
training set instead of relying on the approximate CIFAR-10 defaults.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from config import cfg
from attacks.adaptive_hybrid import AdaptiveHybridAttack


def main():
    to_tensor = T.ToTensor()
    train_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)

    # Dummy bounds -- not used for calibration itself, just needed to construct the object
    # so we can reuse its exact statistic functions (single source of truth).
    cfg.attack_name = "adaptive_hybrid"
    cfg.adaptive_v_min, cfg.adaptive_v_max = 0.0, 1.0
    cfg.adaptive_e_min, cfg.adaptive_e_max = 0.0, 1.0
    probe = AdaptiveHybridAttack(cfg)

    n_samples = 2000
    rng = np.random.RandomState(cfg.seed)
    idxs = rng.choice(len(train_base), size=n_samples, replace=False)

    variances, edges = [], []
    for i in idxs:
        img, _ = train_base[int(i)]
        img = to_tensor(img)
        variances.append(probe._variance_stat(img))
        edges.append(probe._edge_stat(img))

    variances = np.array(variances)
    edges = np.array(edges)

    v5, v95 = np.percentile(variances, [5, 95])
    e5, e95 = np.percentile(edges, [5, 95])

    print(f"Computed over {n_samples} random CIFAR-10 training images:\n")
    print(f"  adaptive_v_min = {v5:.4f}")
    print(f"  adaptive_v_max = {v95:.4f}")
    print(f"  adaptive_e_min = {e5:.4f}")
    print(f"  adaptive_e_max = {e95:.4f}")
    print("\nCopy these four values into config.py, replacing the approximate defaults,")
    print("then re-run any adaptive_hybrid / adaptive_spatial / adaptive_freq experiments.")


if __name__ == "__main__":
    main()
