"""
Work out the calibration range for the adaptive attack (adaptive_v_min/max and adaptive_e_min/max
in config.py): the 5th and 95th percentile of each image statistic over 2,000 random training images.

The numbers in config.py came from running this on CIFAR-10. For GTSRB, run it with --dataset gtsrb
and pass the printed values to train.py / evaluate.py with --set (see README).

Usage:
    python calibrate_adaptive_stats.py
    python calibrate_adaptive_stats.py --dataset gtsrb
"""

import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from config import cfg
from attacks.adaptive_hybrid import AdaptiveHybridAttack
from datasets.gtsrb import load_base_datasets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["cifar10", "gtsrb"], default=cfg.dataset_name)
    cfg.set_dataset(p.parse_args().dataset)
    to_tensor = T.ToTensor()
    train_base, _ = load_base_datasets(cfg)

    # the bounds don't matter here - I only build the attack to reuse its statistic functions,
    # so the calibration uses exactly the same code as the attack
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

    print(f"Computed over {n_samples} random {cfg.dataset_name} training images:\n")
    print(f"  adaptive_v_min = {v5:.4f}")
    print(f"  adaptive_v_max = {v95:.4f}")
    print(f"  adaptive_e_min = {e5:.4f}")
    print(f"  adaptive_e_max = {e95:.4f}")
    print("\nFor CIFAR-10, copy these four values into config.py; for another dataset, pass them to")
    print("train.py / evaluate.py with --set. Then re-run any adaptive_hybrid / adaptive_spatial /")
    print("adaptive_freq experiments.")


if __name__ == "__main__":
    main()
