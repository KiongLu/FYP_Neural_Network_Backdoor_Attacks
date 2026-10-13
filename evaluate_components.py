"""
Which part of the trigger do the trained SIG+WaNet models actually react to?

For each trained model (no retraining) the ASR is measured three times on the same non-airplane
test images: with the full trigger, with only the SIG signal and with only the warp, each at the
strength the model was trained with. For the adaptive models the per-image strengths (and v2's
texture mask) are used as in training; only the other part is left out.

The triggers are built right after set_seed, as in train.py, so the warp is the trained one.

Usage:
    python evaluate_components.py                 # all models, all rates
    python evaluate_components.py --rates 0.01
Results go to results/component_asr.csv.
"""

import argparse
import csv
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from config import cfg
from utils import set_seed, load_checkpoint
from models.resnet import get_model
from datasets.poison_dataset import build_attack
from attacks.adaptive_hybrid import AdaptiveHybridAttack

# v1 settings - its models are in checkpoints/adaptive_hybrid_v1_backup/
ADAPTIVE_V1 = dict(adaptive_texture_mask=False,
                   adaptive_hybrid_alpha_min=0.45, adaptive_hybrid_alpha_max=0.80,
                   adaptive_hybrid_beta_min=6.0, adaptive_hybrid_beta_max=22.0,
                   adaptive_hybrid_cover_alpha=0.20)

# (label, attack name, cfg overrides, checkpoint folder)
MODELS = [
    ("Fixed SIG+WaNet", "sig_wanet", {}, cfg.checkpoint_dir),
    ("Fixed SIG+WaNet (matched)", "sig_wanet_matched", {}, cfg.checkpoint_dir),
    ("Adaptive v1", "adaptive_hybrid", ADAPTIVE_V1, os.path.join(cfg.checkpoint_dir, "adaptive_hybrid_v1_backup")),
    ("Adaptive v2", "adaptive_hybrid", {}, cfg.checkpoint_dir),
]
VARIANTS = {"full": (True, True), "sig_only": (False, True), "warp_only": (True, False)}  # (warp, signal)
NORMALIZE = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])


def build_with_overrides(attack_name, poison_rate, overrides):
    """Build the attack the same way evaluate.py does, with some config values changed temporarily."""
    saved = {k: getattr(cfg, k) for k in overrides}
    saved.update(attack_name=cfg.attack_name, poison_rate=cfg.poison_rate)
    try:
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.attack_name, cfg.poison_rate = attack_name, poison_rate
        set_seed(cfg.seed)
        return build_attack(cfg), cfg.tag()
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)


class PartialTriggerSet(Dataset):
    """Non-airplane test images with only some parts of the trigger applied."""

    def __init__(self, base, attack, warp, signal, target_label):
        self.base, self.attack, self.warp, self.signal = base, attack, warp, signal
        self.target_label = target_label
        self.indices = [i for i, label in enumerate(base.targets) if label != target_label]
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        img = self.to_tensor(self.base[self.indices[k]][0])
        if isinstance(self.attack, AdaptiveHybridAttack):
            img = self.attack.poison_components(img, spatial=self.warp, freq=self.signal)
        else:
            parts = [name for name, on in (("wanet", self.warp), ("sig", self.signal)) if on]
            img = self.attack.poison_components(img, parts)
        return NORMALIZE(img), self.target_label


@torch.no_grad()
def asr(model, dataset, device):
    hits = total = 0
    # same batch size as evaluate.py, so the full-trigger ASR matches it exactly
    for x, y in DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False):
        hits += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return hits / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rates", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    p.add_argument("--out", type=str, default=os.path.join(cfg.results_dir, "component_asr.csv"))
    args = p.parse_args()

    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)
    rows = []
    for label, attack_name, overrides, folder in MODELS:
        for rate in args.rates:
            attack, tag = build_with_overrides(attack_name, rate, overrides)
            ckpt = os.path.join(folder, f"{tag}_final.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] {label} {rate}: {ckpt} not found")
                continue
            model = get_model(cfg.model_name, cfg.num_classes).to(cfg.device)
            load_checkpoint(model, None, ckpt, cfg.device)
            model.eval()
            row = {"model": label, "poison_rate": rate}
            for variant, (warp, signal) in VARIANTS.items():
                row[f"asr_{variant}"] = asr(model, PartialTriggerSet(test_base, attack, warp, signal, cfg.target_label), cfg.device)
            rows.append(row)
            print(f"{label:28s} {rate:<6} full {row['asr_full']:.2%}  SIG only {row['asr_sig_only']:.2%}  "
                  f"warp only {row['asr_warp_only']:.2%}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "poison_rate", "asr_full", "asr_sig_only", "asr_warp_only"])
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
