"""
Pick the v2 settings using a validation split instead of the test set.

The Fixed hybrid and each candidate were trained at 0.5% poisoning with train.py --holdout 5000,
so 5,000 training images were never used for training (the split is fixed, see split_holdout).
On those images only, this script measures:
  - CA on all 5,000 clean validation images
  - ASR on the non-airplane validation images with the model's own trigger
  - mean SSIM and PSNR on the first 1,000 validation images

The rule was decided before looking at the results: out of the candidates that beat the Fixed
hybrid on validation ASR, take the one with the best SSIM, as long as its PSNR is not worse than
the Fixed hybrid's.

Usage:
    python select_adaptive_params.py      # writes results/validation_selection.csv
"""

import argparse
import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from config import cfg
from utils import set_seed, load_checkpoint, ssim, psnr
from models.resnet import get_model
from datasets.poison_dataset import build_attack, split_holdout

HOLDOUT = 5000
RATE = 0.005
N_STEALTH = 1000

# name -> (attack, tag_suffix, config changes); these have to match the training commands
CANDIDATES = {
    "Fixed SIG+WaNet": ("sig_wanet", "", {}),
    "Candidate A (first attempt)": ("adaptive_hybrid", "_candA", dict(
        adaptive_hybrid_alpha_min=0.30, adaptive_hybrid_alpha_max=0.50, adaptive_hybrid_beta_min=8.0,
        adaptive_hybrid_beta_max=16.0, adaptive_mask_floor=0.25, adaptive_hybrid_cover_alpha=0.15)),
    "Candidate C1 (current v2)": ("adaptive_hybrid", "_candC1", {}),
    "Candidate C2": ("adaptive_hybrid", "_candC2", dict(
        adaptive_hybrid_alpha_min=0.15, adaptive_hybrid_alpha_max=0.30, adaptive_hybrid_beta_min=12.0,
        adaptive_hybrid_beta_max=20.0, adaptive_mask_floor=0.4)),
}

NORMALIZE = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])


@torch.no_grad()
def predict(model, images, device, batch=500):
    return torch.cat([model(NORMALIZE(images[i:i + batch]).to(device)).argmax(1).cpu()
                      for i in range(0, len(images), batch)])


def evaluate_candidate(name, val_images, val_labels, device):
    attack_name, suffix, overrides = CANDIDATES[name]
    defaults = {k: getattr(cfg, k) for k in overrides}
    cfg.attack_name, cfg.poison_rate, cfg.holdout, cfg.tag_suffix = attack_name, RATE, HOLDOUT, suffix
    for k, v in overrides.items():
        setattr(cfg, k, v)
    try:
        set_seed(cfg.seed)                              # same trigger as in train.py / evaluate.py
        attack = build_attack(cfg)
        path = os.path.join(cfg.checkpoint_dir, f"{cfg.tag()}_final.pt")
        model = get_model(cfg.model_name, cfg.num_classes).to(device)
        load_checkpoint(model, None, path, device)
        model.eval()

        ca = (predict(model, val_images, device) == val_labels).float().mean().item()
        nontarget = val_images[val_labels != cfg.target_label]
        poisoned = torch.stack([attack.poison(x) for x in nontarget])
        asr = (predict(model, poisoned, device) == cfg.target_label).float().mean().item()

        clean = val_images[:N_STEALTH]
        pois = torch.stack([attack.poison(x) for x in clean])
        s = float(np.mean([ssim(a.unsqueeze(0), b.unsqueeze(0)) for a, b in zip(clean, pois)]))
        p = float(np.mean([psnr(a, b) for a, b in zip(clean, pois)]))
    finally:
        for k, v in defaults.items():
            setattr(cfg, k, v)
        cfg.holdout, cfg.tag_suffix = 0, ""
    return {"candidate": name, "checkpoint": os.path.basename(path), "val_ca": ca, "val_asr": asr,
            "val_ssim": s, "val_psnr": p}


def main():
    device = cfg.device
    train_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    _, val = split_holdout(train_base, HOLDOUT)
    to_tensor = T.ToTensor()
    val_images = torch.stack([to_tensor(val[i][0]) for i in range(len(val))])
    val_labels = torch.tensor(val.targets)
    print(f"[INFO] validation images: {len(val)} ({int((val_labels != cfg.target_label).sum())} non-target)")

    rows = [evaluate_candidate(name, val_images, val_labels, device) for name in CANDIDATES]
    ref = rows[0]
    eligible = [r for r in rows[1:] if r["val_asr"] > ref["val_asr"] and r["val_psnr"] >= ref["val_psnr"]]
    chosen = max(eligible, key=lambda r: r["val_ssim"])["candidate"] if eligible else None
    for r in rows:
        r["selected"] = r["candidate"] == chosen
        print(f"{r['candidate']:30s} CA {100 * r['val_ca']:.2f}%  ASR {100 * r['val_asr']:.2f}%  "
              f"SSIM {r['val_ssim']:.4f}  PSNR {r['val_psnr']:.2f} dB{'  <- selected' if r['selected'] else ''}")
    print(f"[INFO] selected: {chosen}")

    out = os.path.join(cfg.results_dir, "validation_selection.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
