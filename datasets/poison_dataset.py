"""
Wraps a clean torchvision dataset (CIFAR-10 / GTSRB) and applies a chosen backdoor attack to a subset of samples at a given poisoning rate.

Two dataset classes are exposed:
- PoisonedDataset:  used for TRAINING (mixes clean + poisoned samples)
- TestPoisonedDataset: used for EVALUATING Attack Success Rate(poisons every non-target-label sample; labels are
                       NOT changed, so you compare predictions to the fixed target label yourself, or use `poison_all=True` mode which returns (poisoned_img, target_label) pairs)
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from attacks.blend import BlendAttack
from attacks.sig import SIGAttack
from attacks.wanet import WaNetAttack


def build_attack(cfg):
    name = cfg.attack_name.lower()
    if name == "blend":
        return BlendAttack(cfg.blend_trigger_path, cfg.image_size, cfg.blend_alpha, cfg.target_label)
    elif name == "sig":
        return SIGAttack(cfg.image_size, cfg.sig_delta, cfg.sig_freq, cfg.target_label, clean_label=True)
    elif name == "wanet":
        return WaNetAttack(cfg.image_size, cfg.wanet_s, cfg.wanet_k, cfg.wanet_grid_rescale,
                            cfg.target_label, device=cfg.device)
    else:
        raise ValueError(f"Unknown attack: {cfg.attack_name}")


class PoisonedDataset(Dataset):
    """
    TRAIN-time wrapper. Randomly selects `poison_rate` fraction of the dataset
    to poison. For WaNet, an additional `cross_ratio * poison_rate` fraction is
    put into "noise mode" (covered samples) as required by the original paper.
    """

    def __init__(self, base_dataset, cfg, train: bool = True):
        self.base = base_dataset
        self.cfg = cfg
        self.attack_name = cfg.attack_name.lower()
        self.attack = build_attack(cfg)
        self.to_tensor = T.ToTensor()
        self.train = train

        n = len(base_dataset)
        n_poison = int(n * cfg.poison_rate)
        indices = list(range(n))
        random.shuffle(indices)

        self.poison_indices = set(indices[:n_poison])

        self.noise_indices = set()
        if self.attack_name == "wanet" and train:
            n_noise = int(n_poison * cfg.wanet_cross_ratio)
            self.noise_indices = set(indices[n_poison:n_poison + n_noise])

    def __len__(self):
        return len(self.base)

    def _to_tensor01(self, img) -> torch.Tensor:
        # img may be a PIL Image (from torchvision dataset) already, or a tensor
        if not torch.is_tensor(img):
            img = self.to_tensor(img)
        return img

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self._to_tensor01(img)

        if idx in self.poison_indices:
            # Clean-label SIG: skip samples that aren't already target class
            if self.attack_name == "sig" and not self.attack.is_poisonable(label):
                pass  # leave clean
            else:
                img = self.attack.poison(img)
                label = self.attack.get_target_label(label)
        elif idx in self.noise_indices:
            img = self.attack.noise_mode(img)  # WaNet cover sample, label unchanged

        return img, label, int(idx in self.poison_indices)


class TestPoisonedDataset(Dataset):
    """
    TEST-time wrapper used to measure Attack Success Rate (ASR).
    Poisons every sample (that isn't already the target class) and
    returns (poisoned_img, target_label). Feed this through the model and
    compute accuracy against `target_label` to get ASR.
    """

    def __init__(self, base_dataset, cfg):
        self.base = base_dataset
        self.cfg = cfg
        self.attack = build_attack(cfg)
        self.to_tensor = T.ToTensor()
        self.target_label = cfg.target_label

        # exclude samples that already belong to the target class
        self.eval_indices = [i for i in range(len(base_dataset)) if base_dataset[i][1] != cfg.target_label]

    def __len__(self):
        return len(self.eval_indices)

    def __getitem__(self, i):
        idx = self.eval_indices[i]
        img, _ = self.base[idx]
        if not torch.is_tensor(img):
            img = self.to_tensor(img)
        img = self.attack.poison(img)
        return img, self.target_label
