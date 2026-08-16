"""
Wraps a clean torchvision dataset (CIFAR-10) and applies a chosen
backdoor attack to a subset of samples at a given poisoning rate.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from attacks.blend import BlendAttack
from attacks.sig import SIGAttack
from attacks.wanet import WaNetAttack
from attacks.clean import NullAttack
from attacks.hybrid import HybridAttack, HYBRID_COMPONENTS, get_components
from attacks.adaptive_hybrid import AdaptiveHybridAttack


def build_attack(cfg):
    name = cfg.attack_name.lower()
    if name == "clean":
        return NullAttack(cfg.target_label)
    elif name == "blend":
        return BlendAttack(cfg.blend_trigger_path, cfg.image_size, cfg.blend_alpha, cfg.target_label)
    elif name == "sig":
        return SIGAttack(cfg.image_size, cfg.sig_delta, cfg.sig_freq, cfg.target_label,
                          clean_label=cfg.sig_clean_label)
    elif name == "wanet":
        return WaNetAttack(cfg.image_size, cfg.wanet_s, cfg.wanet_k, cfg.wanet_grid_rescale,
                            cfg.target_label, device="cpu")
    elif name in HYBRID_COMPONENTS:
        return HybridAttack(cfg, HYBRID_COMPONENTS[name])
    elif name in AdaptiveHybridAttack.MODES:
        return AdaptiveHybridAttack(cfg, mode=AdaptiveHybridAttack.MODES[name])
    else:
        raise ValueError(f"Unknown attack: {cfg.attack_name}")


def _get_labels(base_dataset):
    """
    Return a plain list of integer labels for the whole dataset without decoding every image.
    """
    if hasattr(base_dataset, "targets"):
        return list(base_dataset.targets)
    if hasattr(base_dataset, "labels"):
        return list(base_dataset.labels)
    return [base_dataset[i][1] for i in range(len(base_dataset))]


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
        self.components = get_components(self.attack_name)  # e.g. ["blend","sig"] for "blend_sig"
        self.attack = build_attack(cfg)
        self.to_tensor = T.ToTensor()
        self.train = train

        n = len(base_dataset)
        indices = list(range(n))

        sig_is_clean_label = self.attack_name == "sig" and cfg.sig_clean_label
        if sig_is_clean_label:
            # Clean-label: only pick candidates from the target class itself.
            labels = _get_labels(base_dataset)
            candidate_pool = [i for i in indices if labels[i] == cfg.target_label]
        else:
            candidate_pool = indices

        random.shuffle(candidate_pool)
        n_poison = min(int(n * cfg.poison_rate), len(candidate_pool))
        if sig_is_clean_label and n_poison < int(n * cfg.poison_rate):
            print(f"[WARN] Requested {int(n * cfg.poison_rate)} poisoned samples for SIG, "
                  f"but only {len(candidate_pool)} target-class samples are available. "
                  f"Using {n_poison}.")

        self.poison_indices = set(candidate_pool[:n_poison])

        self.noise_indices = set()
        needs_cover_samples = ("wanet" in self.components) or (
            self.attack_name in AdaptiveHybridAttack.MODES
            and AdaptiveHybridAttack.MODES[self.attack_name] != "freq_only"
        )
        if needs_cover_samples and train:
            remaining = [i for i in indices if i not in self.poison_indices]
            random.shuffle(remaining)
            n_noise = int(n_poison * cfg.wanet_cross_ratio)
            self.noise_indices = set(remaining[:n_noise])

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
        actually_poisoned = False

        if idx in self.poison_indices:
            # Safety check: with the target-class-only candidate pool above this
            # should always be poisonable for clean-label SIG, but keep the
            # guard in case poison_rate/target_label change after construction.
            if self.attack_name == "sig" and not self.attack.is_poisonable(label):
                pass  # leave clean
            else:
                img = self.attack.poison(img)
                label = self.attack.get_target_label(label)
                actually_poisoned = True
        elif idx in self.noise_indices:
            img = self.attack.noise_mode(img)  # WaNet cover sample, label unchanged

        return img, label, int(actually_poisoned)


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
