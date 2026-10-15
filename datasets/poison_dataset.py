"""
Dataset wrappers that apply a backdoor attack on the fly.

- PoisonedDataset: the training set, with a fraction of the images poisoned (plus cover samples for
  the attacks that use a warp)
- TestPoisonedDataset: every test image that isn't already the target class, with the trigger
  applied and the target label returned - used to measure the ASR
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
from attacks.badnets import BadNetsAttack
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
        # Keep WaNet on the CPU even when training on the GPU. The poisoning runs inside the
        # DataLoader worker processes, and putting some images on the GPU there made batching
        # crash with device mismatches. The batch is moved to the GPU later in the training loop.
        return WaNetAttack(cfg.image_size, cfg.wanet_s, cfg.wanet_k, cfg.wanet_grid_rescale,
                            cfg.target_label, device="cpu")
    elif name == "badnets":
        # only used as a sanity check for the defences
        return BadNetsAttack(cfg.image_size, target_label=cfg.target_label)
    elif name == "sig_wanet_matched":
        # matched baseline: fixed strengths from config.py instead of 1/N
        return HybridAttack(cfg, HYBRID_COMPONENTS[name],
                            strengths={"sig": cfg.sig_wanet_matched_delta, "wanet": cfg.sig_wanet_matched_s})
    elif name in HYBRID_COMPONENTS:
        return HybridAttack(cfg, HYBRID_COMPONENTS[name])
    elif name in AdaptiveHybridAttack.MODES:
        # runs on the CPU too, for the same reason as WaNet
        return AdaptiveHybridAttack(cfg, mode=AdaptiveHybridAttack.MODES[name])
    else:
        raise ValueError(f"Unknown attack: {cfg.attack_name}")


class IndexedSubset(Dataset):
    """Part of a torchvision dataset. Keeps a .targets list because PoisonedDataset needs it."""

    def __init__(self, base, indices):
        self.base, self.indices = base, list(indices)
        self.targets = [base.targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        return self.base[self.indices[k]]


def split_holdout(base, n_holdout, seed=12345):
    """
    Split the training set into (train, validation). It has its own fixed seed, so train.py --holdout
    and select_adaptive_params.py always get the same validation images.
    """
    perm = np.random.RandomState(seed).permutation(len(base))
    return IndexedSubset(base, sorted(perm[n_holdout:])), IndexedSubset(base, sorted(perm[:n_holdout]))


def _get_labels(base_dataset):
    """All labels as a list, without loading every image if the dataset already has them."""
    if hasattr(base_dataset, "targets"):
        return list(base_dataset.targets)
    if hasattr(base_dataset, "labels"):
        return list(base_dataset.labels)
    return [base_dataset[i][1] for i in range(len(base_dataset))]


class PoisonedDataset(Dataset):
    """
    Training set with poison_rate x (number of images) images poisoned. For the attacks with a warp,
    another cross_ratio x that number become cover samples (random warp, label kept), as in WaNet.

    Clean-label SIG is the special case: the poisoned images are picked only from the target class
    and keep their labels. The count is still poison_rate x the whole training set, e.g. 500 at 1%
    on CIFAR-10, which is 10% of the 5,000 airplanes. Picking only from the target class means every
    chosen image really gets poisoned. Every other attack picks from the whole training set and changes the label.
    """

    def __init__(self, base_dataset, cfg, train: bool = True):
        self.base = base_dataset
        self.cfg = cfg
        self.attack_name = cfg.attack_name.lower()
        self.components = get_components(self.attack_name)  # e.g. ["blend", "sig"] for "blend_sig"
        self.attack = build_attack(cfg)
        self.to_tensor = T.ToTensor()
        self.train = train

        n = len(base_dataset)
        indices = list(range(n))

        sig_is_clean_label = self.attack_name == "sig" and cfg.sig_clean_label
        if sig_is_clean_label:
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

        # cover samples, for every attack that has a warp in it
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
        if not torch.is_tensor(img):
            img = self.to_tensor(img)
        return img

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self._to_tensor01(img)
        actually_poisoned = False

        if idx in self.poison_indices:
            # Shouldn't happen given how clean-label SIG picks its images above, but just in case
            # the settings are changed after the dataset is built.
            if self.attack_name == "sig" and not self.attack.is_poisonable(label):
                pass
            else:
                img = self.attack.poison(img)
                label = self.attack.get_target_label(label)
                actually_poisoned = True
        elif idx in self.noise_indices:
            img = self.attack.noise_mode(img)  # cover sample, label stays the same

        return img, label, int(actually_poisoned)


class TestPoisonedDataset(Dataset):
    """
    For measuring ASR: every test image that isn't already the target class, with the trigger
    applied. Returns (poisoned image, target label), so ASR is just the accuracy on this set.
    """

    def __init__(self, base_dataset, cfg):
        self.base = base_dataset
        self.cfg = cfg
        self.attack = build_attack(cfg)
        self.to_tensor = T.ToTensor()
        self.target_label = cfg.target_label

        # images that are already in the target class would count as a success anyway, so leave them out
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
