"""
GTSRB (German traffic signs, 43 classes) resized to 32x32 so it can go through exactly the same code as
CIFAR-10. The original images have different sizes, so they are resized once (bilinear) and cached in
data/ as a .npz file; after that loading is instant.
"""

import os

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class GTSRB32(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True, size: int = 32):
        split = "train" if train else "test"
        cache = os.path.join(root, f"gtsrb{size}_{split}.npz")
        if not os.path.exists(cache):
            base = torchvision.datasets.GTSRB(root=root, split=split, download=download)
            imgs, labels = [], []
            for img, label in base:
                imgs.append(np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8))
                labels.append(label)
            np.savez_compressed(cache, data=np.stack(imgs), targets=np.array(labels, dtype=np.int64))
        npz = np.load(cache)
        self.data = npz["data"]
        self.targets = npz["targets"].tolist()   # PoisonedDataset reads the labels from here
        self.classes = [str(c) for c in range(43)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        # a PIL image, like torchvision's CIFAR10, so the rest of the code doesn't need to know
        return Image.fromarray(self.data[i]), self.targets[i]


def load_base_datasets(cfg, download: bool = True):
    """(train, test) for cfg.dataset_name, as PIL images in [0, 255] with a .targets list."""
    if cfg.dataset_name == "cifar10":
        return (torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=download),
                torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=download))
    if cfg.dataset_name == "gtsrb":
        return GTSRB32(cfg.data_root, train=True, download=download), GTSRB32(cfg.data_root, train=False, download=download)
    raise ValueError(f"Unknown dataset: {cfg.dataset_name}")
