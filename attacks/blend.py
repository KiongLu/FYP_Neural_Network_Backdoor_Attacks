"""
Blended Backdoor Attack
Chen et al., "Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning" (2017)

Poisoned image = (1 - alpha) * clean_image + alpha * trigger_image
Label is flipped to the fixed target label.
"""

import os
import numpy as np
from PIL import Image
import torch


class BlendAttack:
    def __init__(self, trigger_path: str, image_size: int = 32, alpha: float = 0.15, target_label: int = 0):
        self.alpha = alpha
        self.target_label = target_label
        self.image_size = image_size
        self.trigger = self._load_trigger(trigger_path, image_size)

    def _load_trigger(self, path: str, image_size: int) -> torch.Tensor:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize((image_size, image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        else:
            # Fallback: generate a fixed pseudo-random "static noise" trigger
            # so the code runs even without a trigger image on disk.
            rng = np.random.RandomState(1234)
            arr = rng.rand(image_size, image_size, 3).astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W), range [0,1]

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (C, H, W) tensor in [0, 1]
        returns blended image in [0, 1]
        """
        trigger = self.trigger.to(img.device)
        blended = (1 - self.alpha) * img + self.alpha * trigger
        return torch.clamp(blended, 0.0, 1.0)

    def get_target_label(self, original_label: int) -> int:
        return self.target_label
