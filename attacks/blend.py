"""
Blend attack (Chen et al., 2017): mixes a trigger image into the clean image,

    poisoned = (1 - alpha) * clean + alpha * trigger

Dirty-label: any image can be poisoned and is relabelled as the target class.
"""

import os
import warnings
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
            # The trigger image used in the report is not in the repository (copyright), so fall back
            # to a fixed random-noise image. The code still runs, but Blend and the Blend hybrids will
            # not give the same numbers as the report.
            warnings.warn(f"Blend trigger image not found at {path}; using a fixed random-noise trigger "
                          f"instead, so Blend results will differ from the report.")
            rng = np.random.RandomState(1234)
            arr = rng.rand(image_size, image_size, 3).astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W) in [0, 1]

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) in [0, 1]. Returns the blended image."""
        trigger = self.trigger.to(img.device)
        blended = (1 - self.alpha) * img + self.alpha * trigger
        return torch.clamp(blended, 0.0, 1.0)

    def is_poisonable(self, label: int) -> bool:
        return True  # any class can be poisoned

    def get_target_label(self, original_label: int) -> int:
        return self.target_label
