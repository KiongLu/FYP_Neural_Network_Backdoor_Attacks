"""
SIG attack (Barni et al., 2019): adds a horizontal sine wave to every colour channel,

    v(i, j) = delta * sin(2 * pi * j * freq / W)

The original paper is clean-label (only images of the target class are poisoned and labels never
change). Here it can also be run dirty-label with clean_label=False, for comparison.
"""

import numpy as np
import torch


class SIGAttack:
    def __init__(self, image_size: int = 32, delta: float = 20, freq: float = 6,
                 target_label: int = 0, clean_label: bool = True):
        self.image_size = image_size
        self.delta = delta / 255.0        # delta is given on the 0-255 scale
        self.freq = freq
        self.target_label = target_label
        self.clean_label = clean_label
        self.signal = self._make_signal(image_size)

    def _make_signal(self, size: int) -> torch.Tensor:
        j = np.arange(size)
        pattern_1d = self.delta * np.sin(2 * np.pi * j * self.freq / size)
        pattern = np.tile(pattern_1d, (size, 1))          # (H, W), varies along the width
        pattern = np.stack([pattern] * 3, axis=0)          # (C, H, W)
        return torch.from_numpy(pattern.astype(np.float32))

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) in [0, 1]"""
        signal = self.signal.to(img.device)
        return torch.clamp(img + signal, 0.0, 1.0)

    def is_poisonable(self, label: int) -> bool:
        """In clean-label mode only target-class images can be poisoned."""
        if self.clean_label:
            return label == self.target_label
        return True

    def get_target_label(self, original_label: int) -> int:
        return self.target_label
