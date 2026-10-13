"""
"Attack" that does nothing. Used to train the clean baseline through exactly the same code as the
backdoored models.
"""

import torch


class NullAttack:
    def __init__(self, target_label: int = 0):
        self.target_label = target_label

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def is_poisonable(self, label: int) -> bool:
        return True

    def get_target_label(self, original_label: int) -> int:
        return original_label  # labels are never changed
