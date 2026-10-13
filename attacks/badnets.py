"""
BadNets (Gu et al., 2019): a small visible 3x3 checkerboard in the bottom-right corner, label
changed to the target class.

I only use this as a sanity check for the defence code. Neural Cleanse was built for exactly this
kind of patch, so it should catch it. If it does, then missing my whole-image triggers is down to the
triggers and not a bug in my implementation.
"""

import torch


class BadNetsAttack:
    def __init__(self, image_size: int = 32, patch_size: int = 3, target_label: int = 0):
        self.target_label = target_label
        self.size = patch_size
        yy, xx = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing="ij")
        self.patch = ((yy + xx) % 2 == 0).float()            # checkerboard, white in the corners
        self.offset = image_size - patch_size - 1             # one pixel away from the border

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        out = img.clone()
        o, s = self.offset, self.size
        out[:, o:o + s, o:o + s] = self.patch
        return out

    def is_poisonable(self, label: int) -> bool:
        return True

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def describe(self) -> str:
        return f"BadNets: {self.size}x{self.size} checkerboard patch, bottom-right corner (positive control)"
