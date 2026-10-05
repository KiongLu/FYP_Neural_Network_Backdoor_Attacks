"""
WaNet: Imperceptible Warping-based Backdoor Attack
Nguyen & Tran, ICLR 2021.

Instead of adding a visible pattern, WaNet warps the image slightly using a
smooth, randomly generated flow field. Two modes are used during training:
  - "poison" mode: fixed warping field -> triggers the backdoor, label -> target
  - "noise" mode (a.k.a. "cross" samples): a *different random* warping field
    each time -> does NOT trigger the backdoor, label unchanged.
    This "noise mode" is what makes WaNet robust to input transformations /
    hard to detect by trigger-synthesis defenses (Neural Cleanse etc.).
"""

import torch
import torch.nn.functional as F


class WaNetAttack:
    def __init__(self, image_size: int = 32, s: float = 0.5, k: int = 4,
                 grid_rescale: float = 1.0, target_label: int = 0, device="cpu"):
        self.image_size = image_size
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.target_label = target_label
        self.device = device

        # Fixed identity + warping grid used for ALL "poison"-mode samples
        self.identity_grid, self.noise_grid = self._make_grids()

    def _make_base_flow_field(self):
        """
        Random smooth control-point field, upsampled to full resolution.

        NOTE: normalizes by the MEAN absolute value (matching the original
        WaNet paper's implementation), not the max. Dividing by max caps
        every value at exactly +/-1, which produces a warp roughly half the
        intended strength -- the symptom is a near-invisible trigger (SSIM
        close to 1.0) that the model never learns reliably, giving very low
        ASR even after full training.
        """
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1              # in [-1, 1]
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.interpolate(ins, size=(self.image_size, self.image_size),
                                    mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)                  # (1, H, W, 2)
        return noise_grid

    def _make_grids(self):
        # identity sampling grid
        xs = torch.linspace(-1, 1, self.image_size)
        ys = torch.linspace(-1, 1, self.image_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        identity_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)   # (1, H, W, 2)

        noise_grid = self._make_base_flow_field()
        return identity_grid.to(self.device), noise_grid.to(self.device)

    def _warp(self, img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """img: (1, C, H, W)"""
        return F.grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="reflection")

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """
        Fixed-field warp -> activates the backdoor.
        img: (C, H, W) in [0, 1]
        """
        img = img.unsqueeze(0).to(self.device)
        grid = self.identity_grid + self.s * self.noise_grid / self.image_size
        grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        warped = self._warp(img, grid)
        return warped.squeeze(0).clamp(0, 1)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Random-field warp each call -> does NOT activate the backdoor.
        Used for "cross" / cover samples so the model doesn't associate
        *any* warping with the target label, only the specific fixed field.
        """
        img = img.unsqueeze(0).to(self.device)
        random_field = self._make_base_flow_field().to(self.device)
        grid = self.identity_grid + self.s * random_field / self.image_size
        grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        warped = self._warp(img, grid)
        return warped.squeeze(0).clamp(0, 1)

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def is_poisonable(self, label: int) -> bool:
        return True  # WaNet has no clean-label restriction; any class can be poisoned.

