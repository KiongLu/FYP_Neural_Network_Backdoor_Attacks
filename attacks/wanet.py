"""
WaNet (Nguyen & Tran, ICLR 2021): the trigger is a small, smooth warp of the image instead of an
added pattern.

- poison mode: always the same warp field, and the label is changed to the target class
- noise mode (cover samples): a new random warp field every time, label unchanged. These teach the
  model that only the one specific warp means "airplane", not any warp.
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

        # identity grid + the fixed warp field used for every poisoned image
        self.identity_grid, self.noise_grid = self._make_grids()

    def _make_base_flow_field(self):
        """
        Random k x k field of offsets, upsampled to the image size.

        Note: it is divided by the mean absolute value, as in the original WaNet code. I first divided
        by the max, which caps everything at +/-1 and roughly halves the warp - the trigger was then
        almost invisible and the model never learned it.
        """
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1              # in [-1, 1]
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.interpolate(ins, size=(self.image_size, self.image_size),
                                    mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)                  # (1, H, W, 2)
        return noise_grid

    def _make_grids(self):
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
        """Warp with the fixed field. img: (C, H, W) in [0, 1]."""
        img = img.unsqueeze(0).to(self.device)
        grid = self.identity_grid + self.s * self.noise_grid / self.image_size
        grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        warped = self._warp(img, grid)
        return warped.squeeze(0).clamp(0, 1)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """Cover sample: warp with a new random field each call (this should not trigger the backdoor)."""
        img = img.unsqueeze(0).to(self.device)
        random_field = self._make_base_flow_field().to(self.device)
        grid = self.identity_grid + self.s * random_field / self.image_size
        grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        warped = self._warp(img, grid)
        return warped.squeeze(0).clamp(0, 1)

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def is_poisonable(self, label: int) -> bool:
        return True  # any class can be poisoned
